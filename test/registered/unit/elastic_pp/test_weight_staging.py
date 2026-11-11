from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationCoordinator
from sglang.srt.elastic_pp.layer_transport import ElasticPPLayerTransport
from sglang.srt.elastic_pp.topology import (
    DynamicPPTransition,
    LayerRange,
    LayerTransfer,
    PPTopology,
)
from sglang.srt.elastic_pp.weight_staging import (
    PinnedWeightStager,
    WeightPacketPlan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Mailbox:
    def __init__(self):
        self.queue = Queue()
        self.waits = 0
        self.corrupt = False

    def send_tensor_dict(self, payload, *, dst, async_send):
        assert dst == 1 and async_send
        mailbox = self

        class DeferredSend:
            def wait(self):
                mailbox.waits += 1
                # Read only on completion, so early host-buffer reuse corrupts
                # the byte-for-byte round trip below.
                mailbox.queue.put(
                    {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in payload.items()
                    }
                )

        return [DeferredSend()]

    def recv_tensor_dict_into(self, buffers, *, src, metadata_validator):
        assert src == 0
        payload = self.queue.get_nowait()
        if self.corrupt:
            header_key = next(k for k, v in payload.items() if isinstance(v, dict))
            payload[header_key] = dict(payload[header_key], offset=-1)
        metadata_validator(list(payload.items()))
        for key, target in buffers.items():
            assert target.shape == payload[key].shape
            target.copy_(payload[key])
        return dict(payload, **buffers)


def plan(state, generation=1):
    return WeightPacketPlan(
        generation=generation,
        transfer=LayerTransfer(0, 1, LayerRange(1, 2)),
        layer_states={1: state},
    )


@pytest.mark.parametrize("chunk_bytes", [1, 7, 32, 128])
def test_chunked_mixed_dtype_dense_transpose_writes_target_storage(chunk_bytes):
    src = {
        "weight": torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).t(),
        "scale": torch.tensor([1.25, -3.5]),
        "counter": torch.tensor(913, dtype=torch.int64),
        "empty": torch.empty(0),
    }
    dst = {key: torch.empty_like(value) for key, value in src.items()}
    addresses = {key: value.data_ptr() for key, value in dst.items()}
    mailbox = Mailbox()
    sender = PinnedWeightStager(device="cpu", stream=None, chunk_bytes=chunk_bytes)
    receiver = PinnedWeightStager(device="cpu", stream=None, chunk_bytes=chunk_bytes)
    sender.send(
        plan=plan(src), group=mailbox, destination=1, wait=ElasticPPLayerTransport.wait
    )
    receiver.receive(plan=plan(dst), group=mailbox, source=0)
    assert all(torch.equal(src[k], dst[k]) for k in src)
    assert addresses == {k: v.data_ptr() for k, v in dst.items()}
    assert sender.buffer.numel() == receiver.buffer.numel() == chunk_bytes
    assert mailbox.waits == sender.metrics["packets_sent"]
    assert sender.metrics["bytes_sent"] == receiver.metrics["bytes_received"]


@pytest.mark.parametrize("mismatch", ["generation", "shape", "dtype", "packet"])
def test_invalid_header_rejected_before_destination_write(mismatch):
    mailbox = Mailbox()
    src = {"weight": torch.arange(12).float()}
    dst = {"weight": torch.full((12,), -1.0)}
    if mismatch == "shape":
        dst["weight"] = dst["weight"].reshape(3, 4)
    if mismatch == "dtype":
        dst["weight"] = dst["weight"].to(torch.int32)
    sender = PinnedWeightStager(device="cpu", stream=None, chunk_bytes=128)
    receiver = PinnedWeightStager(device="cpu", stream=None, chunk_bytes=128)
    sender.send(
        plan=plan(src), group=mailbox, destination=1, wait=ElasticPPLayerTransport.wait
    )
    mailbox.corrupt = mismatch == "packet"
    with pytest.raises(ValueError, match="layout/generation"):
        receiver.receive(
            plan=plan(dst, generation=2 if mismatch == "generation" else 1),
            group=mailbox,
            source=0,
        )
    assert (dst["weight"] == -1).all()


def test_stager_drains_cuda_reader_on_partial_copy_failure():
    events = []

    class BrokenTensor:
        def copy_(self, *args, **kwargs):
            events.append("copy")
            raise RuntimeError("injected partial H2D")

    class BrokenPlan:
        header = {}
        total_bytes = 4

        def segments(self, offset, size):
            yield BrokenTensor(), 0, size

    class Group:
        def recv_tensor_dict_into(self, *args, **kwargs):
            return {}

    stager = PinnedWeightStager(device="cpu", stream=None, chunk_bytes=4)
    stager._context = nullcontext
    stager._drain = lambda: events.append("drain")
    with pytest.raises(RuntimeError, match="partial H2D"):
        stager.receive(plan=BrokenPlan(), group=Group(), source=0)
    assert events == ["copy", "drain"]
    assert stager.metrics["bytes_received"] == 0


def test_cuda_stager_requires_explicit_stream_before_allocating():
    with pytest.raises(ValueError, match="own migration stream"):
        PinnedWeightStager(device="cuda:0", stream=None)


def test_plan_rejects_non_dense_storage_without_gpu_repacking():
    with pytest.raises(ValueError, match="dense full-storage"):
        plan({"weight": torch.arange(64).reshape(8, 8)[:, ::2]})


def test_coordinator_marks_loaded_only_after_all_packets():
    source = PPTopology.from_layer_counts((2, 0), generation=0)
    target = PPTopology.from_layer_counts((1, 1), generation=1)
    transition = DynamicPPTransition.build(source=source, target=target)
    tensors = [{"weight": torch.arange(15).float()}, {"weight": torch.zeros(15)}]
    mailbox = Mailbox()
    loaded = []

    def make(rank):
        runner = SimpleNamespace(
            device="cpu",
            server_args=SimpleNamespace(elastic_pp_weight_staging="pinned"),
            export_elastic_pp_layer_state=lambda ids: {1: tensors[rank]},
            mark_elastic_pp_layers_loaded=lambda **kwargs: loaded.append(kwargs),
        )
        layers = SimpleNamespace(
            _migration_stream=None,
            _sort_key=ElasticPPLayerTransport._sort_key,
            wait=ElasticPPLayerTransport.wait,
        )
        coordinator = ElasticPPMigrationCoordinator(
            rank=rank,
            current_topology=source,
            group=mailbox,
            model_runner=runner,
            layer_transport=layers,
            kv_transport=None,
            strategy="background",
        )
        coordinator._weight_stager = PinnedWeightStager(
            device="cpu", stream=None, chunk_bytes=7
        )
        return coordinator

    sender, receiver = make(0), make(1)
    sender._precopy_immutable_weights(transition)
    assert not loaded
    receiver._precopy_immutable_weights(transition)
    assert torch.equal(tensors[0]["weight"], tensors[1]["weight"])
    assert loaded == [{"generation": 1, "layer_ids": {1}}]
    assert sender._background_weight_bytes_sent == 60
