"""CPU protocol tests: fake byte transport, real tensors/allocator/patch tracker."""

import ctypes
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.elastic_pp.kv_patch import KVPatchBusy
from sglang.srt.elastic_pp.live_kv_patch import LiveKVSnapshot
from sglang.srt.elastic_pp.nixl_kv_patch import NixlKVPatchCopier, NixlKVReceiver
from sglang.srt.elastic_pp.nixl_transport import NixlCompletion, NixlMigrationEndpoint
from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class Fabric:
    def __init__(self):
        self.agents, self.lock = {}, threading.Lock()


class Agent:
    def __init__(self, fabric, name):
        self.fabric, self.name = fabric, name
        self.messages = defaultdict(list)
        self.released, self.deregistered = 0, 0
        self.fail_cancel = False
        fabric.agents[name] = self

    def register_memory(self, rows, kind):
        return rows

    def deregister_memory(self, rows):
        self.deregistered += 1

    def get_agent_metadata(self):
        return self.name.encode()

    def add_remote_agent(self, metadata):
        return metadata  # Matches the native loadRemoteMD -> py::bytes binding.

    def remove_remote_agent(self, name):
        pass

    def send_notif(self, name, message):
        with self.fabric.lock:
            self.fabric.agents[name].messages[self.name].append(message)

    def get_new_notifs(self):
        with self.fabric.lock:
            result, self.messages = dict(self.messages), defaultdict(list)
        return result

    def get_xfer_descs(self, rows, kind):
        return rows

    def initialize_xfer(self, op, src, dst, name, notification):
        assert op == "WRITE"
        return src, dst, name, notification

    def transfer(self, handle):
        src, dst, name, notification = handle
        for source, target in zip(src, dst, strict=True):
            assert source[1] == target[1]
            ctypes.memmove(target[0], source[0], source[1])
        self.send_notif(name, notification)
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        if self.fail_cancel:
            raise RuntimeError("native cancellation failed")
        self.released += 1


class Pool:
    size, page_size = 8, 1

    def __init__(self, layers=(3,)):
        self.pairs = {
            l: (torch.full((9, 1, 2), -1.0), torch.full((9, 1, 1), -1.0))
            for l in layers
        }
        self.before_apply = lambda: None

    def get_elastic_pp_pending_kv_buffer(self, *, generation, layer_id):
        assert generation == 1
        return self.pairs[layer_id]

    def import_elastic_pp_kv_slots(
        self, *, generation, layer_id, slot_ids, cache_k, cache_v
    ):
        self.before_apply()
        for target, values in zip(
            self.pairs[layer_id], (cache_k, cache_v), strict=True
        ):
            target.index_copy_(0, slot_ids, values)


@pytest.fixture
def endpoints():
    fabric = Fabric()
    endpoints = [
        NixlMigrationEndpoint(
            rank=i,
            world_size=3,
            device="cpu",
            buffer_bytes=4096,
            timeout=3,
            agent=Agent(fabric, str(i)),
        )
        for i in range(3)
    ]
    metadata = [e.metadata() for e in endpoints]
    for endpoint in endpoints:
        endpoint.connect(metadata)
    yield endpoints
    for endpoint in endpoints:
        if not endpoint._stranded:
            endpoint.close()


def ready():
    return SimpleNamespace(synchronize=lambda: None)


def setup_patch(endpoints, *, peers=(1,), layers=(3,), max_rows=2):
    allocator = TokenToKVPoolAllocator(8, torch.float32, "cpu", object(), False)
    slots = allocator.alloc(3)
    source = {
        l: (
            torch.arange(18).view(9, 1, 2).float(),
            torch.arange(9).view(9, 1, 1).float(),
        )
        for l in layers
    }
    receivers, pools = {}, {}
    for peer in peers:
        pools[peer] = Pool(layers)
        receivers[peer] = NixlKVReceiver(endpoints[peer], pools[peer])
        receivers[peer].configure(1, {0: layers})
        receivers[peer].ready(1)
    copier = NixlKVPatchCopier(
        source,
        endpoint=endpoints[0],
        generation=1,
        targets={p: layers for p in peers},
        max_rows=max_rows,
    )
    patch = LiveKVSnapshot(
        generation=1,
        allocator=allocator,
        source=source,
        max_rows=max_rows,
        copier=copier,
        admit_host_bytes=lambda _: pytest.fail("unexpected CPU KV shadow"),
    )
    deadline = time.monotonic() + 2
    while not copier.ready_to_submit and time.monotonic() < deadline:
        time.sleep(0.001)
    assert copier.ready_to_submit
    return allocator, source, slots, patch, receivers, pools


def test_all_candidates_preinitialize_and_prewarm_even_without_layers(endpoints):
    for endpoint in endpoints:
        endpoint.warmup()
        assert endpoint.metrics["warm_channels"] == 2
        assert endpoint.metrics["bytes_sent"] == 16
        assert endpoint.handler is None


@pytest.mark.parametrize("peers", [(1,), (1, 2)])
def test_real_allocator_patch_applies_online_and_seals_exact_coverage(endpoints, peers):
    allocator, source, slots, patch, receivers, pools = setup_patch(
        endpoints, peers=peers
    )
    patch.progress(slots[:2], ready())
    patch.session.poll(generation=1, wait=True)
    for receiver in receivers.values():
        assert receiver.stats["online_applied_rows"] == 2
    # The newly written token was not present in the first data copy.
    source[3][0][slots[2]] = 99
    patch.progress(slots, ready())
    for receiver in receivers.values():
        receiver.mark_fenced()
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    for peer in peers:
        assert receivers[peer].finish(1) == {3}
        for actual, expected in zip(pools[peer].pairs[3], source[3], strict=True):
            assert torch.equal(actual[slots], expected[slots])
    assert patch.copier.shadow_bytes == 0
    assert allocator._elastic_pp_live_patch is None


def test_source_slot_remains_pinned_until_application_ack_not_network_done(endpoints):
    allocator, source, slots, patch, receivers, pools = setup_patch(endpoints)
    entered, allow = threading.Event(), threading.Event()
    pools[1].before_apply = lambda: (entered.set(), allow.wait(timeout=2))
    patch.progress(slots[:1], ready())
    assert entered.wait(timeout=2)
    try:
        assert not patch.session.poll(generation=1)
        with pytest.raises(KVPatchBusy):
            patch.session.free(slots[0].item(), 1, generation=1)
    finally:
        allow.set()
    patch.session.poll(generation=1, wait=True)
    patch.abort()
    allocator.free(slots)


def test_free_and_in_place_rewrite_retire_old_incarnations(endpoints):
    allocator, source, slots, patch, receivers, pools = setup_patch(endpoints)
    patch.progress(slots, ready())
    # A physical free waits for the in-flight network/application ACK.
    allocator.free(slots[:1])
    patch.prepare_writes(slots[1:2])
    source[3][0][slots[1]] = 71
    patch.progress(slots[1:], ready())
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    assert slots[0].item() not in receivers[1].versions[0]
    assert receivers[1].versions[0][slots[1].item()] == 2
    assert pools[1].pairs[3][0][slots[1]].tolist() == [[71, 71]]
    assert receivers[1].finish(1) == {3}


def test_target_rejects_bad_shape_before_modifying_any_layer_and_abort_drains(
    endpoints,
):
    allocator, source, slots, patch, receivers, pools = setup_patch(
        endpoints, layers=(3, 5)
    )
    original = endpoints[1].handler

    def corrupt(peer, message, data):
        if message.get("op") == "data":
            message["segments"][-1][3][-1] = 999
        return original(peer, message, data)

    endpoints[1].handler = corrupt
    patch.progress(slots, ready())
    deadline = time.monotonic() + 2
    while not patch.session._pending[1].future.done() and time.monotonic() < deadline:
        time.sleep(0.001)
    patch.progress(slots, ready())
    assert "shape/dtype" in patch.error
    assert allocator._elastic_pp_live_patch is None
    for pair in pools[1].pairs.values():
        assert all(torch.all(t == -1) for t in pair)
    allocator.free(slots)


def test_seal_without_complete_coverage_and_late_generation_are_rejected(endpoints):
    allocator, source, slots, patch, receivers, pools = setup_patch(endpoints)
    receiver = receivers[1]
    with pytest.raises(ValueError, match="coverage"):
        receiver.receive(
            0, dict(op="seal", generation=1, count=1, signature="bad"), torch.empty(0)
        )
    receiver.retire()
    with pytest.raises(ValueError, match="matching prepared"):
        receiver.receive(
            0, dict(op="forget", generation=1, rows=[[1, 1]]), torch.empty(0)
        )
    patch.abort()


def test_failed_native_cancel_retains_registered_buffers(endpoints):
    endpoint = endpoints[0]
    endpoint.agent.fail_cancel = True
    with pytest.raises(RuntimeError, match="native cancellation"):
        endpoint.warmup()
    with pytest.raises(RuntimeError, match="registration remains live"):
        endpoint.close()
    assert endpoint.agent.deregistered == 0


def test_empty_notification_buckets_do_not_create_a_sender(endpoints, monkeypatch):
    original = Agent.get_new_notifs

    def notifications(agent):
        result = original(agent)
        result["empty-backend-bucket"] = []
        return result

    monkeypatch.setattr(Agent, "get_new_notifs", notifications)
    endpoints[0].warmup()
    assert endpoints[0].metrics["warm_channels"] == 2


def test_binary_notification_agent_names_match_imported_metadata(
    endpoints, monkeypatch
):
    original = Agent.get_new_notifs

    def notifications(agent):
        return {name.encode(): messages for name, messages in original(agent).items()}

    monkeypatch.setattr(Agent, "get_new_notifs", notifications)
    for endpoint in endpoints:
        endpoint.warmup()
        assert endpoint.metrics["warm_channels"] == 2


def test_abort_requires_a_confirmed_source_stream_drain():
    future = Future()
    completion = NixlCompletion(future, 1)
    future.set_exception(RuntimeError("CUDA stream could not drain"))
    with pytest.raises(RuntimeError, match="could not drain"):
        completion.drain()
    future._nixl_source_drained = True
    completion.drain()


def test_hybrid_global_layer_mapping_writes_pending_attention_only(
    endpoints, hybrid_pool_factory
):
    pool = hybrid_pool_factory(start_layer=2, end_layer=4)
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(0, 4), generation=1, total_num_layers=4
    )
    receiver = NixlKVReceiver(endpoints[1], pool)
    receiver.configure(1, {0: (1,)})
    receiver.ready(1)
    allocator = TokenToKVPoolAllocator(8, torch.float32, "cpu", object(), False)
    slots = allocator.alloc(2)
    source = {1: (torch.full((9, 1, 2), 51.0), torch.full((9, 1, 2), 72.0))}
    copier = NixlKVPatchCopier(
        source, endpoint=endpoints[0], generation=1, targets={1: (1,)}, max_rows=2
    )
    patch = LiveKVSnapshot(
        generation=1,
        allocator=allocator,
        source=source,
        max_rows=2,
        copier=copier,
        admit_host_bytes=lambda _: None,
    )
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    k, v = pool.full_kv_pool.get_elastic_pp_pending_kv_buffer(generation=1, layer_id=0)
    assert torch.all(k[slots] == 51) and torch.all(v[slots] == 72)
    assert receiver.finish(1) == {1}
