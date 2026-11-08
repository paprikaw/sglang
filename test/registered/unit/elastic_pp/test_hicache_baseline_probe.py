"""CPU checks for the private probe's actual method and address boundaries."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def probe():
    source = ROOT / "research/pp_elastic/hicache_baseline_probe.py"
    spec = importlib.util.spec_from_file_location("baseline_probe_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def production_method(path, cls, method, namespace):
    tree = ast.parse((ROOT / path).read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == cls
    )
    function = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    for argument in (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ):
        argument.annotation = None
    function.returns = None
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    return namespace[method]


@pytest.fixture
def installed(probe, tmp_path):
    collector = probe.Collector(tmp_path, 0)
    calls = []

    class Scheduler:
        def _record_batch_launch(self, batch):
            calls.append(("launch", batch))

    class Base:
        pass

    Base._track_mamba_state_extend = production_method(
        "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
        "MambaAttnBackendBase",
        "_track_mamba_state_extend",
        {"torch": torch},
    )

    class GDN(Base):
        def forward_extend(self, layer, forward_batch, h, state, metadata):
            calls.append(("extend", layer, forward_batch))
            self._track_mamba_state_extend(forward_batch, h, state, metadata)
            return h

    class Dispatcher:
        def extend(self, q, k, v, g, beta, **kwargs):
            calls.append(("kernel", q, kwargs))
            return q, None, q.new_zeros(1, 9, 1, 2)

    class ReqPool:
        pass

    class Runner:
        pass

    real_cow = production_method(
        "python/sglang/srt/model_executor/model_runner.py",
        "ModelRunner",
        "_maybe_execute_deferred_mamba_cow_and_clear",
        {"HybridReqToTokenPool": ReqPool},
    )

    def counted_cow(self, batch):
        calls.append(("cow", batch))
        return real_cow(self, batch)

    Runner._maybe_execute_deferred_mamba_cow_and_clear = counted_cow

    class Sampler:
        def _preprocess_logits(self, logits, sampling_info):
            calls.append(("preprocess", logits, sampling_info))
            return logits + 1

    class Cache:
        def cache_finished_req(self, req, *args, **kwargs):
            calls.append(("finish", req))

        def match_prefix(self, params):
            calls.append(("match", params))
            return NS(full_kv_hit_length=512, best_match_node=9)

    probe.install_observers(
        Scheduler,
        GDN,
        Dispatcher,
        Base,
        Runner,
        Sampler,
        Cache,
        ReqPool,
        get_collector=lambda: collector,
        get_chunk_size=lambda: 64,
    )
    return NS(
        collector=collector,
        calls=calls,
        Scheduler=Scheduler,
        GDN=GDN,
        Dispatcher=Dispatcher,
        ReqPool=ReqPool,
        Runner=Runner,
        Sampler=Sampler,
        Cache=Cache,
    )


def req(probe, rid):
    return NS(
        origin_input_ids=probe.ACTIVE_PROMPT, rid=rid, output_ids=[], mamba_pool_idx=1
    )


def events(installed):
    path = installed.collector.directory / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_raw_513_metadata_produces_real_512_checkpoint_birth(probe, installed):
    from sglang.srt.layers.attention import hybrid_linear_attn_backend as backend

    # Exercise the production index builder too: raw513 maps to h[8], while
    # the hook records a derived absolute512 boundary and retains raw513.
    original_chunk = backend.mamba_cache_chunk_size
    backend.mamba_cache_chunk_size = lambda: 64
    try:
        batch = NS(
            mamba_track_mask=torch.tensor([True]),
            extend_seq_lens=torch.tensor([513]),
            mamba_track_indices=torch.tensor([2]),
            mamba_track_seqlens=torch.tensor([513]),
            extend_prefix_lens=torch.tensor([0]),
        )
        index_owner = NS(device="cpu")
        h_src, h_dst, final_src, final_dst = (
            backend.MambaAttnBackendBase._init_track_ssm_indices(
                index_owner, torch.tensor([1]), batch
            )
        )
    finally:
        backend.mamba_cache_chunk_size = original_chunk
    assert h_src.tolist() == [8] and h_dst.tolist() == [2]
    installed.Scheduler()._record_batch_launch(NS(reqs=[req(probe, "cold")]))
    state = torch.zeros(5, 1, 2)
    conv = torch.arange(10).reshape(5, 1, 2).to(torch.bfloat16)
    layer_cache = NS(conv=[conv], temporal=state)
    owner = installed.GDN()
    owner.req_to_token_pool = NS(mamba2_layer_cache=lambda _: layer_cache)
    metadata = NS(
        has_mamba_track_mask=True,
        track_ssm_h_src=h_src,
        track_ssm_h_dst=h_dst,
        track_ssm_final_src=final_src,
        track_ssm_final_dst=final_dst,
    )
    h = torch.arange(18).reshape(1, 9, 1, 2).to(torch.bfloat16)
    result = owner.forward_extend(NS(layer_id=0), batch, h, state, metadata)
    assert result is h and installed.collector.layer is None
    born = next(
        event for event in events(installed) if event["stage"] == "checkpoint_birth"
    )
    assert born["track_seqlens"] == [513] and born["track_boundaries"] == [512]
    assert born["h_source_indices"] == [8] and born["physical_slots"] == [2]
    saved = torch.load(born["tensor_file"], weights_only=True)
    assert torch.equal(saved["temporal"], h[0, 8:9].float())
    state.fill_(-999)
    assert torch.equal(
        torch.load(born["tensor_file"], weights_only=True)["temporal"],
        h[0, 8:9].float(),
    )
    assert len([call for call in installed.calls if call[0] == "extend"]) == 1


def test_real_cow_method_clears_fields_after_probe_keeps_addresses(probe, installed):
    installed.collector.identify(req(probe, "cold"))
    installed.Scheduler()._record_batch_launch(NS(reqs=[req(probe, "warm")]))
    pool = installed.ReqPool()
    temporal = torch.arange(15).reshape(1, 5, 3).float()
    conv = temporal.clone()
    state = NS(temporal=temporal, conv=[conv])

    def copy_from(source, destination):
        for tensor in (temporal, conv):
            tensor.index_copy_(
                1, destination.long(), tensor.index_select(1, source.long())
            )

    pool.mamba_pool = NS(mamba_cache=state, copy_from=copy_from)
    pool.mamba_map = {0: 0}
    pool.mamba_ckpt_pool = None
    pool.translate_mamba_indices = lambda slots: slots
    runner = installed.Runner()
    runner.req_to_token_pool = pool
    runner.is_draft_worker = False
    mode = NS(
        is_extend=lambda: True,
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
    )
    batch = NS(
        forward_mode=mode,
        mamba_clear_indices=None,
        mamba_cow_src_indices=torch.tensor([2]),
        mamba_cow_dst_indices=torch.tensor([1]),
    )
    runner._maybe_execute_deferred_mamba_cow_and_clear(batch)
    assert batch.mamba_cow_src_indices is None and batch.mamba_cow_dst_indices is None
    source, target = [
        event for event in events(installed) if event["stage"].startswith("warm_cow_")
    ]
    assert source["physical_slots"] == [2] and target["physical_slots"] == [1]
    assert torch.equal(
        torch.load(source["tensor_file"], weights_only=True)["temporal"],
        torch.load(target["tensor_file"], weights_only=True)["temporal"],
    )


@pytest.mark.parametrize(
    "skip",
    ["non_hybrid", "draft_worker", "decode", "target_verify", "draft_extend_v2"],
)
def test_real_cow_early_return_with_stale_indices_has_no_observations(
    probe, installed, skip
):
    installed.collector.identify(req(probe, "cold"))
    warm = req(probe, "warm")
    if skip == "decode":
        warm.output_ids = [11]
    installed.Scheduler()._record_batch_launch(NS(reqs=[warm]))
    runner = installed.Runner()
    # No pool storage is supplied: the real method must return before copying,
    # and an observer of a no-op must not attempt to inspect nonexistent state.
    runner.req_to_token_pool = object() if skip == "non_hybrid" else installed.ReqPool()
    runner.is_draft_worker = skip == "draft_worker"
    source, destination = torch.tensor([2]), torch.tensor([1])
    batch = NS(
        forward_mode=NS(
            is_extend=lambda: skip != "decode",
            is_target_verify=lambda: skip == "target_verify",
            is_draft_extend_v2=lambda: skip == "draft_extend_v2",
        ),
        mamba_clear_indices=None,
        mamba_cow_src_indices=source,
        mamba_cow_dst_indices=destination,
    )
    runner._maybe_execute_deferred_mamba_cow_and_clear(batch)
    assert batch.mamba_cow_src_indices is source
    assert batch.mamba_cow_dst_indices is destination
    assert len([call for call in installed.calls if call[0] == "cow"]) == 1
    assert not (installed.collector.directory / "events.jsonl").exists()


def test_actual_preprocessed_logits_are_copied_without_changing_return(
    probe, installed
):
    installed.Scheduler()._record_batch_launch(NS(reqs=[req(probe, "cold")]))
    logits = torch.tensor([[1.0, 1.001, -2.0]])
    result = installed.Sampler()._preprocess_logits(logits, NS(is_all_greedy=True))
    assert torch.equal(result, logits + 1)
    event = events(installed)[0]
    assert event["stage"] == "sampler_logits" and event["output_index"] == 0
    assert event["top2_token_ids"] == [1, 0]
    assert event["argmax_token_id"] == 1
    assert torch.equal(
        torch.load(event["tensor_file"], weights_only=True)["logits"], result
    )
    logits.fill_(42)
    assert torch.load(event["tensor_file"], weights_only=True)["logits"][0, 0] == 2


def test_kernel_probe_uses_real_ragged_count_instead_of_padded_projection(
    probe, installed
):
    installed.Scheduler()._record_batch_launch(NS(reqs=[req(probe, "cold")]))
    installed.collector.layer = 0
    q = torch.arange(576).reshape(1, 576, 1, 1).to(torch.bfloat16)
    result = installed.Dispatcher().extend(
        q,
        q,
        q,
        q,
        q,
        cache_indices=torch.tensor([1]),
        ssm_states=torch.zeros(5, 1, 2),
        query_start_loc=torch.tensor([0, 513]),
    )
    assert result[0] is q
    event = events(installed)[0]
    assert event["query_start_loc"] == [0, 513] and event["q_shape"][1] == 576
    saved = torch.load(event["tensor_file"], weights_only=True)
    assert torch.equal(saved["k_tail"], q[:, 512])
    assert len([call for call in installed.calls if call[0] == "kernel"]) == 1


def test_bitwise_comparison_preserves_dtype_and_signed_zero(probe):
    assert not probe.tensor_comparison(torch.tensor([0.0]), torch.tensor([-0.0]))[
        "bitwise_equal"
    ]
    assert not probe.tensor_comparison(
        torch.ones(1), torch.ones(1, dtype=torch.bfloat16)
    )["same_shape_and_dtype"]


def test_probe_does_not_record_unrelated_requests(probe, installed):
    unrelated = req(probe, "other")
    unrelated.origin_input_ids = [1, 2, 3]
    installed.Scheduler()._record_batch_launch(NS(reqs=[unrelated]))
    installed.Sampler()._preprocess_logits(torch.ones(1, 3), NS(is_all_greedy=True))
    assert installed.collector.current is None
    assert not (installed.collector.directory / "events.jsonl").exists()


@pytest.fixture
def complete_evidence(probe, tmp_path):
    references = {
        "cold_active": {"meta_info": {"id": "cold"}, "output_ids": [0] * 6 + [1, 0]},
        "active": {"meta_info": {"id": "warm"}, "output_ids": [0] * 6 + [2, 0]},
    }
    for rank, layer in ((0, 0), (2, 27)):
        recorder = probe.Collector(tmp_path, rank)
        recorder.current = {
            "rid": "cold",
            "label": "cold_active",
            "generated_prefix": [],
        }
        birth = {
            "temporal": torch.ones(1, 1, 2),
            "conv_0": torch.ones(1, 1, 2, dtype=torch.bfloat16),
        }
        recorder.event(
            "checkpoint_birth",
            layer_id=layer,
            prefix_length=512,
            track_seqlens=[513],
            track_boundaries=[512],
            h_source_indices=[8],
            chunk_size=64,
            physical_slots=[2],
            tensors={**birth, "h_source": birth["temporal"].bfloat16()},
        )
        recorder.event(
            "cold_finished_checkpoint",
            layer_ids=[layer],
            node_id=9,
            logical_slots=[2],
            physical_slots=[2],
            tensors={name: value.unsqueeze(0) for name, value in birth.items()},
        )
        for label, refkey in (
            ("cold_active", "cold_active"),
            ("warm_active", "active"),
        ):
            recorder.current = {
                "rid": references[refkey]["meta_info"]["id"],
                "label": label,
                "generated_prefix": [],
            }
            recorder.event(
                "gdn_extend",
                layer_id=layer,
                k_dtype="torch.bfloat16",
                h_dtype="torch.bfloat16",
                tensors={
                    name: torch.ones(1, 1, 2)
                    for name in ("q_tail", "k_tail", "v_tail", "g_tail", "beta_tail")
                },
            )
            if rank == 2:
                for index, token in enumerate(references[refkey]["output_ids"]):
                    recorder.current["generated_prefix"] = references[refkey][
                        "output_ids"
                    ][:index]
                    value = torch.zeros(1, 3)
                    value[0, token] = 2
                    recorder.event(
                        "sampler_logits",
                        output_index=index,
                        argmax_token_id=token,
                        is_all_greedy=True,
                        top2_token_ids=[token, (token + 1) % 3],
                        top2_logits=[2.0, 0.0],
                        top2_margin=2.0,
                        tensors={"logits": value},
                    )
        recorder.current = {
            "rid": "warm",
            "label": "warm_active",
            "generated_prefix": [],
        }
        recorder.event(
            "warm_match",
            full_kv_hit_length=512,
            best_match_node=9,
            logical_cow_source=[2],
            logical_cow_destination=[1],
        )
        for stage, slot in (("warm_cow_source", 2), ("warm_cow_destination", 1)):
            recorder.event(
                stage,
                layer_ids=[layer],
                logical_slots=[slot],
                physical_slots=[slot],
                tensors={name: value.unsqueeze(0) for name, value in birth.items()},
            )
    return tmp_path, references


def replace_event(directory, stage, change):
    path = directory / "rank-2/events.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    chosen = next(event for event in entries if event["stage"] == stage)
    change(chosen)
    path.write_text("".join(json.dumps(event) + "\n" for event in entries))


def test_analysis_reports_bitwise_chain_and_limits_comparison_to_common_history(
    probe, complete_evidence
):
    directory, references = complete_evidence
    report = probe.analyze(directory, references)
    assert report["continuous_fp32_state_at_512_observed"] is False
    sampler = report["ranks"][2]["sampler"]
    assert sampler["common_generated_prefix"] == 6
    assert [
        value["output_index"] for value in sampler["comparisons_on_common_history"]
    ] == list(range(7))
    assert sampler["comparisons_on_common_history"][-1]["bitwise_equal"] is False
    assert (
        report["ranks"][0]["birth_to_cold_end"][0]["temporal"]["bitwise_equal"] is True
    )


@pytest.mark.parametrize(
    "stage,change,expected",
    [
        (
            "checkpoint_birth",
            lambda event: event.update(rid="stale-request"),
            "different request",
        ),
        ("checkpoint_birth", lambda event: event.update(track_seqlens=[512]), "raw513"),
        (
            "warm_match",
            lambda event: event.update(full_kv_hit_length=448),
            "512 boundary",
        ),
        (
            "warm_match",
            lambda event: event.update(best_match_node=999),
            "observed cold checkpoint",
        ),
        (
            "warm_cow_destination",
            lambda event: event.update(layer_ids=[28]),
            "global-layer order",
        ),
        (
            "sampler_logits",
            lambda event: event.update(argmax_token_id=2),
            "actual output IDs",
        ),
        (
            "sampler_logits",
            lambda event: event.update(generated_prefix=[999]),
            "actual output IDs",
        ),
        (
            "sampler_logits",
            lambda event: event.update(output_index=1),
            "first eight fresh logits",
        ),
        (
            "checkpoint_birth",
            lambda event: event.update(tensor_file="/tmp/outside-probe.pt"),
            "private run directory",
        ),
    ],
)
def test_analysis_rejects_stale_or_incomplete_evidence(
    probe, complete_evidence, stage, change, expected
):
    directory, references = complete_evidence
    replace_event(directory, stage, change)
    with pytest.raises(AssertionError, match=expected):
        probe.analyze(directory, references)


def test_analysis_detects_changed_raw_tensor(probe, complete_evidence):
    directory, references = complete_evidence
    path = directory / "rank-2/events.jsonl"
    chosen = next(
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line)["stage"] == "checkpoint_birth"
    )
    values = torch.load(chosen["tensor_file"], weights_only=True)
    values["temporal"].fill_(9)
    torch.save(values, chosen["tensor_file"])
    with pytest.raises(AssertionError, match="identity changed"):
        probe.analyze(directory, references)


def test_analysis_rejects_duplicate_warm_kernel_event(probe, complete_evidence):
    directory, references = complete_evidence
    path = directory / "rank-2/events.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    duplicate = next(
        event
        for event in entries
        if event["stage"] == "gdn_extend" and event["label"] == "warm_active"
    )
    with path.open("a") as stream:
        stream.write(json.dumps(duplicate) + "\n")
    with pytest.raises(AssertionError, match="exactly one event per local layer"):
        probe.analyze(directory, references)


def test_worker_preserves_completed_references_and_cleans_up_after_request_failure(
    probe, tmp_path, monkeypatch
):
    saved, calls, alarms = {}, [], []

    def save(path, value):
        path.write_text(json.dumps(value))
        saved[path.name] = value

    def stream(endpoint, key, payload, timeout):
        calls.append(payload)
        if len(calls) == 4:
            raise OSError("warm diagnostic failed")
        return {
            "output_ids": [1] * payload["sampling_params"]["max_new_tokens"],
            "meta_info": {"id": str(len(calls))},
        }

    cleaned = []
    child = {"process": NS(poll=lambda: None)}
    helper = NS(
        authenticate=lambda *args, **kwargs: None,
        read_key=lambda _: "test-key",
        http_json=lambda *args, **kwargs: {},
        save=save,
        cleanup=lambda children: cleaned.append(list(children))
        or {"cleanup_verified": True},
    )
    smoke = NS(
        h=helper,
        launch=lambda *args: child,
        correctness_prompts=lambda: ([1] * 513, list(probe.ACTIVE_PROMPT), []),
    )
    monkeypatch.setitem(sys.modules, "agentx_replay", NS(stream=stream))
    monkeypatch.setitem(sys.modules, "hicache_transition_smoke", smoke)
    monkeypatch.setattr(probe.signal, "signal", lambda *args: None)
    monkeypatch.setattr(probe.signal, "alarm", alarms.append)
    with pytest.raises(SystemExit) as stopped:
        probe.worker(
            tmp_path,
            {"layouts": [{"port": 1234}], "strategy": "sync", "backing": "overlap"},
        )
    result = saved["results.json"]["baseline_probe"]
    assert stopped.value.code == 1
    assert cleaned == [[child]] and alarms == [225, 0]
    assert (
        result["cleanup_verified"] is True and result["diagnostic_completed"] is False
    )
    assert result["failed_stage"] == "active"
    assert set(result["references"]) == {"cold_prefix", "prefix", "cold_active"}
    contract = saved["reference_request_contract.json"]
    assert contract["requests"]["active"] == calls[-1]
    assert [payload["sampling_params"]["max_new_tokens"] for payload in calls] == [
        24,
        24,
        256,
        8,
    ]
    assert (
        result["generation_qualified"] is False and result["dynamic_qualified"] is False
    )
