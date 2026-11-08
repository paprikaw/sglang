"""Private, static-gen0 observations of the original GDN/cache/sampler calls.

The probe retains CPU copies only. It neither changes a kernel nor replays a
shorter sequence. In particular, continuous FP32 state at token 512 remains
unobserved; a small logit margin does not qualify a numerical tolerance.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import time

ACTIVE_PROMPT = tuple(2000 + index % 160 for index in range(512)) + (2160,)
_COLLECTOR = None


def cpu_copy(tensor):
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def indices(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        return cpu_copy(value).reshape(-1).tolist()
    if isinstance(value, (int, float)):
        return [int(value)]
    return [int(item) for item in value]


def tensor_description(tensor):
    import torch

    assert tensor.device.type == "cpu"
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(
            tensor.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


class Collector:
    def __init__(self, directory, rank):
        self.directory = Path(directory) / f"rank-{rank}"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.rank = rank
        self.current = None
        self.layer = None
        self.labels = {}
        self.sample_counts = {}
        self.birth_slots = {}
        self.sequence = 0

    def event(self, stage, *, tensors=None, **details):
        import torch

        self.sequence += 1
        event = {
            "stage": stage,
            "rank": self.rank,
            "sequence": self.sequence,
            "time": time.time(),
            **dict(self.current or {}),
            **details,
        }
        if tensors:
            copies = {name: cpu_copy(tensor) for name, tensor in tensors.items()}
            target = self.directory / f"{self.sequence:04d}-{stage}.pt"
            torch.save(copies, target)
            target.chmod(0o600)
            event["tensor_file"] = str(target)
            event["tensors"] = {
                name: tensor_description(value) for name, value in copies.items()
            }
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        (self.directory / "events.jsonl").chmod(0o600)
        return event

    def identify(self, req):
        if tuple(req.origin_input_ids) != ACTIVE_PROMPT:
            return None
        if req.rid not in self.labels:
            if len(self.labels) >= 2:
                raise AssertionError(
                    "static probe received an unexpected third active request"
                )
            self.labels[req.rid] = ("cold_active", "warm_active")[len(self.labels)]
        return {
            "label": self.labels[req.rid],
            "rid": req.rid,
            "output_length": len(req.output_ids),
            "generated_prefix": list(req.output_ids[:8]),
            "logical_active_slots": indices(req.mamba_pool_idx),
        }

    def observe_batch(self, batch):
        self.current = self.identify(batch.reqs[0]) if len(batch.reqs) == 1 else None

    def state(self, stage, pool, slots, *, physical=False, **details):
        import torch

        logical = indices(slots)
        device = pool.mamba_pool.mamba_cache.temporal.device
        values = torch.tensor(logical, dtype=torch.int64, device=device)
        addresses = values if physical else pool.translate_mamba_indices(values)
        state = pool.mamba_pool.mamba_cache
        self.event(
            stage,
            logical_slots=None if physical else logical,
            physical_slots=indices(addresses),
            layer_ids=list(pool.mamba_map),
            tensors={
                "temporal": state.temporal.index_select(1, addresses),
                **{
                    f"conv_{index}": tensor.index_select(1, addresses)
                    for index, tensor in enumerate(state.conv)
                },
            },
            **details,
        )


def collector():
    global _COLLECTOR
    if _COLLECTOR is None:
        from sglang.srt.distributed.parallel_state import get_pp_group

        _COLLECTOR = Collector(
            os.environ["PP_HICACHE_PROBE_DIR"], get_pp_group().rank_in_group
        )
    return _COLLECTOR


def checkpoint_nodes(cache):
    """Read the existing tree without issuing another match or changing LRUs."""
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    result = []
    stack = [(cache.tree_core.root_node, ())]
    while stack:
        node, prefix = stack.pop()
        if node.parent is not None:
            prefix += tuple(node.key.raw_token_ids())
        if prefix == ACTIVE_PROMPT[:512]:
            data = node.component_data[ComponentType.MAMBA]
            result.append(
                {
                    "node_id": node.id,
                    "prefix_length": len(prefix),
                    "logical_device_slots": indices(data.value),
                    "host_slots": indices(data.host_value),
                }
            )
        stack.extend((child, prefix) for child in node.children.values())
    return result


def track_boundaries(forward_batch, chunk_size):
    return [
        prefix + ((raw - prefix) // chunk_size) * chunk_size
        for raw, prefix in zip(
            indices(forward_batch.mamba_track_seqlens),
            indices(forward_batch.extend_prefix_lens),
        )
    ]


def track_chunk_size():
    from sglang.srt.runtime_context import mamba_cache_chunk_size

    return mamba_cache_chunk_size()


def install_observers(
    Scheduler,
    GDNAttnBackend,
    GDNKernelDispatcher,
    MambaAttnBackendBase,
    ModelRunner,
    Sampler,
    UnifiedRadixCache,
    HybridReqToTokenPool,
    *,
    get_collector=collector,
    get_chunk_size=track_chunk_size,
):
    """Wrap real entry points; delegate each original exactly once."""
    original_launch = Scheduler._record_batch_launch

    def launch(self, batch):
        result = original_launch(self, batch)
        get_collector().observe_batch(batch)
        return result

    Scheduler._record_batch_launch = launch
    original_extend = GDNAttnBackend.forward_extend

    def extend(self, layer, forward_batch, *args, **kwargs):
        probe = get_collector()
        previous = probe.layer
        probe.layer = int(layer.layer_id)
        try:
            return original_extend(self, layer, forward_batch, *args, **kwargs)
        finally:
            probe.layer = previous

    GDNAttnBackend.forward_extend = extend
    original_kernel = GDNKernelDispatcher.extend

    def kernel(self, q, k, v, g, beta, **kwargs):
        probe = get_collector()
        observed = probe.current is not None and probe.current["output_length"] == 0
        if observed:
            cache_indices = kwargs["cache_indices"]
            before = cpu_copy(
                kwargs["ssm_states"].index_select(0, cache_indices.long())
            )
        result = original_kernel(self, q, k, v, g, beta, **kwargs)
        if observed:
            # Original query_start_loc gives the real ragged token count,
            # independently of any padded projection/kernel launch shape.
            valid_tokens = indices(kwargs["query_start_loc"])[-1]
            last = valid_tokens - 1
            h = result[2]
            probe.event(
                "gdn_extend",
                layer_id=probe.layer,
                q_dtype=str(q.dtype),
                k_dtype=str(k.dtype),
                h_dtype=str(h.dtype) if h is not None else None,
                h_shape=list(h.shape) if h is not None else None,
                temporal_dtype=str(kwargs["ssm_states"].dtype),
                query_start_loc=indices(kwargs["query_start_loc"]),
                kernel_cache_indices=indices(cache_indices),
                q_shape=list(q.shape),
                tensors={
                    "initial_temporal": before,
                    "final_temporal": kwargs["ssm_states"].index_select(
                        0, cache_indices.long()
                    ),
                    "q_tail": q[:, last],
                    "k_tail": k[:, last],
                    "v_tail": v[:, last],
                    "g_tail": g[:, last],
                    "beta_tail": beta[:, last],
                },
            )
        return result

    GDNKernelDispatcher.extend = kernel
    original_track = MambaAttnBackendBase._track_mamba_state_extend

    def track(self, forward_batch, h, ssm_states, forward_metadata):
        result = original_track(self, forward_batch, h, ssm_states, forward_metadata)
        probe = get_collector()
        if (
            probe.current is not None
            and probe.current["label"] == "cold_active"
            and probe.current["output_length"] == 0
            and track_boundaries(forward_batch, get_chunk_size()) == [512]
        ):
            destinations = forward_metadata.track_ssm_h_dst
            sources = forward_metadata.track_ssm_h_src
            if destinations is not None and destinations.numel():
                addresses = indices(destinations)
                probe.birth_slots[probe.layer] = addresses
                state = self.req_to_token_pool.mamba2_layer_cache(probe.layer)
                probe.event(
                    "checkpoint_birth",
                    layer_id=probe.layer,
                    prefix_length=512,
                    physical_slots=addresses,
                    h_source_indices=indices(sources),
                    track_seqlens=indices(forward_batch.mamba_track_seqlens),
                    track_boundaries=track_boundaries(forward_batch, get_chunk_size()),
                    chunk_size=get_chunk_size(),
                    tensors={
                        "temporal": ssm_states.index_select(0, destinations.long()),
                        "h_source": h.squeeze(0).index_select(0, sources.long()),
                        **{
                            f"conv_{index}": tensor.index_select(0, destinations.long())
                            for index, tensor in enumerate(state.conv)
                        },
                    },
                )
        return result

    MambaAttnBackendBase._track_mamba_state_extend = track
    original_finish = UnifiedRadixCache.cache_finished_req

    def finish(self, req, *args, **kwargs):
        probe = get_collector()
        identity = probe.identify(req)
        result = original_finish(self, req, *args, **kwargs)
        if identity is not None and identity["label"] == "cold_active":
            previous = probe.current
            probe.current = identity
            try:
                nodes = checkpoint_nodes(self)
                probe.event("cold_finished_radix", nodes=nodes)
                for node in nodes:
                    if node["logical_device_slots"]:
                        probe.state(
                            "cold_finished_checkpoint",
                            self.req_to_token_pool,
                            node["logical_device_slots"],
                            node_id=node["node_id"],
                            prefix_length=512,
                        )
                for layer_id, addresses in probe.birth_slots.items():
                    import torch

                    state = self.req_to_token_pool.mamba2_layer_cache(layer_id)
                    slots = torch.tensor(
                        addresses, dtype=torch.int64, device=state.temporal.device
                    )
                    probe.event(
                        "birth_slots_at_cold_finish",
                        layer_id=layer_id,
                        physical_slots=addresses,
                        # A raw slot after retirement is an observation only;
                        # radix ownership is reported separately above.
                        tensors={
                            "temporal": state.temporal.index_select(0, slots),
                            **{
                                f"conv_{index}": tensor.index_select(0, slots)
                                for index, tensor in enumerate(state.conv)
                            },
                        },
                    )
            finally:
                probe.current = previous
        return result

    UnifiedRadixCache.cache_finished_req = finish
    original_match = UnifiedRadixCache.match_prefix

    def match(self, params):
        result = original_match(self, params)
        req = params.req
        if req is not None:
            probe = get_collector()
            identity = probe.identify(req)
            if identity is not None and identity["label"] == "warm_active":
                previous = probe.current
                probe.current = identity
                try:
                    probe.event(
                        "warm_match",
                        full_kv_hit_length=result.full_kv_hit_length,
                        best_match_node=result.best_match_node,
                        logical_cow_source=indices(req.mamba_cow_src_index),
                        logical_cow_destination=indices(req.mamba_pool_idx),
                    )
                finally:
                    probe.current = previous
        return result

    UnifiedRadixCache.match_prefix = match
    original_cow = ModelRunner._maybe_execute_deferred_mamba_cow_and_clear

    def cow(self, forward_batch):
        probe = get_collector()
        source = forward_batch.mamba_cow_src_indices
        destination = forward_batch.mamba_cow_dst_indices
        observed = (
            isinstance(self.req_to_token_pool, HybridReqToTokenPool)
            and not self.is_draft_worker
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and probe.current is not None
            and probe.current["label"] == "warm_active"
            and probe.current["output_length"] == 0
            and source is not None
            and source.numel() > 0
        )
        if observed:
            source = source.clone()
            destination = destination.clone()
            probe.state("warm_cow_source", self.req_to_token_pool, source)
        result = original_cow(self, forward_batch)
        if observed:
            probe.state("warm_cow_destination", self.req_to_token_pool, destination)
        return result

    ModelRunner._maybe_execute_deferred_mamba_cow_and_clear = cow
    original_preprocess = Sampler._preprocess_logits

    def preprocess(self, logits, sampling_info):
        result = original_preprocess(self, logits, sampling_info)
        probe = get_collector()
        if probe.current is not None:
            rid = probe.current["rid"]
            index = probe.sample_counts.get(rid, 0)
            probe.sample_counts[rid] = index + 1
            if index < 8:
                actual = cpu_copy(result)
                if actual.shape[0] != 1:
                    raise AssertionError(
                        "static sampler probe requires one actual request"
                    )
                values, tokens = actual[0].float().topk(2)
                probe.event(
                    "sampler_logits",
                    output_index=index,
                    is_all_greedy=bool(sampling_info.is_all_greedy),
                    argmax_token_id=int(actual.argmax(-1).item()),
                    top2_token_ids=tokens.tolist(),
                    top2_logits=values.tolist(),
                    top2_margin=float(values[0] - values[1]),
                    tensors={"logits": actual},
                )
        return result

    Sampler._preprocess_logits = preprocess


def install_hooks():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import (
        GDNAttnBackend,
        GDNKernelDispatcher,
    )
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.model_executor.model_runner import ModelRunner

    install_observers(
        Scheduler,
        GDNAttnBackend,
        GDNKernelDispatcher,
        MambaAttnBackendBase,
        ModelRunner,
        Sampler,
        UnifiedRadixCache,
        HybridReqToTokenPool,
    )


def tensor_comparison(left, right):
    import torch

    same_layout = left.shape == right.shape and left.dtype == right.dtype
    result = {"same_shape_and_dtype": same_layout, "bitwise_equal": False}
    if same_layout:
        result["bitwise_equal"] = torch.equal(
            left.view(torch.uint8), right.view(torch.uint8)
        )
        result["max_abs_difference"] = (
            float((left.float() - right.float()).abs().max()) if left.numel() else 0.0
        )
    return result


def analyze(directory, references):
    """Require fresh, same-rid coverage; report equality without a tolerance."""
    import torch

    directory = Path(directory).resolve()
    report = {"ranks": {}, "continuous_fp32_state_at_512_observed": False}
    for rank in (0, 2):
        event_file = directory / f"rank-{rank}/events.jsonl"
        events = [json.loads(line) for line in event_file.read_text().splitlines()]
        for event in events:
            label = event.get("label")
            if label in ("cold_active", "warm_active"):
                response = references[
                    "cold_active" if label == "cold_active" else "active"
                ]
                if event.get("rid") != response["meta_info"]["id"]:
                    raise AssertionError("probe event belongs to a different request")

        def select(stage, label=None):
            return [
                event
                for event in events
                if event["stage"] == stage
                and (label is None or event.get("label") == label)
            ]

        def one(stage, label=None):
            found = select(stage, label)
            if len(found) != 1:
                raise AssertionError(
                    f"rank {rank} requires exactly one fresh {stage}, found {len(found)}"
                )
            return found[0]

        def tensors(event):
            path = Path(event["tensor_file"]).resolve()
            if not path.is_relative_to(directory):
                raise AssertionError("probe tensor escaped the private run directory")
            result = torch.load(path, map_location="cpu", weights_only=True)
            if {
                name: tensor_description(value) for name, value in result.items()
            } != event["tensors"]:
                raise AssertionError("probe tensor identity changed")
            return result

        match = one("warm_match", "warm_active")
        if match["full_kv_hit_length"] != 512:
            raise AssertionError(
                "warm diagnostic did not match the original 512 boundary"
            )
        ended = one("cold_finished_checkpoint", "cold_active")
        source = one("warm_cow_source", "warm_active")
        destination = one("warm_cow_destination", "warm_active")
        if (
            match["best_match_node"] != ended["node_id"]
            or match["logical_cow_source"] != ended["logical_slots"]
            or source["logical_slots"] != match["logical_cow_source"]
            or destination["logical_slots"] != match["logical_cow_destination"]
            or source["physical_slots"] != ended["physical_slots"]
        ):
            raise AssertionError(
                "warm CoW does not originate from the observed cold checkpoint"
            )
        layer_ids = source["layer_ids"]
        if rank not in (0, 2) or not layer_ids:
            raise AssertionError("probe lacks actual active GDN layers")
        births = select("checkpoint_birth", "cold_active")
        if len(births) != len(layer_ids) or {
            event["layer_id"] for event in births
        } != set(layer_ids):
            raise AssertionError(
                "512 checkpoint birth coverage differs from actual local GDN layers"
            )
        end_state, source_state, destination_state = map(
            tensors, (ended, source, destination)
        )
        if not (ended["layer_ids"] == layer_ids == destination["layer_ids"]):
            raise AssertionError("checkpoint chain changed global-layer order")
        if not (set(end_state) == set(source_state) == set(destination_state)):
            raise AssertionError("checkpoint chain omitted a state component")
        chain = {}
        for name in source_state:
            chain[name] = {
                "cold_end_to_warm_source": tensor_comparison(
                    end_state[name], source_state[name]
                ),
                "warm_source_to_destination": tensor_comparison(
                    source_state[name], destination_state[name]
                ),
            }
        born = {}
        for event in births:
            if (
                event["prefix_length"] != 512
                or event["track_seqlens"] != [513]
                or event["track_boundaries"] != [512]
                or event["h_source_indices"] != [8]
                or event["chunk_size"] != 64
            ):
                raise AssertionError(
                    "birth is not the original raw513/chunk64 boundary512"
                )
            index = layer_ids.index(event["layer_id"])
            values = tensors(event)
            born[event["layer_id"]] = {
                name: tensor_comparison(value, end_state[name][index])
                for name, value in values.items()
                if name != "h_source"
            }
            born[event["layer_id"]]["h_cast_to_checkpoint"] = tensor_comparison(
                values["h_source"].to(values["temporal"].dtype), values["temporal"]
            )
        inputs = {}

        def original_calls(label):
            calls = select("gdn_extend", label)
            if len(calls) != len(layer_ids) or {
                event["layer_id"] for event in calls
            } != set(layer_ids):
                raise AssertionError(
                    "original GDN calls require exactly one event per local layer"
                )
            return {event["layer_id"]: event for event in calls}

        cold_layers = original_calls("cold_active")
        warm_layers = original_calls("warm_active")
        for layer in layer_ids:
            left, right = tensors(cold_layers[layer]), tensors(warm_layers[layer])
            inputs[layer] = {
                "cold_k_dtype": cold_layers[layer]["k_dtype"],
                "cold_h_dtype": cold_layers[layer]["h_dtype"],
                "warm_k_dtype": warm_layers[layer]["k_dtype"],
                "warm_h_dtype": warm_layers[layer]["h_dtype"],
                "last_valid_token_inputs": {
                    name: tensor_comparison(left[name], right[name])
                    for name in ("q_tail", "k_tail", "v_tail", "g_tail", "beta_tail")
                },
            }
        logits = {}
        # Only the final PP stage executes the sampler. Both active stages
        # must supply state coverage; requiring logits on PP0 would be false.
        if rank == 2:
            sample_tensors = {}
            for label, reference in (
                ("cold_active", references["cold_active"]),
                ("warm_active", references["active"]),
            ):
                samples = sorted(
                    select("sampler_logits", label),
                    key=lambda event: event["output_index"],
                )
                if [event["output_index"] for event in samples] != list(range(8)):
                    raise AssertionError(
                        "actual sampler lacks the first eight fresh logits"
                    )
                for index, event in enumerate(samples):
                    value = tensors(event)["logits"]
                    actual_argmax = value.argmax(-1).item()
                    if (
                        not event["is_all_greedy"]
                        or value.shape[0] != 1
                        or actual_argmax != reference["output_ids"][index]
                        or event["argmax_token_id"] != actual_argmax
                        or event["generated_prefix"] != reference["output_ids"][:index]
                    ):
                        raise AssertionError(
                            "captured sampler logits do not explain actual output IDs"
                        )
                sample_tensors[label] = [tensors(event)["logits"] for event in samples]
                logits[label] = [
                    {
                        key: event[key]
                        for key in (
                            "output_index",
                            "argmax_token_id",
                            "generated_prefix",
                            "top2_token_ids",
                            "top2_logits",
                            "top2_margin",
                        )
                    }
                    for event in samples
                ]
            logits["common_generated_prefix"] = next(
                (
                    index
                    for index, pair in enumerate(
                        zip(
                            references["cold_active"]["output_ids"][:8],
                            references["active"]["output_ids"],
                        )
                    )
                    if pair[0] != pair[1]
                ),
                8,
            )
            # The first differing output is still conditioned on the common
            # history. Later logits have different histories and are retained
            # as raw evidence without attributing their difference to cache.
            logits["comparisons_on_common_history"] = [
                {"output_index": index, **tensor_comparison(left, right)}
                for index, (left, right) in enumerate(
                    zip(sample_tensors["cold_active"], sample_tensors["warm_active"])
                )
                if index <= logits["common_generated_prefix"]
            ]
        report["ranks"][rank] = {
            "layer_ids": layer_ids,
            "birth_to_cold_end": born,
            "checkpoint_chain": chain,
            "kernel_inputs": inputs,
            "sampler": logits,
        }
    return report


def worker(run, cfg):
    import agentx_replay
    import hicache_transition_smoke as smoke

    h = smoke.h
    h.authenticate(run, cfg, os.environ, local=True)
    key = h.read_key(cfg)
    row = cfg["layouts"][0]
    started = time.monotonic()
    children = []
    evidence = {
        "passed": False,
        "diagnostic_completed": False,
        "acceptance_class": "static generation-zero diagnostic",
        "dynamic_qualified": False,
        "generation_qualified": False,
        "continuous_fp32_state_at_512_observed": False,
        "cuda_graph_enabled": False,
        "transitions": [],
        "strategy": cfg["strategy"],
        "backing": cfg["backing"],
    }

    def checkpoint(stage):
        evidence["stage"] = stage
        h.save(
            run / "worker_stage.json",
            {
                "stage": stage,
                "timestamp": time.time(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    def expired(*_):
        raise TimeoutError(
            "225-second static probe deadline; 75 seconds reserved for cleanup"
        )

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(225)
    try:
        checkpoint("startup")
        children.append(smoke.launch(run, cfg, row, key))
        while True:
            if children[0]["process"].poll() is not None:
                raise RuntimeError("engine exited during startup")
            try:
                h.http_json(row["port"], "/get_server_info", key, None, timeout=2)
                break
            except (OSError, ValueError):
                time.sleep(0.25)
        prefix, active, _ = smoke.correctness_prompts()
        references, contracts = {}, {}
        evidence["references"] = references
        for label, tokens, output in (
            ("cold_prefix", prefix, 24),
            ("prefix", prefix, 24),
            ("cold_active", active, 256),
            ("active", active, 8),
        ):
            checkpoint(label)
            payload = {
                "input_ids": tokens,
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output,
                    "ignore_eos": True,
                },
            }
            # This is the exact object handed to stream(), before HTTP JSON
            # encoding, unlike reconstruction of a historical request.
            contracts[label] = payload
            target = run / "reference_request_contract.json"
            h.save(
                target,
                {
                    "capture": "actual pre-serialization stream payload",
                    "requests": contracts,
                },
            )
            target.chmod(0o600)
            references[label] = agentx_replay.stream(
                {"host": "127.0.0.1", "port": row["port"], "ep": 1},
                key,
                payload,
                timeout=45,
            )
        evidence["observed_output_prefix_equal"] = (
            references["cold_prefix"]["output_ids"]
            == references["prefix"]["output_ids"]
            and references["cold_active"]["output_ids"][:8]
            == references["active"]["output_ids"]
        )
        evidence["reference_qualification_scope"] = (
            "prefix24 and first8 active IDs; cold256/warm8"
        )
        checkpoint("collect_probe_evidence")
        info = smoke.state_evidence(
            h.http_json(row["port"], "/get_server_info", key, None, timeout=10)
        )
        evidence["after"] = info
        smoke.rank_states(info["internal_states"])
        if any(
            state["elastic_pp_state"]["generation"] != 0
            for state in info["internal_states"]
        ):
            raise AssertionError("static diagnostic changed topology generation")
        analysis = analyze(run / "baseline_probe", references)
        h.save(run / "diagnostic_analysis.json", analysis)
        evidence["diagnostic_completed"] = True
        # This probe cannot qualify a full 256-token warmed trajectory or a
        # transition. The separate observed prefix result is all it compares.
        evidence["passed"] = evidence["observed_output_prefix_equal"]
        if not evidence["observed_output_prefix_equal"]:
            evidence["error"] = (
                "cold/warm generation-zero output IDs differ; tolerance unqualified"
            )
    except BaseException as error:
        evidence["failed_stage"] = evidence.get("stage")
        evidence["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        checkpoint("cleanup")
        cleanup = h.cleanup(children)
        elapsed = time.monotonic() - started
        evidence.update(
            elapsed_seconds=elapsed,
            cleanup_verified=cleanup.get("cleanup_verified", False),
        )
        evidence["passed"] = (
            evidence["passed"] and evidence["cleanup_verified"] and elapsed <= 300
        )
        evidence["stage"] = "done" if evidence["passed"] else "failed"
        h.save(run / "worker_cleanup.json", cleanup)
        h.save(run / "results.json", {"baseline_probe": evidence})
        h.save(run / "worker_progress.json", {"completed": int(evidence["passed"])})
    raise SystemExit(0 if evidence["passed"] else 1)
