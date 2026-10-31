"""Select existing SGLang overlap flags; observe streams without changing math."""
from functools import wraps
import json
import os
from pathlib import Path

_INSTALLED = False
_RECORDS = {}
_TBO_CALLS = set()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def install_runtime():
    global _INSTALLED
    if _INSTALLED or os.environ.get("EP2_PP2_ARM") != "ep2":
        return
    import torch
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.environ import envs
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
    root = Path(os.environ["EP2_PP2_RUN"]) / "ep2/native_overlap_audit"
    forward = Qwen2MoeSparseMoeBlock._forward_deepep
    shared = Qwen2MoeSparseMoeBlock._forward_shared_experts

    @wraps(forward)
    def forward_wrapped(self, hidden_states, forward_batch):
        if self.layer_id == 0:
            self._native_overlap_parent_stream = torch.cuda.current_stream().cuda_stream
        return forward(self, hidden_states, forward_batch)

    @wraps(shared)
    def shared_wrapped(self, hidden_states, *args, **kwargs):
        if self.layer_id == 0 and hidden_states.shape[0] <= 16:
            n = int(hidden_states.shape[0])
            capture = torch.cuda.is_current_stream_capturing()
            key = f"tokens{n}.capture{int(capture)}"
            if key not in _RECORDS:
                current = torch.cuda.current_stream().cuda_stream
                parent = getattr(self, "_native_overlap_parent_stream", None)
                _RECORDS[key] = dict(tokens=n, cuda_graph_capture=capture,
                    qwen_overlap_enabled=envs.SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP.get(),
                    current_stream=current, parent_stream=parent,
                    alt_stream=self.alt_stream.cuda_stream if self.alt_stream is not None else None,
                    on_independent_stream=parent is not None and current != parent,
                    has_shared_expert=self.shared_expert is not None)
                save(root / f"shared.rank{get_tp_group().rank_in_group}.json", _RECORDS)
        return shared(self, hidden_states, *args, **kwargs)

    Qwen2MoeSparseMoeBlock._forward_deepep = forward_wrapped
    Qwen2MoeSparseMoeBlock._forward_shared_experts = shared_wrapped
    if os.environ.get("NATIVE_OVERLAP_TBO") == "1":
        execute = MaybeTboDeepEPDispatcher._execute

        @wraps(execute)
        def execute_wrapped(self, name, tbo_subbatch_index=None, **kwargs):
            key = (name, tbo_subbatch_index)
            if key not in _TBO_CALLS:
                _TBO_CALLS.add(key)
                save(root / f"tbo.rank{get_tp_group().rank_in_group}.json", dict(
                    calls=[dict(operation=n, subbatch_index=i) for n, i in sorted(_TBO_CALLS, key=str)],
                    inner_dispatchers=len(self._inners)))
            return execute(self, name, tbo_subbatch_index=tbo_subbatch_index, **kwargs)

        MaybeTboDeepEPDispatcher._execute = execute_wrapped
    _INSTALLED = True


def install(base):
    previous_args, previous_environment, previous_workloads = base.engine_args, base.environment, base.run_workloads

    def engine_args(cfg, arm):
        args = previous_args(cfg, arm)
        if cfg["native_two_batch_overlap"]:
            args += ["--enable-two-batch-overlap"]
        return args

    def environment(cfg, arm, inherited):
        env = previous_environment(cfg, arm, inherited)
        env.update(SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP=str(int(cfg["native_shared_overlap"])),
                   NATIVE_OVERLAP_TBO=str(int(cfg["native_two_batch_overlap"])))
        return env

    def workloads(run, cfg, arm, key, health):
        previous_workloads(run, cfg, arm, key, health)
        receipts = []
        for rank in (0, 1):
            path = run / arm / "native_overlap_audit" / f"shared.rank{rank}.json"
            observed = json.loads(path.read_text())["tokens4.capture1"]
            if observed["qwen_overlap_enabled"] != cfg["native_shared_overlap"]:
                raise RuntimeError("Native overlap setting differs from frozen configuration")
            if observed["on_independent_stream"] != cfg["native_shared_overlap"]:
                raise RuntimeError("Native Qwen shared-expert stream branch not executed")
            receipts.append(dict(rank=rank, **observed))
        info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
        args = info.get("server_args", {})
        flags = {k: args.get(k) for k in ("enable_two_batch_overlap", "enable_single_batch_overlap", "disable_overlap_schedule")}
        tbo = []
        if cfg["native_two_batch_overlap"]:
            for rank in (0, 1):
                tbo.append(json.loads((run / arm / "native_overlap_audit" / f"tbo.rank{rank}.json").read_text()))
        save(run / arm / "native_overlap_qualification.json", dict(
            passed=True, shared_stream_receipts=receipts, native_server_flags=flags,
            requested_two_batch_overlap=cfg["native_two_batch_overlap"], tbo_dispatch_receipts=tbo,
            actual_tbo_subbatch1_seen=any(c["subbatch_index"] == 1 for r in tbo for c in r["calls"])))

    base.engine_args, base.environment, base.run_workloads = engine_args, environment, workloads
