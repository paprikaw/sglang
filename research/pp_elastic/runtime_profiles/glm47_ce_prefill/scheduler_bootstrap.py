"""Importable scheduler target: install startup-only evidence in each actual worker."""
import hashlib
import inspect
import os
from pathlib import Path
import time

_INSTALLED_PID = None
_INSTALLED_RANK = None
_INSTALLATION = None


def _ensure_installation(rank):
    global _INSTALLED_PID, _INSTALLED_RANK, _INSTALLATION
    if _INSTALLED_PID == os.getpid():
        if _INSTALLED_RANK != rank:
            raise RuntimeError('Scheduler rank changed in one process')
        return _INSTALLATION
    from cache_guard import isolate_humming_cache
    isolate_humming_cache()
    from sglang.srt.plugins import load_plugins
    load_plugins()
    from startup_hooks import install
    installation = install()
    installation.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    _INSTALLED_PID, _INSTALLED_RANK, _INSTALLATION = os.getpid(), rank, installation
    return installation


def run_scheduler_process(server_args, port_args, gpu_id, tp_rank,
                          attn_cp_rank=0, moe_dp_rank=0, moe_ep_rank=0,
                          pp_rank=0, dp_rank=None, pipe_writer=None,
                          display_tp_rank=None, display_dp_rank=None,
                          display_moe_ep_rank=None):
    from common import case, save
    rank = int(pp_rank)*4+int(tp_rank)
    run = Path(os.environ['RUN_DIR']).resolve()
    arm = os.environ['ARM']
    selected = case(arm, run)
    output = run / os.environ['ATTEMPT']
    evidence = dict(passed=False, rank=rank, node_rank=rank//4, local_rank=rank%4,
        arm=arm, pid=os.getpid(), parent_pid=os.getppid(), timestamp=time.time(),
        target_module=__name__, target_path=str(Path(__file__).resolve()),
        effective_run_dir=str(run), original_scheduler_called=False)
    save(output / f'bootstrap_entry_rank{rank}.json', evidence)
    try:
        if (not 0 <= rank < selected['world'] or int(os.environ['NODE_RANK']) != rank//4
                or int(gpu_id) != rank%4 or os.environ['ATTEMPT'] != arm):
            raise RuntimeError('Scheduler global/local rank or case mismatch')
        installation = _ensure_installation(rank)
        from sglang.srt.managers.scheduler import run_scheduler_process as original
        evidence.update(passed=True, installation=installation, original_scheduler_called=True,
            original_scheduler_path=inspect.getsourcefile(original), timestamp=time.time())
        save(output / f'bootstrap_rank{rank}.json', evidence)
        return original(server_args, port_args, gpu_id, tp_rank, attn_cp_rank,
            moe_dp_rank, moe_ep_rank, pp_rank, dp_rank, pipe_writer,
            display_tp_rank, display_dp_rank, display_moe_ep_rank)
    except BaseException as error:
        evidence.update(passed=False, error_type=type(error).__name__, error=str(error), timestamp=time.time())
        save(output / f'bootstrap_failure_rank{rank}.json', evidence)
        save(output / f'bootstrap_rank{rank}.json', evidence)
        raise
