"""Launch pinned HTTP serving with an explicit importable scheduler target.

Hook installation and validated startup run in that actual target, rather than
depending on multiprocessing replay of this main module. Model-cache admission
belongs to Guard/model_node, before this process is started.
"""
import os
import sys


def main():
    if os.environ.get('EP16_NO_MODEL_PROBE') == '1':
        raise RuntimeError('NO_MODEL_PROBE requires the dedicated bootstrap launcher, not HTTP serving')
    from cache_guard import isolate_humming_cache
    isolate_humming_cache()
    # Private cache selection precedes all engine/Humming imports.
    from scheduler_bootstrap import run_scheduler_process
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    server_args.resolve_once()
    if any(getattr(server_args, flag, False) for flag in
           ('encoder_only', 'smg_grpc_mode', 'grpc_mode', 'use_ray')):
        raise RuntimeError('This entry supports only the pinned default HTTP configuration')
    try:
        launch_server(server_args, run_scheduler_process_func=run_scheduler_process)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == '__main__':
    main()
