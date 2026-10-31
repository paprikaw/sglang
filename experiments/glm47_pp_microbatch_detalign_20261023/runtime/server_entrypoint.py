import os,sys
from pathlib import Path
from common import config,server_args

def run_scheduler_process(*args,**kwargs):
    from startup import install
    install()
    from sglang.srt.managers.scheduler import run_scheduler_process as original
    return original(*args,**kwargs)

def main():
    from startup import isolate_humming
    isolate_humming()
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import kill_process_tree
    load_plugins();cfg=config(Path(os.environ['GLM_RUN_DIR']))
    args=prepare_server_args(server_args(cfg,int(os.environ['GLM_NODE_RANK'])));args.resolve_once()
    try:launch_server(args,run_scheduler_process_func=run_scheduler_process)
    finally:kill_process_tree(os.getpid(),include_parent=False)
if __name__=='__main__':main()
