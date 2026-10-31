import os
from pathlib import Path
import runpy
import sys
from stream_common import ROOT, config, server_args, verify_streaming_entry
from stream_auth import authenticate

def main():
    run=Path(os.environ['HY3_RUN_DIR']);cfg=config(run)
    verify_streaming_entry(cfg);authenticate(run,cfg,local=True)
    from loader_patch import install
    install()
    from scheduler_hooks import install as install_scheduler
    install_scheduler()
    sys.argv=['sglang.launch_server',*server_args(cfg,int(os.environ['HY3_NODE_RANK']))]
    runpy.run_module('sglang.launch_server',run_name='__main__')
if __name__=='__main__':main()
