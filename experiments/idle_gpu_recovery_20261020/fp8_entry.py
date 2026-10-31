"""Reuse the qualified scheduler-local A100 FP8 adapter before real model construction."""
import sys

def run_scheduler_process(*args, **kwargs):
 from sglang.srt.managers import scheduler
 scheduler.load_plugins()
 from runtime_setup import install
 install()
 return scheduler.run_scheduler_process(*args, **kwargs)

if __name__=='__main__':
 from sglang.srt.plugins import load_plugins
 from sglang.srt.server_args import prepare_server_args
 from sglang.srt.entrypoints.http_server import launch_server
 load_plugins()
 args=prepare_server_args(sys.argv[1:]);args.enable_multimodal=False
 args.resolve_once()
 launch_server(args,run_scheduler_process_func=run_scheduler_process)
