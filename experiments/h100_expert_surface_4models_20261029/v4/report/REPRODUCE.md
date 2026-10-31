# Reproduce the absolute raw view on CPU

Run `/home/bxb1/conda/bin/python /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/report_tools/absolute_raw.py --run /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_expert_surface_4models_20261029/v4/runs/r6 --event 1`.

Each displayed value is read directly from raw_ms[event - 1] in the accepted result JSON. No reduction across samples or GPUs is performed. The source experiment is not rerun.
