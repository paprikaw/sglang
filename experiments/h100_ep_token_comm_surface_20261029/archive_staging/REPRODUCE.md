# Reproduction

The accepted original run is `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_token_comm_surface_20261029/runs/r4`. All GPU measurements are frozen in that
run; no completed run or shared native library was edited. Read `REPORT.md`
for the exact original-token, chunk, topology, dtype and timing definitions.

CPU report: `/home/bxb1/conda/bin/python /data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_token_comm_surface_20261029/build_report.py --run /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_token_comm_surface_20261029/runs/r4 --out NEW_REPORT_DIR`.

GPU rerun: copy the frozen harness to a fresh directory, refresh only currently
RUNNING booked node/job identities, create a fresh run using `prepare.py`, and
run its `launch.json` command through the canonical Experiment Guard. Existing
CPU model/KV banks are not used or altered. No Slurm allocation is submitted;
workers run as normal steps of the specified existing Slurm jobs.

Every table cell is original aligned event 1/30, with no averaging or median.
All 30 iterations and all ranks are in CSV. N4096/8192/16384/32768 use measured
2/4/8/16 chunks of up to2048 global tokens identically across EP2/4/8.
EP8 spans two nodes; EP2/EP4 are node-local NVLink.
