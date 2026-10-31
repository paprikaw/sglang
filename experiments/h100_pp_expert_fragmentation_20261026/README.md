# H100 PP Expert fragmentation — completed

Full BF16 Qwen3.6-35B-A3B, the same H100 NVLink GPU0/1 pair, matched cold Prefill and GPU-cached Decode. Shape-adaptive Expert execution improves PP2 Decode throughput by 20.83% and reduces routed GEMM time by 38.83%. Prefill has no stable gain; generation is not bitwise equivalent, and task quality has not been evaluated.

- [Report with protocol and all limitations](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/REPORT.md)
- [End-to-end and component figure](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/end_to_end_and_components.png) / [PDF](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/end_to_end_and_components.pdf)
- [Same-work mechanism figure](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/fragmentation_mechanism.png) / [PDF](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/fragmentation_mechanism.pdf)
- [Full metrics](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/full_metrics.csv) / [Round audit](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/run_audit.json)
- [Canonical archive](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_expert_fragmentation_20261026/manifest.json)

Runtime adapter: `fragmentation.py`. Alternative kernel: `route_gemv.py` (measured, not selected). `prepare.py` freezes a fresh private run; `launch.py --run PATH` uses Guard. New launches require fresh resource/cache checks; completed run inputs remain immutable.

Offline replay: `analyze.py`, `validate_probabilities.py --batched`, `plot.py` (Matplotlib Python), then `finalize.py`. Original large logs and GPU traces stay in `runs/`. Seven launches completed cleanup and release; one exploratory run failed its later probability validation and is excluded from the headline comparison.

Offline headroom analysis: [Expert roofline and PP2 scenarios](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/HEADROOM_ESTIMATE.md) (modeled, not new measurements).
