<!-- begin qwen35b_h100_decode_pool_necessity_20261029 -->
### qwen35b_h100_decode_pool_necessity_20261029

Snapshot: /data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_decode_pool_necessity_20261029

- New single-node Qwen3.6-35B BF16 Decode measurements on two separately controlled four-H100 nodes; EP4,2xEP2,4xsingle,optimized2xPP2,nativePP4. P256/4096/32768/131072,globalN8..128,Q1/O128,real GPU-hot prefixes,no HiCache or CPU KV replay. Three-round scans and independent reverse-order five-round confirmations remain separate.
- Primary011 scan: PP2 has lowest mean TPOT in0/16 workloads. P4096/N8: EP4 6.122ms,PP2 5.495ms,4single 4.752ms. P256/N128: EP4 10.630ms versus PP2 15.402ms. Strong low-load PP-versus-EP benefits are covered by single-GPU replicas.
- P131072/N32: PP2 mean request E2E2181.57ms versus EP4 2284.38ms(4.50% lower),but TPOT15.293 versus14.936ms. Five-round171 confirmation preserves this small E2E benefit. Same-profile common-feasible ideal selection proxy at most7.46% meanTPOT,4.83% meanE2E,6.74% serial-cohort makespan improvement over strongest static candidate; removingPP changes none of those optima. These are homogeneous-cohort proxies,not online autoscaling measurements or general upper bounds.
- 122 accepted layout/workload/run records,388 formal cohorts,16376 requests. Whole-run Guard postcheck passes in5 accepted runs;011_screen_r3 ends in an untimed single_dense startup failure,with44 completed timing points independently verified using original Guard loader/model/cleanup checks. Failed postcheck retained. N128 single-GPU retuning also fails untimed native GDN CoW;no performance value assigned. Initial six EP N128 capacity skips came from undersized Mamba pools and are superseded by successful dense profiling.
- Recommend pausing35B single-node PP/EP ratio-controller implementation; retain benchmark and examine replicas-versus-EP or models that require weight sharding. No larger-model or single-node8GPU claim. Foreground processes released,CPU banks preserved,normal background benchmarks verified recovered on011/171.

Markdown report: /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/REPORT.md

<!-- end qwen35b_h100_decode_pool_necessity_20261029 -->
