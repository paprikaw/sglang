# Hy3 / A100: node-local EP4 versus cross-node EP8 communication

Same nodes065/167 and accepted native c3405cad as the completed full-model comparison. No model, Attention, router, Expert GEMM, HiCache or PP boundary transfer is timed. BF16 H4096, 192 experts, top-k8, capacity128/source rank, eager, recv-hook API, NVLink within pairs and RDMA outside pairs; CE disabled. CPU-host-memory NIC handler, QP depth4096, 48 QPs/rank EP4 versus24 EP8. Three warmups and three measured calls per point.

## Measured communication spans

| workload | placement | source rows/rank | global source tokens | dispatch median ms | combine median ms | pair median ms | pair mean ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | local065 | 1 | 4 | 0.088832 | 0.069632 | 0.161536 | 0.402165 |
| prefill | local065 | 128 | 512 | 1.725216 | 1.733632 | 3.425824 | 3.504917 |
| decode | local167 | 1 | 4 | 0.089248 | 0.076800 | 0.162816 | 0.393589 |
| prefill | local167 | 128 | 512 | 1.707424 | 1.738752 | 3.435072 | 3.534155 |
| decode | cross8 | 1 | 8 | 0.140288 | 0.113664 | 0.247808 | 0.424779 |
| prefill | cross8 | 128 | 1024 | 2.753536 | 2.540544 | 5.266432 | 5.244939 |

## Interpretation and limits

- These match the full-model maximum Prefill chunk: local512/TP4 =128 source rows/rank; Decode B4 per Attention group gives1 row/rank. Sources and received assignments are matched per rank, not fixed globally. EP4 communicates global4/512 tokens per call; EP8 global8/1024. Balanced synthetic routes are not captured Hy3 router traces.
- Prefill pair median EP4 3.426/3.435ms versus EP8 5.266ms: EP4 about35% lower, difference1.831-1.841ms per call. All three EP8 samples exceed every local EP4 sample. Group width, experts/QPs and topology all change; not a physical-wire-only latency ablation.
- Decode medians EP4 0.162/0.163ms versus EP8 0.248ms. All three cases have a slow second dispatch sample (pair maxima0.915/0.858/0.803ms); pair means0.402/0.394/0.425ms. Three calls do not establish a stable53% penalty or service P99; do not silently discard outliers.
- In this balanced routing each source token sends two of eight expert copies to each owner in EP4, one to each owner in EP8. Logical per-node same-pair NVLink/self/cross-pair-RDMA/cross-node-RDMA shares are25/25/50/0 percent for EP4 and12.5/12.5/25/50 for EP8. Thus RDMA expert-copy share is50% versus75%. These are logical route counts, not physical NIC bytes or a bandwidth measurement.
- CUDA-event API spans include native control, packing, reduction, waiting and CPU submission gaps. Independent dispatch/combine rank maxima and medians must not be summed; pair is separately measured over the same rank interval.
- No microbenchmark result is multiplied into a claimed full-model speedup. At cachedP2304, the prior full model PP has Prefill1769.84 versus1381.21 tok/s, but Decode TPOT176.28 versus169.83ms; communication savings alone do not prove faster PP TPOT. At P3712 PP wins cohort Decode through capacity admission, a separate effect.
- Exact source IDs, counts, BF16 payload and identity combine checked before timing and after changing payload. All16 rank results, independent aligned-max recomputation, Guard scientific postcheck and owned cleanup pass; lease released.

Raw source run: /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_ep4_ep8_comm_20261020/run_r1
Full model source: /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_stream_cached_measure_v4_20261020/comparison.json
