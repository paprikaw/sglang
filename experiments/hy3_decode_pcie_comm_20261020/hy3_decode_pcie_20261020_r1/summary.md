# Hy3 / A100 Decode: local SM PCIe versus RDMA

Pure communication on booked065/167, same accepted native c3405cad for both policies. No model, Attention, Expert GEMM or HiCache. BF16 H4096, 192 experts, top8, capacity128, one source row/rank, recv-hook API, eager/CUDA Graph off. Three warmups and three samples per case.

Transport threshold0 uses NVLink inside pairs and RDMA outside pairs. Threshold1 enables mapped same-node cross-pair SM PCIe payload for these source1 calls; cross-node remains RDMA. Cross-pair control/notification can still use RDMA. Copy Engine is not configured. Policy and mapped-peer receipts are evidence of branch eligibility, not physical wire counters.

| placement / policy | Dispatch median ms | Combine median ms | D+C median ms | D+C mean ms | D+C rank-max samples ms |
|---|---:|---:|---:|---:|---|
| local065_rdma | 0.092864 | 0.072704 | 0.165888 | 0.751179 | 0.165888, 1.925152, 0.162496 |
| local065_sm_pcie | 0.159040 | 0.150528 | 0.306496 | 0.503371 | 0.303104, 0.900512, 0.306496 |
| local167_rdma | 0.095232 | 0.074752 | 0.160448 | 0.357707 | 0.158720, 0.753952, 0.160448 |
| local167_sm_pcie | 0.181504 | 0.152576 | 0.326912 | 0.524757 | 0.305152, 0.942208, 0.326912 |
| cross8_rdma | 0.187456 | 0.119808 | 0.289856 | 0.456672 | 0.248832, 0.831328, 0.289856 |
| cross8_sm_pcie | 0.336640 | 0.190464 | 0.527104 | 0.692171 | 0.497664, 1.051744, 0.527104 |

## Limits

- For each aligned sample take maximum across ranks, then descriptive median/mean. Dispatch, combine and pair maxima/medians are independent; do not sum the displayed segment medians.
- Per-rank work is matched: EP4 global4 tokens, EP8 global8. Balanced synthetic expert routes are not recorded Hy3 routes. EP4/EP8 also differ in group width, ownership and QPs/rank (48/24).
- CUDA-event API spans include native packing/control/reduction/wait and CPU enqueue gaps. They are not physical-link latency or full-model TPOT. Three samples cannot establish population tails or a stable small difference.
- Exact IDs/counts/BF16 payload and identity combine checked before timing and after a changed payload. All32 rank results independently recomputed. Guard postcheck and owned cleanup pass.

Raw source run: /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_decode_pcie_comm_20261020/hy3_decode_pcie_20261020_r1
