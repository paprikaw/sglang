# GLM-4.7 A100 cached Decode: EP8 vs PP2 x EP4

Status: completed paired_r1; both arms and Guard scientific/cache/cleanup postchecks passed.
The approved Qwen RAM-bank replacement and complete GLM CPU preload were performed before measurement.

Historical workload source:
`experiment_manager/experiments/glm47_a100_m655360_b8_hicache_cold_20261015_r2/manifest.json`.

- Nodes: booked A100 065 and 167, 4 GPUs each, same pair sequentially.
- Model: complete GLM-4.7-AWQ, Humming INT4 weights, BF16 activations/KV.
- EP8: PP1, global TP8/Attention TP4 x DP2, MoE EP8.
- PP2 x EP4: Attention TP4, MoE EP4 within each stage, 46/46 layers, two B4 slots.
- Eight independent requests; each P=81,920 cached tokens, Q=1 new token, 128 output tokens; total P=655,360.
- Full topology-independent canonical CPU weight bank preloaded before serving.
- Engine 870269702285d94bdd926a50cc29ad1a20bcf60c; Triton Attention, Humming MoE, Graph OFF, overlap OFF.
- DeepEP native SHA256 9dc348878f7a3ad13fa8766a0d2738ca0d19ab116a3a5cc8e6e4e21b12988581; pair NVLink, RDMA outside pairs; PCIe threshold 0, same on both arms.
- Page 64, natural GPU token capacity (mem fraction .90); no artificial max-total-tokens cap.
- HiCache `--hicache-size 35` per rank (native allocated bytes recorded in all-rank capacity receipts); reserve 144 GiB per node; no persistent synthetic KV arena.
- Synthesize finite BF16 KV directly in native CPU HiCache before timing, with identical global layer/head mapping across layouts. This preparation differs from the historical source-arena staging but produces the same timing boundary.
- Warm target shapes and real PP boundary transport; wait for native transfers to quiesce, evict target GPU KV, prove CPU-full/GPU-empty on all eight ranks.
- Timed: CPU HiCache -> GPU H2D, request admission, forward, communication, streamed output.
- Excluded: CPU-bank construction, weights load, synthetic KV generation, shape/PP prewarm.
- Record natural capacity, all-rank host/device residency, actual scheduler batches/slots, per-request token events, TTFT/TPOT/E2E (mean/median/sample P99), cohort output throughput and cleanup.
- A repeated single matched cohort is not a statistically strong P99 estimate; disclose N=8.

Runtime sources were frozen before launch and were not changed during measurement. Both arms ran under one canonical Experiment Guard lease with complete full-model bank admission and separate per-arm usage logs. Final native scheduler receipts show B4 on every Decode launch: 127 launches per EP8 rank and 254 per PP rank, with two PP request groups. GPU leases were released after owned cleanup; CPU weight banks were retained.

Headline result: EP8 38.8787 output tok/s, mean TTFT 2.4386 s, mean TPOT 188.1597 ms, mean E2E 26.3353 s; PP2xEP4 39.0979 output tok/s, 1.7455 s, 192.0627 ms, 26.1382 s. Throughput differs by only 0.56% in this one paired cohort. Natural token capacities: EP8 443200 per DP group / 886400 total, PP2xEP4 908096; both fit all 655360 cached tokens.

The first offline summary recognized textual DECODE but missed the native IntEnum value "2" in its batch appendix. Its latency/throughput values were unaffected. The separately generated accepted_v2 corrects only offline batch recognition, preserves the original accepted directory, and asserts exact B4 Decode counts on all eight ranks.
