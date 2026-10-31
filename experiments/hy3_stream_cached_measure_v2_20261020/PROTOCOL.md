# Hy3 cached EP8 / PP2 x EP4, A100

## Fixed controls

- Nodes: booked `spartan-gpgpu065` / `spartan-gpgpu167`, four A100 80GB per node.
- Both NICs live-checked: MT2892 ConnectX-6 Dx, device `0x101d`, PCIe 16 GT/s x8.
- Full 80-layer Hy3 BF16; original checkpoint revision and engine commit frozen in each spec.
- EP8: Attention TP4 x DP2; routed experts EP8. PP2 x EP4: Attention TP4 and routed experts EP4 per 40-layer stage.
- Same DeepEP native, BF16 low-latency path: NVLink within pairs, RDMA outside pairs. Copy Engine PCIe is not enabled.
- Triton Attention and MoE; eager execution, overlap disabled. Local Prefill budget 1024 for both layouts (EP8 CLI global chunk 2048, PP CLI chunk 1024).
- Eight independent requests; EP Attention DP affinity `request_index % 2`; PP maximum microbatch four. Real launched composition is recorded in rank receipts.

## Cached workload and timing

Old context P is synthesized directly into the native Unified HiCache CPU pool. Values are finite BF16, consistent in global layer/KV-head coordinates across layouts. This is a performance workload, not a quality evaluation. No source KV arena or model Prefix forward is used.

The first run selects page-aligned P from 90% of the smallest measured rank capacity, divided by its local request count, minus 384 tail tokens/request. Paired runs use identical P and request IDs/token sequences. Actual GPU/CPU capacity receipts accompany every result; different occupancy is not silently labelled identical capacity.

| Phase | Cached prefix/request | New input Q/request | New output/request |
|---|---|---|---|
| Prefill | P | 256 | 1 |
| Decode | P | 1 | 128 |

For each phase, run one full-shape warmup, then wait for native transfer bookkeeping to drain and evict GPU prefix pages. Require all eight rank receipts to show CPU hit P and GPU hit zero before the formal cohort. The first new token differs between warm and formal requests, preventing warm suffix hits.

Formal timing includes native HiCache H2D, server admission/queue, computation, and PP/EP communication. It excludes weight streaming, host KV construction, and warmup. There is no L3 storage backend or timed synthetic generation. Cohort throughput starts when all eight prepared clients cross a barrier. TTFT, TPOT and end-to-end latency retain per-request values and mean/median/sample P99; eight-request P99 is descriptive only.

Cleanup is recorded by the CPU owner even on startup failure; old failed receipts and frozen sources are preserved. Existing resident CPU weight banks are not replaced.
