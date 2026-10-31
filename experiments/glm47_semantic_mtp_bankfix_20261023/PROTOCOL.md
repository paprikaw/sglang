# Semantic MTP Decode: matched production plus separate profile

065/167, 8 A100 80 GB, same ConnectX-6 Dx NICs. Complete GLM-4.7-AWQ
CPU weight banks are reused, not rebuilt. No synthetic KV or random token IDs.

Eight distinct real code/documentation prompts, each P81920 + Q1. Both layouts
use the identical frozen token payloads. Actual model Prefill constructs every
target and MTP KV tensor; it is untimed. Native EAGLE bigram/page matching retains
81856 cached tokens/request; the first forward recomputes 65 input tokens.

PP2×EP4 runs first: short semantic MTP qualification precedes long Prefill.
Attention TP4 and Expert EP4 in each 46-layer stage, two B4 slots. EP8 uses
Attention TP4×DP2 and Expert EP8, same B8 cohort. MTP is native single-layer EAGLE,
steps2/topk1/draft3. Humming INT4 experts, BF16 KV, target Verify Graph enabled.

For each layout: real Prefill → CPU-native HiCache proof → Decode warmup128 →
evict GPU KV → formal Decode128. TTFT includes HiCache H2D; preparation, model
loading and warmup are excluded. CUPTI and GPU markers stay OFF during this
production cohort. Dormant CPU wrappers are present in both layouts.

Only after the complete production result is saved, evict GPU KV again and
profile a separate 64-output-token cohort from the same semantic HiCache.
Capture 12 actual target Verify calls per rank. Keep target, draft, draft-extend,
unknown GPU activity and no-traced-activity gaps separate. Profile wall times
are not production metrics, and local Verify time is not per-output-token TPOT.

Communication is pinned to historical NVLink-pair + RDMA outside pairs; no CE.
Never compare semantic MTP speedup against older synthetic-KV Decode results.
Preserve partial completed points if the second arm or profile fails.
