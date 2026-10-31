# GLM semantic-bank four-way Decode experiment

## Fixed protocol

Model: GLM-4.7-AWQ, Humming expert kernels, BF16 KV; engine commit
`9e48f876a8eb2d7460317bc713bfa8680d78aea5`. Nodes: booked A100
`spartan-gpgpu065` and `spartan-gpgpu167`; no additional allocation.

| Arm | Attention | Expert | Pipeline | MTP |
|---|---|---|---|---|
| ep8_mtp_on | TP4 × DP2 | EP8 | 1 | EAGLE, steps2/topk1/draft3 |
| ep8_mtp_off | TP4 × DP2 | EP8 | 1 | Off |
| pp2ep4_mtp_on | TP4 per stage | EP4 per stage | 2, two B4 slots | Same EAGLE; qualified microbatch scheduler |
| pp2ep4_mtp_off | TP4 per stage | EP4 per stage | 2, two B4 slots | Off; native PP scheduler |

Global B8, semantic logical prefix P4096/request, Q1, 64 output tokens/request.
Every timed cohort starts with CPU4032/GPU0. The page-alignment tail is 64
original tokens; the initial forward therefore handles 65 tokens/request.
Native B4 CUDA Graph execution is checked for actual target Decode/Verify and
applicable draft calls. Request composition and PP slot membership are recorded.

## Shared bank and timing

A single PP2/MTP producer performs real semantic Prefill and exports the 92
target layers plus one typed draft layer. Eight rank pieces contain one logical
payload of 12,287,213,568 bytes (11.4434 GiB); no second complete canonical copy
is materialized. Publication requires completed producer exit and owned cleanup
on both nodes. Archive qualification additionally depends on final Guard checks.

Each fresh consumer maps global layer/head/request coordinates into new native
HiCache host slots. Native plain or bigram keys and token identities are checked;
source pages and host readback bytes are hashed. No synthetic KV is permitted.
Every warm, timed, repeat and profile cohort resets and reimports this immutable
bank before release, preventing plain-radix warmup from changing prefix length.

CPU weight loading and source-bank-to-HiCache import are excluded from request
timing. Native HiCache CPU-to-GPU transfer is included in TTFT and end-to-end
latency. TPOT is the client per-request `(E2E - TTFT)/(output_tokens - 1)`.
Two unprofiled measurement cohorts precede a separate profiled cohort. Eight
request p99 values are small-sample nearest-rank estimates, not tail guarantees.

## Communication and profiling

All arms use the same pinned DeepEP low-latency native library: NVLink inside
GPU pairs; RDMA outside each pair, including same-node cross-pair traffic.
PCIe Copy Engine is not enabled. This experiment isolates MTP/layout effects;
it does not compare communication-library variants.

Profiles separate Attention, routed/shared Expert, DeepEP Dispatch/Combine,
TP collectives, PP boundary operations, other GPU work and idle/wait. CUDA
activity is captured outside performance measurement. Kernel attribution uses
observed calibration; ambiguous work remains unknown. Overlap, active kernel
time and elapsed communication spans are distinguished. Per-Verify and
per-output-token normalizations use observed forwards and actual output counts.
PP rank times cover only that rank's layer partition and must not be compared
to EP8 full-depth rank times without an explicit critical-path calculation.

See `STATUS.json` and each run's Guard/cleanup records for actual completion.
Preparation directories and historical reference measurements are not results
of this four-way experiment.
