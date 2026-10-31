# PP2 x EP4 first Prefill proposal

The target and draft Prefill already compute a real top1 proposal after the
first sampled token. The pinned PP retirement discards it and starts Verify
with root + zero + zero. Seed column1 of that same Q3 chain with the already
computed draft top1. Column0 (root), column2 (fallback), and chain topology stay
unchanged. All proposals remain verified by the target. No new draft forward,
KV allocation, payload, collective, or inference setting is introduced.

Same setup: GLM4.7-AWQ, 065/167, 8 A100, two B4 slots, target50/42, MTP2/top1,
repaired Verify Attention, deterministic Humming, NVLink pairs + RDMA.
Semantic P4096, native CPU4032/GPU0, Q1 to64 outputs. Bank import and warmup
excluded; native HiCache H2D included. The early-feedback candidate was rejected
after a negative performance screen and is disabled here.

All stages must report real proposal adoption. Output IDs/text must match the
qualified same-PP baseline exactly; timings alone do not establish correctness.
The diagnostic DP-placement gate now distinguishes Attention DP2 from PP.
