# PP2 x EP4 early-feedback screening

Hypothesis: after qualified local-tail feedback removed the head-to-tail echo,
the tail retains the old receive-before-send order even though receive is now
local retirement. Submit the current result before retiring the other slot.
The real pinned exchange body and all model math are unchanged.

Fixed setup: GLM-4.7-AWQ, 8 A100 on 065/167, Attention TP4 and Expert EP4
per pipeline stage, 50/42 target layers, two B4 slots, MTP steps2/top1/Q3,
Verify attention fix, deterministic Humming, native pair NVLink + RDMA.
Global B8, semantic P4096, native CPU4032/GPU0 at each timer start, Q1 to64
outputs. Weights, bank import and warmup excluded; HiCache H2D included.

Tests reuse the real pinned exchange through AST extraction and CPU queues;
cover slot ancestry, uneven drain, no duplicate send, and unchanged P2P work
ownership. Every real worker must report activation and drained counters.
Generated output IDs and text must exactly match the qualified same-PP baseline.

This first candidate-only run is an efficient screen against a historical
same-node/job/source/Bank baseline. A small timing delta requires fresh paired
confirmation before a speedup is accepted. No EP8 rerun is needed for this screen.
