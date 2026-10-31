# Captured external-event timing qualification

No model, Attention or Expert compute, no `torch.profiler` imports or CUPTI API/session. Reuse the qualified EP8 DeepEP native/wrapper and pair-NVLink + nonpair-RDMA CPU-handler transport on booked 065/167, 8 A100. Each rank sends 12 BF16 H5120 rows, top8/E160, capacity256/QPs20, balanced routes.

One Graph retains six `Event(enable_timing=True, external=True)` objects: Graph start, dispatch start, dispatch receive-hook end, combine start, combine receive-hook end, Graph end. Capture once, warm twice, then replay three changed payloads. Every replay completes once before reading timestamps and validating exact source IDs, receive counts, payload and identity combine.

A separate noncaptured GPU origin event is recorded before capture. Origin-relative timestamps must be positive, correctly nested and agree with independently read component elapsed times. Three Graph start timestamps must refresh after deliberate 20ms CPU-only inter-sample gaps; a repeated capture-time timestamp is rejected. CPU sleeps, barriers, payload changes and correctness reads are outside the six event bounds.

The GPU spans include communication wait/polling and event-node overhead. This qualifies measurement correctness only; it does not prove uninstrumented full-model timing or synchronized cross-host critical paths. No per-layer host fences are introduced. Graph is reset before the native buffer is destroyed; existing Guard ownership and cleanup remain intact. Root launches the frozen single-mode spec separately.
