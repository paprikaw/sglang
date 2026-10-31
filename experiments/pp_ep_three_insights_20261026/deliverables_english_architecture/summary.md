# EP / PP deployment and proposed autoscaling architecture

Six English slides. The last slide proposes a lightweight Profiler, a Performance Model,
and an Autoscaler. The Profiler captures Expert execution and communication by payload
and domain. The model predicts latency/throughput for memory-feasible PP x EP layouts and
selects an initial configuration. The Autoscaler reuses the predictor to compare adding
pipeline stages with widening the expert group as load grows. Runtime measurements
calibrate the predictor. Model details and design scope are in the speaker notes.

The architecture uses editable native PowerPoint shapes and connectors. A standalone
PNG and one-page PDF are in figures/. The preceding five slide XML files, preview images,
plots, and numerical tables are unchanged from the verified two-domain-case revision.
No new GPU experiments were run. Rebuild: build_english.py --with-architecture.
