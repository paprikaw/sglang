# PP2 MTP first-divergence correctness diagnostic

This diagnostic reuses the detalign source harness and identical semantic corpus bytes. The runtime engine commit is pinned separately from the original f415858 semantic source-code corpus; no re-tokenization and no synthetic KV occur.

Two existing A100 nodes 065/167; full canonical CPU weight bank is required with on_miss=fail. Native deterministic mode and explicit Prefill truncation alignment 1024 apply identically. Workload: one atomic B8 submission, P4096 Q1 →64, two fixed B4 PP slots. Real model Prefill constructs native process-local HiCache; before each cohort exact host P4032 and GPU0 are proved on all eight ranks.

Two initial timed-/repeat- cohorts have no witness clones/readback and require all-eight-rank zero witness counters. Two subsequent untimed witness-/witness-repeat- cohorts run in the same process. A bounded tail-stage witness records owned GPU snapshots for their first 16 Verifies per slot. Device-to-host readback and file serialization run only at fully drained Scheduler idle. Controller must validate four tail-rank flush receipts, re-parse every .pt payload and check content SHA, size, cohort and full B8/two-slot coverage before the next cold command or normal shutdown.

The CPU acceptance reference traverses recorded retrieve_index, child and sibling links and validates exact accepted indices, token values and lengths. Unsupported metadata is reference-inapplicable, not evidence of a native bug. Request-global retrieve offsets are normalized only within validated own-request ranges; differing whole-batch masks make request-level mask equivalence unconfirmed.

All eight initialized ranks save loaded Humming _id2kernel metadata after model/Graph warmup and at idle command finalization: every registered dense and grouped_masked kernel must be SM80, batch-invariant=true and StreamK=false, with effective shapes and cubin/source identities. This is registered-kernel metadata, not a launch trace.

The witness installs after all other startup hooks. CUPTI instrumentation is not installed, and no separate GPU profile runs. Snapshot clones/readback are restricted to the separate untimed witness cohorts. Initial timing is retained, but performance_comparison_claim_valid remains false pending independent correctness and witness review. This run diagnoses repeatability and first divergence and does not authorize a speedup claim.
