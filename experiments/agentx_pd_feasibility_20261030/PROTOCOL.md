# AgentX data and real P/D feasibility

2026-10-30. Existing H100 bookings only: 011 and 171, four GPUs per node.
Official full and 256k corpora downloaded at immutable revisions with LFS SHA256 verification.
Qwen3.6-35B-A3B native context is 262144; use 256k corpus for follow-up.

First gate: equal four-GPU paired replicas versus actual Prefill/Decode split.
011 uses EP2 + EP2; 171 uses PP2 + PP2. Each node has its own colocated baseline;
cross-node numbers are not treated as matched layout comparisons.
Require serial greedy output agreement on fresh and extended requests before
timing the PD screen. Transfer must include GDN state. No fake backend.

Screening subset is selected before GPU measurements: eight five-turn contiguous
top-level windows of one source model, 1024–65536 input and 16–512 output tokens,
non-overlapping calls, 0–2 second gaps, at least 1024 prefix tokens reused.
One window per source session; evenly sample the eligible windows by new-input/output
ratio. Retain exact input/output lengths, hash-block identities and inter-turn gaps.
Synthetic token IDs reconstruct prefix geometry; they are not original user content,
and they do not reproduce real router/expert distributions or model quality.
No subagent/DAG coverage claim and no official AgentX benchmark score.

Native PP scheduling is used for qualification. C1 and C8 runs use three repetitions.
All episodes include an initial cache-primer request; its cost remains in total episode
wall time and is excluded only from separately labelled later-turn mean TTFT/TPOT/E2E.
HiCache is mandatory on all roles: 32 decimal GB host pool per GPU, identical 128 GB
per node, with a conservative 128 GiB Guard reservation. GPU KV capacity is automatic
at static memory fraction 0.95; do not artificially cap total KV tokens. Enable
PD Decode radix caching and preserve session/DP-rank affinity. Verify actual HiCache
state and native GPU/CPU hit records; never infer CPU restoration from flags alone.
Flush at independent repetition boundaries only, never between turns in an episode.
Transferred bytes and Prefill/Decode cache hits must be reported separately.
All raw repetitions survive. No best-of or maximum throughput selection.

This gate does not prove dynamic allocation useful. A full evaluation must compare
the best equal-eight-GPU colocated deployment, best static homogeneous PD, best static
heterogeneous PD, P:D-only adaptation, and joint P:D plus EP/EP1-family adaptation.
Preserve the official graph replay for that evaluation, include cache/state migration,
weight duplication and configuration transition costs, and separate routing from
actual changes in GPU/instance proportions.
