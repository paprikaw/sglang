# AgentX with cache-preserving P/D and EP/EP1 allocation

User goal, 2026-10-30: fix technical compatibility problems and, using the two
existing H100 nodes, test whether real P/D disaggregation is needed and whether
EP versus EP=1 (single GPU or pure PP) dynamic allocation adds value. Combine
new AgentX evidence with existing fixed-phase data and produce an indexed
Markdown report. BTD segregation is interpreted as P/D disaggregation from the
conversation; no change to model or workload scope to avoid hard cases.

Required conditions:
- HiCache on all roles, retain GPU KV and session/DP affinity between turns.
- Correct GDN/recurrent state, KV prefix locks, transfer and host restore.
- Real P->D handoff; whole-request routing is not P/D separation.
- Fixed equal eight-H100 budget, existing booked011/171 nodes only.
- Include EP4 and competitive EP1 candidates (PP2 replicas and single GPUs).
- Full AgentX dependency/fan-out/tool-wait replay, or clearly scoped derived
  workloads with enough coverage to test the hypothesis; no official score
  claim without official protocol. Means and absolute latency values.
- Compare strongest fixed colocated, fixed homogeneous PD, fixed heterogeneous
  PD, P:D-only adaptation, and joint P:D plus layout-family adaptation. Dynamic
  benefit includes switching/state/cache/batch-fragmentation costs, not oracle
  component sums. If evidence rejects benefits, report that finding honestly.
- Preserve source tags, raw trials, failure evidence, full-model weight cache
  admission, Guard leases and owned cleanup; do not overwrite earlier artifacts.

Stages: fix and qualify -> matched static AgentX experiment -> dynamic necessity
and measured policy/transition evaluation -> independent evidence audit/report.

Baseline source: 0f2f1990994774472b1089d2b0682f0bf926a6b4.
Working branch: codex/agentx-gdn-pd-hicache-20261030.
Previous completed evidence: ../agentx_pd_feasibility_20261030.
