# Shared idle-GPU benchmark reconciliation

## 2026-10-22: discard idle benchmark results

The user authorized removing historical idle benchmark payloads after inode
exhaustion recurred. Removed 427,300 stage files plus 170,915 directories
(6,437,253,421 bytes) and 211 auxiliary/history files (77,331,868 bytes).
Foreground results, sources, datasets, model weights, KV and Guard cache/lease/
cleanup evidence were excluded. Historical server.log files remain as the
original Guard cache-loader evidence; new runs retain only loader records.

The shared private config now enforces `artifact_retention: none`, uses
`prefill_worker_ephemeral.py` and the versioned Qwen/FP8 ephemeral runners.
Requests, synthetic tokens, batch results and trace cursors stay in RAM. Only
a latest `prefill/nodes/<node>.json` (maximum32KiB) is replaced for live status;
no stage/request/CSV/summary/cursor/serving-log output is retained. Lease, cache
admission and verified cleanup remain required. Service/attempt control files
are retained, without per-batch file creation. This policy supersedes the older
append-only storage notes below; normal experiment result retention is unchanged.


Enabled on 2026-10-20. The canonical Experiment Guard calls
`/home/bxb1/experiment-guard/skills/llm-experiment-guard/scripts/backfill_trigger.py`
on match, prepare, check, and before/after run. Background children skip the hook.
Dashboard serving stays read-only. Existing frozen Guard bundles remain immutable;
the shared coordinator polls bookings and leases every 10 seconds to cover them.

## Policy

- Discover all currently running GPU bookings, including newly added allocations.
- Normal experiment leases and pending preemption reservations take precedence.
  Background managers use the existing cooperative Guard preemption protocol and
  automatically start another benchmark round after the foreground lease ends.
- Select a qualified complete serving recipe matching the current live CPU model
  bank and engine revision. Verify worker freshness and run identity before
  reporting a node as benchmarking; a manager heartbeat alone is insufficient.
- For an empty model bank on a verified four-A100 node, build the validated small
  Qwen fallback cache. After the first successful start, seal its control profile
  to reuse-only, so future restarts cannot rebuild an obsolete model after an
  experiment changes the cache. Frozen run inputs are never rewritten.
- Unknown hardware, incompatible or partial model banks, stale daemon identities,
  unregistered GPU processes, and protected leases produce explicit waiting states.
  They do not authorize cache eviction or arbitrary model/topology construction.
  Qualified loader layouts cover BF16 Qwen3.6-35B-A3B, FP8
  Qwen3.5-122B-A10B and AWQ GLM-4.7, each using four A100 GPUs. Model paths
  are discovered from the current bank rather than assigned permanently.
- Never submit bookings or signal foreign GPU processes. A singleton lock prevents
  concurrent agents from creating duplicate coordinators. Managers have seven-day
  windows bounded by the current booking expiry; continuation requires a valid
  allocation and the coordinator rechecks it on each cycle.

The precise node169 HY3 partial CPU-weight bank was retired with the user's
explicit authorization, then replaced with a complete Qwen small-model bank.
This did not delete disk model weights or KV checkpoint files. Other protected
banks retain their normal protection.

## Operation and evidence

- Shared configuration:
  `/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/backfill/config.json`
- Coordinator source: `tools/idle_gpu_service/cluster_backfill.py` relative to the
  autoscaling workspace. The configuration pins its SHA256; changing source needs
  an intentional configuration hash update and restart of the verified owner.
- `state.json`: coordinator PID/start-ticks/host, audit timestamp, node decisions.
- `services.json`: adopted and created manager roots, recipe IDs, bootstrap seals.
- `events/`: CPU-only Guard invocation audit events; `supervisor.log`: coordinator
  errors; `observations/`: validated canonical model-bank key discovery.
- `all_nodes_state.json`: read-only fleet-monitor output. Existing resource
  dashboard on port 19510 reads this fleet root.
- `retire_169/`: exact authorized retirement evidence. Receipt files are private.
- `validation/final_snapshot.json`: verified four-node worker and dashboard snapshot.
- `validation/normal_handoff/result.json`: normal maintenance claim preempted
  node169 benchmark in 6.8014 seconds, then a new benchmark run automatically
  resumed and completed 1432 requests at the recorded verification point.

Validation: 10 reconciliation policy tests and 5 Guard-trigger tests passed;
all Guard JavaScript templates passed syntax checks. The broader existing Guard
suite had 3 pre-existing CPU-budget exception-type errors in
`GlobalBudgetRemovalTests.test_removal_keeps_node_limit_headroom_and_active_lease_for_both_admissions`;
its log is `/tmp/guard_backfill_tests_20261020.log`. Those unrelated checks were
not reported as passing. GPU utilization is sampled telemetry and varies with
the prefill workload; a running benchmark does not guarantee constant 100% use.

## Expired-lease recovery fix, 2026-10-21

The coordinator originally interpreted raw active registry records as occupancy
even after their heartbeat TTL. This stalled recovery before the next Guard
claim could perform its normal atomic expiry transaction. The advisory lease
view now applies Guard's TTL and quarantine-grace rules, while retaining the
serial GPU-process audit and locked Guard claim before starting any model.
An expired lease does not certify GPU process cleanup. Live foreground leases,
unexpired quarantine and pending preemption reservations remain protected.
The policy suite now has 13 passing tests, including stale records coexisting
with a live foreground lease, quarantine within its cleanup grace, and a
remaining GPU process preventing launch despite an expired lease.

Recovery audit files are `validation/expired_lease_audit_20261021.json`,
`validation/dead_bank_cleanup_20261021.json`, and
`validation/dead_manifest_archive_20261021.json`. Only node065's exact old
daemon identity was confirmed dead and its persistent reservation released;
its matching old manifest was archived. Node069 and node167 live CPU banks
were preserved and reused. Node065 has a new Slurm booking, so it builds a
fresh qualified full-model cache through normal Guard admission.
`validation/expired_lease_recovery_20261021.json` records successful requests
from all three currently booked nodes after recovery; node169's prior booking
has ended, so it is not restarted without an allocation.

## Persistent dashboard and quota recovery, 2026-10-21 evening

The later repeated failure has direct evidence: all three managers, the
coordinator, monitor, and Guard heartbeat writes failed with EDQUOT (errno122)
around 18:25 Melbourne time. The project's hard limit is 3,000,000 inodes;
per-64-request stage directories consumed about 572,000 inodes in four old
background services alone. Lease expiry was a consequence of failed persistence,
not an expired Slurm booking. Increasing the lease TTL would not fix this.

Three enabled user services on spartan-login2 now own the fixed-port dashboard,
read-only monitor and coordinator. Each has Restart=always and RestartSec=5.
Unit rendering/install source is `tools/idle_gpu_service/install_user_services.py`.
Other hosts' Guard hooks audit the shared owner rather than launching unmanaged
competitors. For diagnosis/recovery use:

```bash
systemctl --user status gpu-resource-dashboard gpu-resource-monitor gpu-idle-backfill
journalctl --user -u gpu-idle-backfill -n 40 --no-pager
systemctl --user restart gpu-resource-dashboard
```

The dashboard retains port19510, so an existing forwarded browser page reconnects
without changing its URL. A forced SIGKILL test recovered HTTP200 in 5.402s;
see `validation/dashboard_systemd_recovery_20261021.json`.
User services are enabled at login. The OS denied enabling user linger, so
operation after all login sessions end or before the next login after host reboot
is not guaranteed; that setting requires administrator permission.

All historical measurements remain in place. Instead of deleting their source
files, new benchmark cohorts freeze `prefill_worker_append.py`: one stage
directory per deployment/node, append-only phase-linked workloads, raw requests,
CSV metrics and summaries, plus bounded latest-summary/progress files. Twenty
batches in the regression fixture create the same number of files as one batch.
The workload, trace cursor commit, eight concurrent lanes and preemption gates
are unchanged. Old frozen workers and runners remain intact. New versioned
resilient runners ensure report I/O failure never interrupts owned cleanup.

Only 3457 regenerable, Git-ignored Python bytecode files were removed after
backup and SHA256 verification; no sources or measurements were removed.
Backup: `/tmp/dashboard-regenerable-pyc-20261021-1792577092296505362.tar.gz`;
audit: `/tmp/dashboard-pyc-cleanup-result-20261021.json`.
Coordinator state writes now retry storage faults, and background managers drain
when fewer than1024 free inodes remain, preserving metadata headroom for normal
experiments and cleanup. The monitor retains old timestamps on failed writes.

Final verification: all three current nodes (065/069/167) completed real requests
using the new append layout, with exactly six stage files each. Killing only the
systemd-owned coordinator recovered in5.345s without restarting any model manager
or changing a benchmark run ID; request counts increased throughout. Evidence:
`validation/coordinator_systemd_recovery_20261021.json` and
`validation/persistent_services_final_20261021.json`.
CPU validation:102 service tests,5 append-worker tests and7 trigger tests passed;
all7 Guard template JavaScript files passed syntax checking. The Guard full suite
ran379 tests with0 failures and3 previously known CPU-budget exception-type errors.
Log: `/tmp/dashboard_guard_tests_20261021.log`.

## Current-weight recovery and isolated probes, 2026-10-23

Nodes065/167 had completed foreground work and retained complete GLM-4.7-AWQ
canonical CPU banks, but the background catalog contained only two Qwen model
paths. They therefore waited indefinitely for a matching recipe. The old UI
label incorrectly suggested incomplete model coverage. A separate systemd PATH
issue produced repeated `FileNotFoundError: scontrol` during uncached-node
qualification; that exception aborted the entire reconciliation pass.

The coordinator now uses an absolute, configurable Slurm path and isolates
hardware/cache/GPU-probe errors per node. Missing loader configuration and
incomplete coverage have distinct dashboard labels. GPU utilization collection
remains independent of both foreground leases and benchmark readiness.

`tools/idle_gpu_service/cache_model_discovery.py` separates checkpoint discovery
from qualified loader templates. It scans bounded checkpoint metadata catalogs,
then authenticates the physical node, daemon PID/start ticks, UID, Slurm job,
source revision and ready shard sizes/keys. Only a complete fingerprint match
is a model candidate. No weight payload is read or cache rebuilt by discovery.
The canonical Guard invocation hook records one latest metadata path hint per
node from SHA-verified node specs, so paths outside the default HF catalog can
also be discovered. Hints alone never establish readiness or authorize loading.
Existing frozen Guard bundles and active runners are unchanged.

A discovered checkpoint with a qualified architecture/quantization/layout and
within its validated weight budget receives a fresh model path, config/index
revision, exact node spec and daemon-bound plan ID. Qualified layout hashes
exclude checkpoint name/provenance/version labels, not tensor shapes, dtypes or
quantization. This supports new checkpoint directories/revisions without adding
model-path entries. Different layouts or engine revisions need qualification;
unknown or partial banks wait explicitly without falling back to historical
weights or evicting anything. Most recent Guard path hints fix the model choice only
when they match a complete live bank; an unsupported current model then waits
rather than selecting older weights. Without such a match, largest qualified bank,
then path and template ID, is the stable selection policy.

The new versioned `node_runner_cache_model_ephemeral.py` preserves the existing
TP4/PP1/EP1 full-model recipes and FP8 entry adapter, checks checkpoint metadata
again before loading, and requires the admitted daemon/group to match discovery.
It still uses ordinary background/preemptible Guard leases, exact-rank cache
loader evidence and verified cleanup. Benchmark retention remains disabled.
The only continuing benchmark telemetry is the overwritten bounded live JSON.

Live CPU discovery matched GLM65-shard banks on065/167 and the QwenFP8 39-shard
bank on069. CPU-only prepare passed for all three dynamic plans. Both GLM
replicas completed actual requests, automatically yielded to the foreground
`glm47-verify-attention-ab-20261023-original_vs_causal_decode_r1` experiment,
and resumed new rounds after release. The original069 deployment was preserved. A controlled065 background restart
then exercised the new dynamic runner with the real GLM bank; readiness and
successful streaming requests passed, while069/167 continued unchanged.

Service regression:183 tests passed. All7 Guard template JavaScript syntax
checks passed. Log: `/tmp/idle-gpu-current-model-tests-20261023.log`.
Full Guard validation:379 tests,0 failures and the same3 pre-existing CPU-budget
exception-type errors; log `/tmp/guard-current-cache-discovery-tests-20261023.log`.

## 2026-10-24: booking rollover left a dead CPU-bank reservation

069 job 33042490 timed out at 00:57 and job 33463295 started at 00:58.
The old daemon process (PID 256657, start ticks 407165598) and socket were
absent on the physical node, but the persistent CPU ledger retained the old
job. Recipe selection therefore waited forever before reaching Guard admission.

The coordinator now performs an advisory exact-process death check only on free,
unreserved nodes. It requires the current Slurm cgroup, physical hostname, UID,
a canonical full-bank identity and an absent socket. Live, partial, unknown or
unreachable banks remain protected. A qualified fallback gets a unique
new-job/old-daemon recovery plan containing the exact old manifest. Its existing
Guard dead-daemon recovery helper reauthenticates the acquired background lease,
current job and unchanged ledger snapshot, repeats local death proof under the
ledger lock, preserves the old manifest and releases only that dead reservation.
The new job then builds a qualified complete model through normal cache
admission; old entries are never rebound to the new job. Current ready banks
still use dynamic model discovery. Foreground leases and reservations take
precedence before this recovery path. No benchmark result retention is added.

Validation: idle_gpu_service suite 190 tests passed, including live daemon,
occupied socket, unsupported runner/partial cache, unreachable/mismatched proof
and new-job bootstrap contract coverage.

Live verification: Guard released only old daemon a00e45f8780e4b41a977c13e9ded05f4
with action released_dead, signals_sent=false and cache_evicted=false. The new
069 daemon 5bd7b1edb5df41da9ff11f27a427b8bf belongs to job 33463295. The
qualified Qwen3.6-35B-A3B bootstrap passed streamed/non-streamed API readiness,
all four ranks appended canonical-loader records, and the continuous prefill
worker processed 57 successful requests with zero failures at first validation.
065/167 foreground experiment ended independently during recovery and their
existing managers automatically resumed benchmarks through ordinary Guard.

## 2026-10-25: H100 backfill support

The coordinator previously hardcoded gpu:A100:4 in its Slurm inventory gate.
Recipe qualification now declares qualified_gpu_types; legacy recipes default
to A100. The Qwen3.6-35B-A3B TP4/EP4 loader explicitly admits A100 and H100
whole nodes with exactly four GPUs. Untyped gpu:4 requests are resolved using
the physical node's typed Slurm inventory, never its hostname. Unknown types,
mixed GPUs and other GPU counts remain blocked. The selected hardware identity
is saved in the new frozen serving profile. Current-bank checkpoint discovery
continues to match complete weights and compatible loader layouts; H100 does
not grant blanket support to other loader recipes.

Live target: spartan-gpgpu178, job 33646188, 4x NVIDIA H100 80GB HBM3,
compute capability 9.0, driver 570.211.01. Slurm Gres=gpu:H100:4.
Engine remains clean pinned commit 870269702285d94bdd926a50cc29ad1a20bcf60c;
no engine code, compiled extensions, active frozen runs or A100 launches changed.
The previous job's dead bank is reconciled using exact leased Guard recovery,
then the new job builds the qualified complete model. Continuous eight-lane
BurstGPT prefill, background preemption, automatic recovery and artifact_retention
none remain in force. Only bounded current telemetry and Guard evidence persist.

Validation: idle_gpu_service suite 195 tests passed, including explicit H100
admission, legacy A100 defaults, unknown/untyped/mixed/wrong-count rejection,
and H100 reconciliation forwarding frozen hardware qualification.

Real H100 validation passed: streamed and non-streamed chat, model identity and
health passed on deployment idle-spartan-gpgpu178-1792894057-spartan-gpgpu178-df9db1aab1.
TP0..TP3 each emitted a canonical CPU weight loader record (3.6-4.0 seconds).
The continuous eight-lane prefill worker completed 713 successful requests at
first validation, no worker error, and coordinator state verified_live_worker.
The bootstrap profile was sealed to cached_weights_only=true/on_miss=fail.
Four pre-existing A100 benchmark deployment IDs remained unchanged.

## 2026-10-25: recovery with mixed hardware qualifications

After H100 foreground runs h100_ep2_pp2_r1..r7, the current canonical bank
contained both Qwen3.5-122B-A10B-FP8 and Qwen3.6-35B-A3B. No authenticated
latest-model hint was present, so discovery ranked the larger complete bank
first. That FP8 loader remains A100-only. The coordinator rejected its hardware
and stopped without trying the already cached, qualified H100 small model.
The initial H100 qualification tested cold startup but missed this mixed-cache
post-experiment recovery case.

Selection now intersects the discovered complete model candidates with loader
hardware compatibility before reconciling the current manager identity. It
uses the first qualified ready bank in discovery order, preserving an explicit
preferred-model candidate restriction. It does not rebuild/evict either bank,
rebind job identities, authorize the A100-only FP8 adapter on H100, or stop an
existing compatible manager because an incompatible earlier candidate differs.
Bootstrap recovery retains its exact old-daemon snapshot and never changes the
model implicitly. Foreground and reserved-node checks still precede selection.

Validation: 201 idle_gpu_service tests passed, including mixed cached FP8/small
models on H100, explicit preferred-model rejection, unchanged A100 ranking,
bootstrap recovery identity preservation and compatible-manager preservation.

Live recovery verified on deployment
idle-spartan-gpgpu178-1792898924-spartan-gpgpu178-9d2548fa74. All four ranks
loaded canonical CPU-bank weights; streamed and non-streamed chat passed.
Continuous prefill completed 226 successful requests with no worker error
at first validation. Cached-only/on_miss=fail admission preserved both ready
models and all four A100 benchmark deployment IDs.

## 2026-10-29: GPFS registry locking recovery

Five background services repeatedly stopped after GPU lease heartbeats disappeared. The runs recorded `GPU lease is absent or expired` during cache admission and `GPU lease heartbeat expired` after launch. Space and inode quotas were healthy. A disposable cross-host lock probe allowed remote `flock` while the login host held it, whereas a POSIX record lock blocked the remote contender. A subsequent mixed flock/record-lock rollout still stalled: strace captured a heartbeat thread blocked inside `flock`. It was not accepted as a successful recovery.

Canonical Guard now uses one POSIX record-lock protocol, with an inode-keyed process-local mutex to serialize threads. Lock ownership ends by closing the descriptor before releasing the mutex. GPU leases, CPU daemon budgets, and CPU cache registries use the same protocol. Lease TTL and fail-closed preemption behavior remain unchanged. Background profiles select a separately frozen Guard bundle through `guard_override`, validating the Guard script and three registry dependency digests; historical frozen bundles are preserved. Active service bundle: `guard_bundle_gpfs_readonly_20261029`. Never mix flock and POSIX locks on the same shared lock inode.

Regression coverage includes a subprocess with flock disabled, concurrent local threads, and rejection of modified Guard scripts/dependencies. Live acceptance additionally requires all five current runs to process requests and keep fresh leases across multiple TTL periods. Only bounded current telemetry is retained; no benchmark request/result archives are enabled.

Cross-host writer-lock handoffs still delayed heartbeat acquisition after the single-protocol change. Ownership assertions were unnecessarily using the mutating/pruning writer context. `assert_owned` and `assert_owned_nodes` now authenticate an atomic read-only snapshot, explicitly rejecting expired leases and preserving token/spec/node checks. A live disposable-registry probe from gpgpu011 validated ownership in 0.0141 seconds while the login host held the writer lock. This keeps remote cache admission off the heartbeat writer path. Regression coverage confirms that assertions do not acquire a writer lock or modify registry bytes and still reject expired receipts. Full Guard checks: 382 tests with the same three pre-existing CPU-budget fixture errors; idle-service tests: 203 passed.

The second recovery loop came from the ephemeral request worker: once started, a stale/temporarily unready aggregated readiness record was treated as a terminal stop, so the manager reloaded an otherwise leased model. A separately frozen worker, `tools/idle_gpu_benchmark/prefill_worker_ephemeral_lease.py`, retains initial readiness validation but continuously authenticates its exact lease/current run thereafter; request transport errors keep the existing bounded retry, while preemption, lease loss, replacement, cancellation and deadlines still stop dispatch. The original frozen worker is preserved. CPU regression forces readiness 120 seconds stale and deployment status unhealthy after startup, verifies continued requests, then verifies foreground preemption stops them. Benchmark suite: 141 tests passed; idle service suite: 203 passed (190 subtests).

Live acceptance completed at approximately 21:27 AEDT: all five nodes (gpgpu011, 065, 069, 167, 171) ran the read-only-auth Guard bundle and lease-authorized ephemeral worker. Successful request counters increased on every node across repeated samples; all five leases stayed fresh across multiple 15-second TTL periods with no new heartbeat-timeout releases after final rollout. Coordinator, resource monitor and dashboard user services were active. The check validates recovery over the observed interval, not a guarantee against future failures. Existing foreground-preemption behavior remains covered by the CPU worker regression.
