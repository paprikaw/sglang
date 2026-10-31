# Idle GPU model service

One independent, low-priority model replica per compatible free RUNNING GPU
booking. The CPU manager runs for at least seven days, follows current bookings,
and sends every launch through the shared Experiment Guard. It never requests
Slurm allocations and never preempts normal work. A normal Guard experiment
reserves a background replica's node, asks its owner to stop, and waits for the
ordinary cleanup/release gate before acquiring the node.

The user defines the benchmark later. Model readiness and health checks are
operational evidence; they are not throughput or latency benchmark results.

## Idle benchmark retention

The default background fleet uses `artifact_retention: "none"`. The coordinator
requires `prefill_worker_ephemeral.py`, propagates the policy into the frozen
profile/workload, and rejects a legacy result-writing worker in this mode.
The worker keeps BurstGPT cursors, requests and batch metrics in memory and
atomically replaces just `prefill/nodes/<node>.json` (at most 32 KiB). It creates
no stages, request rows, workload snapshots, CSVs, summaries or cursor files.
Each new background deployment starts reading its trace from the beginning.

The versioned `node_runner_qwen_ephemeral.py` and `node_runner_fp8_ephemeral.py`
discard worker and remote serving output. Their `server.log` retains only CPU
cache-loader records needed by Guard's unchanged exact-rank/source postcheck;
malformed, duplicate and fallback loader lines remain visible to that verifier.
Guard lease/cache/cleanup evidence and fixed latest live status remain required.
Foreground experiment output and model/KV caches are unaffected. Completed run
control directories are kept as small admission/cleanup records, without any
per-batch file creation. Historical benchmark payload cleanup was explicitly
authorized on 2026-10-22; it excludes foreground data and Guard evidence.

## Commands

Use an absolute JSON profile path and a persistent private runtime directory:

```bash
python3 tools/idle_gpu_service/manager.py run --config /absolute/profile.json
python3 tools/idle_gpu_service/manager.py status --config /absolute/profile.json
python3 tools/idle_gpu_service/manager.py stop --config /absolute/profile.json
```

`run` stays in the foreground and supports SIGTERM/SIGINT. The caller can launch
it under an existing host process supervisor. `stop` signals only the saved local
manager PID with matching process start ticks. A stopped manager's recorded
children can be recovered on the same host using their saved identities.

Manager profile keys:

- `runtime_root`, `guard_script`, `node_runner`: absolute paths.
- `python`: CPU manager/Guard interpreter (default: current interpreter).
- `gpu_registry_dir`: shared Guard registry; defaults to the existing workspace
  registry. The service never creates a separate registry.
- `duration_s`: at least `604800`; default seven days.
- `poll_s`: default `10`, maximum `30`.
- `booking_margin_s`: default `60`; stop before the booking expires.
- `known_node_specs`: node-to-profile mapping consumed by `node_runner.py`.
  A profile can use `disabled: true` plus `reason` to protect an incompatible
  existing cache. Such nodes appear with the reason in service status.
- `default_node_spec` plus `allowed_gpu_types`: optional reusable node profile
  for newly discovered compatible hardware. The GPU type must match a complete
  typed Slurm GPU token. Unknown hardware is skipped.
- `cached_weights_only`: when `true`, new attempts require an existing active
  CPU daemon for the current node/job and pinned source/cache scheme, plus a
  nonempty all-ready recorded snapshot. Missing, building or incompatible caches
  are skipped. Existing admitted capacity, headroom and socket are preserved.
  Snapshots are candidates only; Guard must confirm the exact live model with
  `model_cache.on_miss: fail`, without building or replacing weights.
- `cpu_cache_registry_dir`: shared CPU daemon ledger directory; defaults to
  `cpu_weight_daemons` beside the shared GPU registry. Missing/corrupt metadata
  pauses new admission and leaves active attempts running.
- `prepare_timeout_s`: default `60`; preparation is a cancellable local child.
- `retry_backoff_s`, `max_backoff_s`: defaults `30`, `600`.
- `stop_grace_s`: default `150`, allowing Guard to complete owned cleanup.

The remaining model, source, cache and launch settings belong to the node runner.
Its `prepare` action must print exactly JSON `{ "spec_path": "/absolute/spec" }`
and do only local preparation. The manager validates that spec before starting
Guard. It writes `attempt.json` with `stop_at`, booking identity and service
identity before preparation; the runner independently honors that bound.

Each attempt has its own run directory and logs. `state.json` aggregates
`remote_status.json` or `task/progress.json` when present. A single-owner `flock`
prevents two managers from using the same runtime directory. State files are
written atomically with mode `0600`; the root directory uses mode `0700`.

Discovery failures stop owned attempts and pause new admission. Normal or unknown
registry owners and active preemption reservations block launches. Registry
observations remain advisory: Guard makes the final atomic admission decision,
maintains the background lease and handles cooperative preemption. The service
does not infer that unrelated non-Guard GPU work is idle.

## CPU-only verification

```bash
python3 -m unittest discover -s tools/idle_gpu_service -p 'test_*.py' -v
```

The manager suite covers exact booking discovery, reservation races, protected
profiles, hardware matching, spec validation, cancellation, deadlines, expiry,
process identity and bounded retries without requesting GPUs or starting models.


For the current fleet, `cached_weights_only: true` selects only compatible
existing weight daemons from the shared CPU ledger. These metadata are advisory;
Guard verifies the actual model bank with `model_cache.on_miss: fail` before a
server starts. A missing, incomplete, incompatible, or unavailable cache is
skipped. Background work never builds, replaces, or retires a weight bank.
The existing daemon's admitted capacity, headroom and socket are preserved.
Normal experiments determine the model/cache; idle serving follows that state.

The persistent caches currently used here live in CPU RAM. Each idle replica
loads its existing bank onto its node's four GPUs under an exclusive background
lease; a normal resource request stops the replica and checks GPU cleanup before
handoff. CPU banks remain available for the foreground owner.
