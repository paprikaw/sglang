# Idle benchmark recovery — 2026-10-20

> Superseded for idle result retention on 2026-10-22: the active cluster fleet
> uses `artifact_retention: none` and `prefill_worker_ephemeral.py`. Historical
> idle stage payloads were removed by explicit user request; latest live state
> replaces one bounded JSON per node. Guard evidence, foreground results and
> model/KV caches remain. See ../cluster_backfill_20261020/README.md.


Restored from current authenticated canonical CPU caches, without cache replacement.

- spartan-gpgpu065: Qwen3.6-35B-A3B, Attention TP4 / DP1, MoE EP4, PP1.
- spartan-gpgpu069: Qwen3.5-122B-A10B-FP8, Attention TP4 / DP1, MoE EP4, PP1. Uses the existing qualified A100 Marlin adapter and receipt, installed in spawned scheduler workers.
- spartan-gpgpu169: waiting for full HY3 layer coverage. Node 168 booking expired; never serve a partial model.

Each independent manager runs for seven days at most, and stops its exact owned model before booking expiry (90 second margin). Guard background priority, automatic foreground preemption, private receipt validation, exact-node cache admission, reuse_only and on_miss=fail apply. After foreground release, each manager retries its qualified Qwen recipe. Changed model/source or unsupported coverage waits; this restoration does not implement a universal recipe planner or submit bookings.

Workload: official BurstGPT input lengths, synthetic token IDs, eight concurrent requests per instance, one output token, context <=32768, no RPS/queue gate. Each model starts a separate trace/result cohort; previous evidence remains untouched. Metrics and durable cursors are in each node's prefill directory. TTFT/e2e are recorded; TPOT is null for one output token.

Dashboard: http://127.0.0.1:19510/ . The separate read-only monitor merges manager and worker state and samples GPU telemetry sequentially. Dashboard lifecycle record is also updated in ../gpu_resource_overview_20261019/launch.json .

Managers: tools/idle_gpu_service/manager.py run --config <this directory>/<node>/profile.json . Process identities are recorded in each node's state.json. Do not start a duplicate manager; stop only a matching PID/start-ticks identity through SIGTERM to allow Guard cleanup.
