# 空闲 GPU 后台 benchmark

当前默认使用 `prefill_worker_ephemeral.py`，策略为 `retention: none`：持续并发 Prefill，每条输出 1 token，无 RPS 或队列限制，Guard 抢占结束后重新启动。后台不保存请求、workload、CSV、批摘要、历史或 cursor；仅替换 `<runtime_root>/nodes/<node>.json` 这一个不超过 32 KiB 的实时状态文件。Trace cursor 在内存中推进，进程重启从原始 trace 开头开始。详见 [PREFILL.md](PREFILL.md)。

旧 worker 与下述持久化协议保留为冻结的历史实现；coordinator 在 `retention: none` 时拒绝这些旧 worker。这个规则仅适用于空闲后台负载，前台实验仍按其自身协议保存结果。

# 历史协议：Guard-supervised BurstGPT capacity benchmark

Each model's own background Guard controller supervises one CPU traffic worker.
The coordinator only writes plans and reads control-plane status. It acquires no
GPU lease and sends no inference. Requests are assigned by global index modulo
current instance count, preserving the same global trace clock after partition.
All workload epochs are bound to exact deployment and lease IDs. Preemption or a
membership change cancels the phase; retries start a fresh measurement epoch.

The official BurstGPT trace contains timestamps and input/output token counts,
not text. This benchmark generates ephemeral ordinary GLM vocabulary token IDs
at the recorded input length and forces the recorded output length with
`ignore_eos`. Token IDs and generated text are never persisted. Length filtering
never clips a record. Request statistics retain actual server token counts.

`dataset.py` prepares a reproducible chronological eligible cohort. Calibration
uses input <=8192/output <=256; full-length validation removes individual length
caps while retaining the server's total-context limit and output >=2 for TPOT.
A finite cohort is an observed window, not proof of the whole population tail.
After calibration/confirmation, trace_stream.py reads the full original CSV
chronologically in 256-request batches, without individual length caps and
including one-token outputs (TPOT is null for those requests). A durable cursor
advances only after a complete valid batch; overload retries the same source
records. The finite-phase drain boundary is explicit in every result.

The load is open-loop. Relative BurstGPT arrival gaps and simultaneous arrivals
survive uniform time scaling; a phase's target RPS is the global request count
per nominal dispatch window. Clients never wait for an earlier response before
sending the next scheduled request. A dispatch delay above the configured limit
invalidates the phase instead of producing a catch-up burst or hiding overload
behind a concurrency semaphore.

User-selected queue policy is `no_backlog`: allow short admission waits, reject
sustained/growing queues using distinct fresh scheduler snapshots, and record all
request queue times and TP0 queue histogram deltas. `backlog_grace_s=2.0` is the
initial explicit threshold. Retractions, failed requests, missing/stale telemetry,
client dispatch lag or an in-flight safety limit invalidate a phase. Only proven
queue overload can provide a rate upper bound; instrumentation failures cannot.
`strict` mode additionally rejects observed queue occupancy or wait >5ms.

The coordinator increases rate, brackets overload and refines the interval to
10% relative resolution, then confirms with a longer sequence. Subset and full
lengths have independent rate validation. A passed point means no sustained
queue observed in that test window; it is not a mathematical global maximum.
After an invalid/aborted phase, the next phase waits until both servers drain.

Metrics use native SGLang streaming `/generate` and exact server token counts:

- TTFT: HTTP dispatch to the first generated token.
- TPOT: `(last generated token time - first token time)/(output tokens - 1)`.
- E2E: dispatch to stream completion, including final framing.
- Queue time: native per-request scheduler admission wait, separately from HTTP
  and client-side dispatch delay.

Per-request JSONL and CSV, per-node token/request totals, p50/p90/p95/p99 latency,
RPS, queue telemetry and partial/preempted results are saved privately. Latencies
in JSON are seconds; exported CSV latency columns are milliseconds. Completion
RPS includes draining time and therefore differs from offered dispatch RPS on a
finite phase. No failed or censored phase is a validated capacity point.

Run CPU verification using the serving Python environment:

```bash
python -m unittest discover -s tools/idle_gpu_benchmark -p 'test_*.py' -v
python -m unittest discover -s tools/idle_gpu_service -p test_benchmark_runner.py -v
```

To operate the current benchmark, use the immutable runtime `benchmark.json`,
`state.json`, `phase.json`, and `stages/` under
`experiments/idle_gpu_fleet_20261019/benchmark_20261019/`. The fleet's existing
manager controls all model/worker lifetimes. Use its standard stop command to
stop GPU work; the coordinator cancels the active plan when workers disappear.
