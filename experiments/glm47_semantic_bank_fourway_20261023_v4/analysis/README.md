# Four-arm activity and performance analysis

This directory analyzes saved files only. It does not start an engine or use a GPU.

```sh
python3 analysis/stage_activity.py /absolute/run/ep8_mtp_on
python3 analysis/stage_activity.py /absolute/run/ep8_mtp_off
python3 analysis/stage_activity.py /absolute/run/pp2ep4_mtp_on
python3 analysis/stage_activity.py /absolute/run/pp2ep4_mtp_off
python3 analysis/summarize.py /absolute/run
```

Use a completed run selected by the controller. Prepared runs are not measurements.
The files `stage_activity_summary.json` and `fourway_summary.json`/`.md` are new
derived artifacts; immutable source traces/results are never rewritten.

## Timing boundaries

- Production TTFT/TPOT/E2E and throughput come only from the two **unprofiled**
  atomic B8, Q1→64 cohorts. The cohorts remain separate. TTFT includes native
  CPU HiCache→GPU load and initial tail forward. Persistent source-bank import,
  weights, calibration and profiler startup are outside timing.
- P4096 is the raw context. Native page/bigram matching retains 4032 cached
  tokens, so the initial forward recomputes 64 context tokens plus Q1.
- Component activity uses a separate cohort. CUDA-only coarse markers identify
  target Decode/Verify and draft API windows. Starting/stopping CUPTI is legal
  only while the scheduler and HiCache are idle. There are no per-layer CUDA
  event fences or synchronizations in the model.
- A steady per-device window starts at target #1 and ends at the start of target
  #12: **11 closed local B4 microbatch cycles**, including intervening draft,
  transport and idle. The last unclosed target/draft cycle is never divided by
  output tokens. PP has two B4 request slots, so a microbatch cycle is not a
  same-request token interval.
- Output normalization uses CPU output-ID advances recorded after native result
  processing. Missing records produce `null`, not an assumed acceptance width.
  GPU work per committed local output token is not request TPOT.

## Attribution and limitations

Actual calibration GPU correlation IDs connect to unique CUDA runtime launches
and innermost explicit CPU scopes. Exact names require independent existing
Graph warmup witnesses; conflicting generic names are rejected. An entire exact
ordered Graph name/phase sequence may resolve generic names only if it reproduced
in held-out warmup and matches on the **same** calibrated rank. Otherwise the
fallback requires a complete exact launch fingerprint: name, grid, block,
registers per thread and shared memory. Its native CPU-scope/correlation phase
must agree in both warmups of the same graph, on the same rank and target/draft
role. Conflicting phases, missing geometry and other ranks/roles remain unknown;
no partial matching or timing-based redistribution is used. The original
name/sequence unknown counts and exclusive wall time remain alongside the
post-fingerprint coverage. Other TP ranks do not inherit positional or geometry
maps.

Attention includes QKV/output projections, RoPE and KV store. Routed expert GEMMs
and activation are separate from shared expert and dense FFN. Dispatch and combine
are separate from TP collectives and PP boundary operations. Kernel time can
include polling/waiting and does not measure wire transfer alone.

For each rank, categories form an exhaustive disjoint wall-clock partition.
Same-category overlap is counted once. Cross-category overlap has its own bucket.
Idle means no traced device activity, not a proven pipeline bubble. No rank/stage
maxima or clocks are summed into a synthetic TPOT. Stage/group means/min/max are
rank-local descriptive statistics only. Profile overhead must not be used to
adjust unprofiled production metrics.

Exact on/off and cross-topology generated IDs/text are compared rather than
assumed equal. P99 over eight requests is just the observed sample maximum.
