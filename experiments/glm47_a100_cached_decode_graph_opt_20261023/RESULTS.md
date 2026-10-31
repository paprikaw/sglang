# GLM A100 Decode: matched CUDA Graph optimization

## Result

Both complete-model arms passed; Guard postcheck passed and global node lease was released with verified cleanup. This is one paired cohort, not a statistical significance claim.

| Configuration | Mean TTFT (s) | Mean TPOT (ms) | Median request TPOT (ms) | Request P99 TPOT (ms, N=8) | Mean E2E (s) | Output tok/s | Steady token16..112 interval (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| EP8 | 2.5357 | 152.4875 | 150.5102 | 166.3379 | 21.9023 | 46.7466 | 150.6578 |
| PP2 x EP4 | 1.7461 | 150.8185 | 150.7579 | 158.8106 | 20.9005 | 48.8999 | 148.1734 |

PP mean TPOT is 1.095% lower; steady cadence is 1.649% lower; output throughput is 4.606% higher. Request-TPOT median is not lower, so do not describe every request statistic as improved. The steady interval is a secondary diagnostic, not a replacement for HTTP TPOT.

## Fixed protocol

- Nodes 065/167: 8 A100 80GB, matching ConnectX-6 Dx100Gb/s NICs. No new booking.
- Full GLM-4.7-AWQ/Humming INT4; BF16 activations and KV. Engine `870269702285d94bdd926a50cc29ad1a20bcf60c`.
- Eight requests, each P81920 cached context, Q1 new input, 128 output tokens. Total cached context655360. All native CPU HiCache ready, target GPU prefixes empty at timing start; H2D/admission timed. Synthetic CPU KV preparation, full CPU weight-bank reuse, graph capture and warmup excluded.
- EP8: Attention TP4xDP2, Expert EP8; PP2xEP4: Attention TP4, Expert EP4 perstage, 46/46 layers, two real B4 slots.
- Both arms full Decode CUDA Graph, local bucket [4], max4; Prefill graph disabled. Schedule overlap remains OFF.
- Same DeepEP native `9dc348878f7a3ad13fa8766a0d2738ca0d19ab116a3a5cc8e6e4e21b12988581`; NVLink withinpairs, RDMA outsidepairs, PCIe threshold0, no CE.

## Verification

- All8ranks: EP127 actual full-backend Decode replays each, PP254 each; local B4, correct capturekey4, no silent eager fallback.
- All8requests perarm completed128outputs, zero request retractions. CPU prefix-hit identities and final GPU prefixes verified; native HiCache queues empty aftercompletion.
- 33 CPU tests passed; no CUDA events or profiler in timed path. Proof hooks count CPU identities and actual replay submission; they do not claim an independent arithmetic validation of a new kernel. No new compute kernel was introduced.
- Owned processes/GPU allocations cleaned; original complete CPU weight banks retained. Guard postcheck passed, global registry history says released/cleanup_verified.

## Relation to preceding profile

The preceding **Graph-OFF CUDA-profile cohort** (32outputs, GPU-hot) gave the following per-GPU work budgets. PP rows are stage-device-time sums, **not PP walltime or TPOT**:

| Component (ms) | EP8 | PP stage0+stage1 |
|---|---:|---:|
| Attention |112.127|110.639|
| Expert/FFN |8.489|9.164|
| A+E |120.616|119.804|
| Communication/polling |69.990|57.404|
| No-kernel idle |16.068|29.041|

Thus A+E was already close. Communication saved12.586ms of device budget while idle grew12.973ms. Some differences are within uncertain kernel attribution; idle includes control/dependency waits and cannot all be labeled pipeline bubble. The new Graph experiment reuses model kernels and reduces CPU launch work; its verified paired TPOT shows a small PP advantage. No new post-Graph component profile was collected, so do not assign an exact fraction of the gain to an individual gap.

Historical unprofiled Graph-OFF128output cohort: EP188.160ms, PP192.063ms meanTPOT. Preserve that as a separate cohort, not as a simultaneous Graph toggle control.

## Sources

- [Complete comparison](runs/graph_r1/comparison.json)
- [Paired result and owned cleanup](runs/graph_r1/task/results.json)
- [EP8 raw metrics and graph receipts](runs/graph_r1/ep8/p81920_decode_results.json)
- [PP raw metrics and graph receipts](runs/graph_r1/pp2ep4/p81920_decode_results.json)
- [Global release proof](runs/graph_r1/lease_release.json)
- [Frozen launch spec](runs/graph_r1/dashboard_spec.json)
- [CPU tests](CPU_TESTS.json)

The optional47/45 layer-balance candidate was prepared separately, marked NOT_RUN, and was not launched because the matched46/46Graph cohort already reached the requested measured target.
