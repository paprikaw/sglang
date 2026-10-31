# Matched Decode CUDA Graph qualification

Hypothesis: repeated CPU dispatch and launch gaps cost PP proportionally more than EP8. Full Decode CUDA Graph may reduce those gaps; test both layouts with identical Graph settings so the comparison remains fair.

- Nodes: existing booked spartan-gpgpu065 and spartan-gpgpu167, 8 A100 80GB, same ConnectX-6 Dx NICs. Canonical Experiment Guard acquires the whole pair and verifies full cached weights.
- Model/compute: full 92-layer GLM-4.7-AWQ, Humming INT4, BF16 activations and KV. Fixed SGLang 870269702285d94bdd926a50cc29ad1a20bcf60c.
- EP8: Attention TP4 x DP2, Expert EP8; PP2 x EP4: Attention TP4, 46/46 layers, two B4 slots. Graph local bucket B4 in both layouts. Overlap remains disabled.
- Fixed workload: 8 requests, each P81920 CPU-cached tokens, Q1 new input, 128 output tokens. Each timed cohort starts native HiCache ready and target GPU prefixes empty. HiCache H2D is timed; synthetic native CPU-KV creation, full weight load, graph capture, and warmup are not timed.
- Transport unchanged: native 9dc348878f7a3ad13fa8766a0d2738ca0d19ab116a3a5cc8e6e4e21b12988581, NVLink within pairs and RDMA outside pairs, threshold 0. No CE path.
- Only performance configuration change: enable full Decode CUDA Graph with bs=[4], max_bs=4 for both arms; Prefill graph disabled. No new compute kernel or arithmetic change.
- Evidence: CPU-only wrappers count actual full-backend replay and ModelRunner output flags for every real Decode. No profiler, CUDA events, or tensor readbacks in timed telemetry. Existing accepted Graph backend is reused, not a new numerical implementation.
- Acceptance: all ranks use actual graph replay during timed steady Decode; native CPU hit metadata, all 128 outputs, B4 per GPU, two PP slots, natural capacity and successful owned cleanup must pass. Report SSE TPOT, TTFT, E2E and throughput; never substitute component device-time sums for request TPOT.
- This is a new Graph-ON cohort. Do not merge it with the historical Graph-OFF cohort or claim a win before the paired result and Guard postcheck.
