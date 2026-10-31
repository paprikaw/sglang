# GLM expert-only: EP8 versus PP2 x EP4

Status: **completed** on booked A100 node065, GPU0. Run `runs/r2` contains
36 measured configurations, independent numerical checks, passing Guard
scientific postcheck, verified idle-GPU cleanup and released lease.
Run `r1` failed before GPU runtime import because the remote launcher did not
forward authenticated lease variables; its original failed evidence is retained.

One A100 GPU emulates one destination rank at a time. Only real GLM-4.7-AWQ
layer-3 routed-expert W13, masked SiLU and W2 are timed. This is not an end-to-end
EP/PP benchmark. Attention, router, shared expert, dispatch/combine and PP stage
communication/scheduling are excluded.

| Geometry | EP8 | PP2 x EP4 |
|---|---|---|
| Logical batch | B | two slots of B/2 |
| Resident local experts | 20 | 40 |
| Physical TP source copies D | 4 | 4 |
| Local routed assignments per invocation | D*B | D*B |
| Mean rows per expert when all experts hit | D*B/20 | D*B/40 |
| GEMM FLOPs per GPU/layer/invocation | 6*5120*1536*D*B | same |

B = 8, 32, 128, 512, 2048, 4096. Deterministic top-8 routing is balanced;
no imbalance sweep. Both PP slots are recorded. At small B the number of
active experts need not double merely because resident expert count doubles.

Use the same Humming W4A16 GROUPED_MASKED backend and pinned source as the
validated historical expert probe. Fixed per-expert receive capacity is 1024
for both layouts to avoid changing buffer stride across the sweep. This
controlled buffer is not asserted to reproduce production communication
buffer allocations. The native tuning expected_m follows each layout's
physical source rows. Report tuning configurations with the measurements.

Warm both layouts before paired alternating measurements; save three blocks
of ten CUDA-event samples, eager and CUDA Graph separately. Compare common
experts against the validated independent AWQ dequantization reference.
Save raw samples, finite-output checks, weight tensor provenance, counts,
active experts, FLOPs, source/config hashes and GPU identity.

Timing is per local expert invocation. Do not add the two PP slot latencies
and compare that sum to one EP8 invocation: each stage owns only half the
model layers. Their ideal steady-state normalized compute costs are L*tEP
and (L/2)*(tPPslot0+tPPslot1). No inference about PP bubbles is made here.

The user authorized archive/checksum/cleanup of regenerable Python caches
and a bounded `expert_layer_only` Guard contract. No full-model bank was built
or changed. Guard source is isolated and published on
`codex/glm47-expert-layer-contract-20261021`, commit
`88645ebd8e6c35aa66f653049fa1129d8f6eab68`, immutable tag
`experiment/glm47-expert-layer-contract-20261021-r2`. This contract is not an
exemption for complete-model experiments, multi-node runs, or arbitrary models.

Results and reproducible analysis:

- `runs/r2/profile/result.json`: raw samples, real-weight hashes, numerical checks.
- `runs/r2/analysis/summary.md`: full table, interpretation and scope caveats.
- `runs/r2/analysis/comparison.csv`: 12 paired comparisons.
- `runs/r2/analysis/full_metrics.csv`: all36 configurations and count vectors.
- `runs/r2/analysis/kernel_geometry.csv`: source-derived static block coverage;
  not measured CTA/occupancy/bandwidth counters.
- `runs/r2/analysis/expert_compute.png` and `.pdf`: independent Eager/Graph curves.

Analysis uses Python3 with matplotlib; the GPU environment is not modified.
In this session plotting wheels were installed only under `/tmp`.
The worker/controller/selection files are frozen by the run spec and are not
rewritten for postprocessing. Final outputs are archived at:

`/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep8_pp2ep4_expert_compute_20261021_r2`

All14 snapshot artifact hashes and independent raw-sample/CSV recomputation
pass. The experiment-manager registry has been updated. After release the
existing background manager resumed its own workers; no model bank was changed.
