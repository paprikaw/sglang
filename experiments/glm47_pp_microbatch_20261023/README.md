# PP2 × EP4 semantic microbatch A/B

No model data, inputs or runs have been copied from the frozen historical experiment.
This is a new source-only harness. Baseline and candidate use the **same pinned
new engine revision**, different servers and Guard runs, toggling only
`SGLANG_PP_SPEC_MICROBATCH=0/1`. Both use 065/167, Attention TP4 per stage,
EP4, native EAGLE 2/1/3, B8 split into two B4 slots, full Decode CUDA Graph,
Q1 and 64 generated tokens. Start with P4096; P16384 is an optional second point.

Inputs are distinct semantic source/documentation excerpts with a meaningful
question and assistant suffix. Actual model Prefill constructs target and MTP KV.
There is no synthetic KV, manual KV injection, source-pool copy, or cross-process
checkpoint replay. Existing complete canonical CPU weight banks are reused.

After startup and semantic sanity, prepare real prefixes once per process,
warm, evict GPU prefixes while retaining HiCache CPU prefixes, then measure
production Decode. TTFT includes native HiCache H2D. CUDA profiling is disabled
during production; a subsequent separate profile reuses the same semantic KV
and captures 12 Verify calls per rank.

## Prepare (CPU only)

Do not prepare/freeze until engine changes are committed and clean. Substitute
the exact final 40-character candidate revision below. Build each P input once,
then reuse identical input files for both modes. No GPU is launched by these commands.

```bash
PYTHON=/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python
ROOT=/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_microbatch_20261023
ENGINE=/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/glm47-pp-spec-microbatch-20261023
REV=<exact-final-commit>
$PYTHON "$ROOT/build_semantic_inputs.py" --prefix-tokens 4096 --engine-root "$ENGINE" --engine-commit "$REV"
$PYTHON "$ROOT/prepare_measure.py" --name baseline_p4096_r1 --mode baseline --prefix-tokens 4096 --engine-root "$ENGINE" --engine-commit "$REV"
$PYTHON "$ROOT/prepare_measure.py" --name candidate_p4096_r1 --mode candidate --prefix-tokens 4096 --engine-root "$ENGINE" --engine-commit "$REV"
```

Run each generated exact controller command only through canonical
`dashboard_guard.py run --spec ... --resource-choice auto --exit-after-run -- ...`.
The harness retains required reuse-only canonical bank admission, frozen
sources/inputs, exact per-role loader logs, native topology proofs, leases and
owned cleanup. It never books nodes or evicts weight banks.

Actual receipts require the selected scheduler mode, two slots, B4 bound, all
eight requests and candidate feedback-group identity on every rank. Profiler
output and production output have distinct request IDs. A profile alone cannot
count as a valid production measurement.

## Read results

Run `analyze.py --run <run>` separately on each completed run. It preserves
unknown GPU time, idle time and overlap; communication kernels include waiting.
Same-request matched stage sums are not TPOT.

Run `finalize_pair.py --baseline <run> --candidate <run> --out <new-output-dir>`
only after both runs and cleanup pass. It rejects source, workload, prompt or
scheduler-proof mismatches. A short-context A/B is not directly comparable to
the historical P81920 experiment.

CPU checks: `python3 -m unittest discover -s "$ROOT" -p 'test_*.py'`.

The pair report requires exact greedy generated-token and text parity. On any
mismatch it retains all performance metrics and both outputs but marks
`correctness_unconfirmed`, disallows a validated speedup claim, and exits 2.
Actual per-rank slot/RID compositions are retained to diagnose any grouping
changes affecting quantized execution.
