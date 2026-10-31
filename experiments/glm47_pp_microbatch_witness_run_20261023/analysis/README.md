# Offline first-divergence audit

This tool reads saved artifacts only. It does not import the engine, launch GPU
work, or change the frozen runtime or run directories. Outputs are exclusively
created outside those directories. A missing/invalid receipt stays explicit;
unreceipted tensor files are never used as a fallback.

```bash
PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=/home/bxb1/conda/lib \
/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python \
experiments/glm47_pp_microbatch_witness_run_20261023/analysis/analyze_witness.py \
  --baseline /absolute/path/to/baseline/run \
  --candidate /absolute/path/to/candidate/run \
  --prefix-tokens 4096 \
  --out /absolute/path/outside/runs/first-divergence.json
```

The JSON report includes all four tail ranks' stable, SHA-validated receipts,
every retained snapshot's independent CPU greedy-acceptance check, within-arm
repeat and cross-arm comparisons, rank 4 against TP ranks 5–7, actual output
token/text equality, eight-rank slot/microstep audits, and zero-witness proofs
for the separate timed/repeat cohorts. Matching uses observed request identities
and incoming histories, not timestamps or ordinal positions. Two payloads at
most are cached; full GPU tensor snapshots remain on disk and are never copied
into JSON. CPU oracle evaluations are reused within each comparison.

Missing later snapshots, changed tree/mask metadata, actual GPU KV values not
captured, and the absence of an independent non-speculative target reference are
explicit limitations. The tool never labels a performance comparison validated.
Its exit status indicates report generation, not that model correctness passed.

CPU tests:

```bash
PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=/home/bxb1/conda/lib \
/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python \
-m unittest discover -s experiments/glm47_pp_microbatch_witness_run_20261023/analysis -v
```
