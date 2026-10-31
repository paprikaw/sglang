# Four-way saved-result report

CPU-only standard-library reporter. It reads completed v5 performance and optionally
strictly validated v7 profile follow-up results. It does not import the engine,
launch GPU work, modify frozen runs, or push Git changes.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 generate_report.py \
  --profile-run ../glm47_semantic_bank_fourway_20261023_v7/runs/profile_followup_r2
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_generate_report.py' -v
```

The default output directory `current/` contains `summary.json`, `performance.csv`
and `report.md`. The performance validator and profile join validator are imported
read-only from v7/analysis; no new trace attribution rules are introduced.

Headlines average the two unprofiled cohort estimates. Per-cohort means, medians
and P99 remain separate. Half-range is descriptive, not a confidence interval.
Profile data stay null until existing all-rank qualification and cleanup gates
pass. Local B4 stage-cycle activity is never relabeled request TPOT.
