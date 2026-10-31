# Deferred diagnostic bootstrap

This unfrozen fallback keeps v4's model, semantic inputs, Graph settings,
source validation, two unprofiled cohorts, and cache residency unchanged.

Strict CUDA marker bootstrap runs only for `profile_cold`, after both
production cohorts and exact within-arm repeatability have passed and been
saved. No profiling or new GPU forward is added to production timing.

`performance_valid` and `profile_valid` are independent. After valid repeated
production, a profile-only exception is saved separately and production data
remain available. Continuing to the next consumer requires verified owned/GPU
cleanup, valid production, and no operational failure. Production, source,
repeatability, or cleanup failure still stops the sequence.

A valid profile requires both the existing all-rank receipts and the existing
CPU-only saved-trace auditor's strict actual GPU marker/stream/window checks.
Unknown kernel classifications remain unknown. Because this auditor is now a
runtime result gate, `analysis/stage_activity.py` is included in frozen inputs.

Overall success still requires four valid production arms and four valid
profiles; `expected`, `expected_performance`, and `expected_profiles` remain
four. `valid_points` counts fully complete arms, while `performance_points`
and `profile_points` expose partial completion. A profile failure does not
become a passing arm or passing overall task.

Preparation/freezing is a separate CPU-only step; this fallback never launches
GPU workers by itself.

The profile-only bootstrap candidate now waits 0.25 seconds after collector
entry before priming, then observes three 100,000,000-cycle proof sleeps.
Receipts record both exact settings; this does not establish the collector
failure's root cause. CPU marker scope selection excludes same-name
`gpu_user_annotation` mirrors while still requiring actual runtime-to-GPU
correlations, the correct device, and two independent positive witnesses.
