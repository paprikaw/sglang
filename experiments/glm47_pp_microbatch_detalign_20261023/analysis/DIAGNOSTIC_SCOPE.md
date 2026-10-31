# Deterministic baseline diagnostic only

This directory contains an offline analysis of `runs/align_base_r1` only. There is no completed matched candidate in this cohort, so these artifacts do **not** establish a microbatch performance improvement.

The baseline's repeated greedy generation is not stable: `runs/align_base_r1/pp2ep4/p4096_decode_repeatability.json` reports `passed: false`, with 4 of 8 requests having different output token IDs and text between the original and repeat cohort. The next diagnostic must locate this disagreement before treating scheduling speed measurements as a validated implementation result.

All 8 ranks have complete GPU marker coverage in the saved baseline profile. Its local Verify component measurements and PP-kernel waiting evidence remain inspectable as baseline diagnostics, not a correctness proof or paired comparison.

Native deterministic execution with alignment 1024 changes computation costs substantially. Do not merge these timings with the non-deterministic `glm47_pp_microbatch_atomic_20261023` cohort or use that cohort as its comparison arm. The failed earlier `det_base_r1` with alignment 4096 and chunk 1024 produced no valid timed cohort and is excluded.

The report preserves local clock boundaries. PP kernel time includes transfer and polling/waiting, and cross-host pipeline overlap is not inferred from unmatched clocks. No frozen runtime, run specification, or raw trace was changed by this analysis.
