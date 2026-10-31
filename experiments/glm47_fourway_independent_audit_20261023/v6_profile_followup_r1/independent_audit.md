# Independent four-way Decode audit

| Configuration | Cohort | Validity | Output tok/s | TTFT mean/median/P99 ms | TPOT mean/median/P99 ms | E2E mean/median/P99 ms |
|---|---|---|---:|---:|---:|---:|
| ep8_mtp_on | timed | performance_valid_profile_pending_or_failed | 97.340 | 556.754/556.759/556.809 | 68.963/69.647/74.651 | 4901.452/4944.525/5259.805 |
| ep8_mtp_on | repeat | performance_valid_profile_pending_or_failed | 95.664 | 549.500/549.504/549.601 | 70.478/71.233/76.228 | 4989.625/5037.156/5351.935 |
| ep8_mtp_off | timed | performance_valid_profile_pending_or_failed | 84.802 | 538.848/538.851/538.900 | 87.279/87.279/87.281 | 6037.435/6037.437/6037.525 |
| ep8_mtp_off | repeat | performance_valid_profile_pending_or_failed | 85.076 | 527.740/527.743/527.789 | 87.148/87.148/87.149 | 6018.033/6018.039/6018.123 |
| pp2ep4_mtp_on | timed | performance_valid_profile_pending_or_failed | 88.341 | 501.410/501.406/604.841 | 77.132/78.295/82.394 | 5360.715/5389.328/5795.621 |
| pp2ep4_mtp_on | repeat | performance_valid_profile_pending_or_failed | 88.228 | 501.688/501.686/605.311 | 77.233/78.400/82.499 | 5367.336/5396.337/5802.697 |
| pp2ep4_mtp_off | timed | performance_valid_profile_pending_or_failed | 83.663 | 494.942/494.938/593.933 | 88.920/88.920/90.131 | 6096.901/6096.925/6119.721 |
| pp2ep4_mtp_off | repeat | performance_valid_profile_pending_or_failed | 83.391 | 497.564/497.562/596.168 | 89.194/89.194/90.396 | 6116.771/6116.770/6139.669 |

## Separate profile: absolute ms per GPU / local B4 target cycle

| Configuration | Rank | Target layers | A | Routed E | Shared E | Dense | Comm | Other GPU | Unknown | Overlap | Idle | Total cycle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ep8_mtp_on | Missing/invalid; not estimated | — | — | — | — | — | — | — | — | — | — | — |
| ep8_mtp_off | Missing/invalid; not estimated | — | — | — | — | — | — | — | — | — | — | — |
| pp2ep4_mtp_on | Missing/invalid; not estimated | — | — | — | — | — | — | — | — | — | — | — |
| pp2ep4_mtp_off | Missing/invalid; not estimated | — | — | — | — | — | — | — | — | — | — | — |

## Boundaries and caveats

- Production: B8 P4096 raw/4032 native CPU cached, Q1->64. Source import excluded, native HiCache H2D included.
- TPOT=(last received-first received)/63, not verify time or profile cycle.
- Profile ms are exclusive per-device wall partition per local B4 target cycle, including intervening draft/transport/idle.
- EP8 rank executes92 target layers; PP rank executes46. PP cycles alternate two B4 request slots and are not request TPOT.
- MTP Verify advances a measured variable output count; per-committed-local-output ms divides all B4 outputs, not one request.
- Unknown GPU activity and overlap remain explicit; idle is not proof of a pipeline bubble.
- P99 over only eight requests equals the sample maximum.

## Failures and cleanup

- ep8_mtp_on: production cleanup=True; profile=followup_failed_before_complete_profile_not_estimated; errors=[]; original profile failure={"type": "RuntimeError", "message": "Model worker exited during idle command profile_cold", "profile_only": true, "performance_preserved": true, "performance_repeated_and_valid": true, "source_bank_import_timed": false, "profiling_failure_not_a_performance_result": true}; follow-up failure={"type": "RuntimeError", "message": "No actual local model forwards"}
- ep8_mtp_off: production cleanup=True; profile=followup_missing_or_cleanup_pending_not_estimated; errors=[]; original profile failure={"type": "RuntimeError", "message": "Model worker exited during idle command profile_cold", "profile_only": true, "performance_preserved": true, "performance_repeated_and_valid": true, "source_bank_import_timed": false, "profiling_failure_not_a_performance_result": true}; follow-up failure=null
- pp2ep4_mtp_on: production cleanup=True; profile=followup_missing_or_cleanup_pending_not_estimated; errors=[]; original profile failure={"type": "RuntimeError", "message": "Model worker exited during idle command profile_cold", "profile_only": true, "performance_preserved": true, "performance_repeated_and_valid": true, "source_bank_import_timed": false, "profiling_failure_not_a_performance_result": true}; follow-up failure=null
- pp2ep4_mtp_off: production cleanup=True; profile=followup_missing_or_cleanup_pending_not_estimated; errors=[]; original profile failure={"type": "RuntimeError", "message": "Model worker exited during idle command profile_cold", "profile_only": true, "performance_preserved": true, "performance_repeated_and_valid": true, "source_bank_import_timed": false, "profiling_failure_not_a_performance_result": true}; follow-up failure=null
