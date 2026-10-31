# PP HiCache simultaneous restore diagnosis

2026-10-30. E4 production source remains frozen at `f8237955ef141d6e8c9e78c7d7ad93854671e81b`.

`171_pp2_host_r7` executed 12 restore cases, 24 outputs. All 12 warm sequences and all six single-side cases matched the serial reference. Four restored outputs mismatched, all in both-side cases: both_first_r1 at 4096/4097; both_last_r1 at 4097; both_first_r2 at 4097. This is not accepted as numerical tolerance. The first failure repeats token 369 after a short correct prefix.

Both Decode stages each report nine real 4096-token host restores with `copy_mamba_to_request=False`. Both Prefill stages report eighteen loads. Thus the first-stage-only HTTP state list is insufficient evidence; per-stage files were checked. Cleanup passed and Guard released the node. No PP performance result is admitted.

[Raw cases](runs/171_pp2_host_r7/evidence/pd/host_restore_isolation.json) · [Recomputed per-case and stage evidence](runs/171_pp2_host_r7/analysis/restore_pattern.json)

The next diagnostic `171_pp2_prefill_fence_r9` uses the same E4 engine and 12-case reproducer, adding a CPU wait on the local Prefill H2D finish event immediately after `ready_to_load_host_cache`. It requires nonzero fence receipts from both Prefill stages and actual restore receipts from both Decode stages. It is a diagnostic, **not a production fix or performance sample**. A pass would motivate examining Prefill load/forward ordering, but would not uniquely identify the race. A failure would redirect investigation toward cached state correctness or P→D publication. The frozen prepared r8 was never launched and is superseded before GPU use because its runtime-role check did not normalize the role string.

Current single-GPU eight-card baseline is independent of the PP-only failure and is running with native HiCache. It has passed all eight ranks' cold/cache warmup through 208320 tokens. Do not cancel that valid run to overwrite its protocol.

## 2026-10-31 Prefill fence diagnostic

`171_pp2_prefill_fence_r9` finished and cleaned. Both Prefill stages performed 18 actual completion fences; both Decode stages performed nine real restores. It still failed the exact same four 4096/4097 output cases as r7. Therefore this Prefill completion fence does not fix the failure. Matched PP2 and EP2 state-fingerprint diagnostics are being prepared to compare source state before final send with destination state after transfer commit. These invasive GPU fences/copies are not performance samples.

## Source-localized mapping failure and E5 candidate

Matched diagnostics: `011_ep2_state_probe_r1` passed all 24 restored outputs; `171_pp2_state_probe_r10` repeated the same four failures. All 50 paired requests in the PP probe had bit-identical GDN state at P final-send and D commit, so a missing GDN payload in that transfer does not explain these samples. KV hashes can differ legitimately when one side reuses previously computed prefixes.

`_MambaStrategy.build` passed global layer keys into a controller iterating stage-local IDs. On the second PP2 stage, keys 20..39 are queried with 0..19: no H2D layer matches, although completion events are recorded. D2H writes all layers without that mapper. The resulting reused device contents explain why a completion event alone was insufficient proof of restore.

E5 `37e0828c642ce4f3795c4c4394f4ce913bcd1a35` subtracts the PP stage start from transfer keys while retaining global forward/RDMA mappings. CPU regression reproduced an empty restore set on stage 20..39 and incorrect partial mapping at a nonuniform boundary; after repair all 27 related checks pass. `171_pp2_state_probe_r11` and `011_pp2_host_r12` are testing the repair with and without intrusive fingerprints. GPU confirmation remains pending.

[Source-function reproduction](pp_layer_mapping_repro.json) · [Failing regression](unit_r5_red.log) · [Passing regression](unit_r5_green.log)

## E5 GPU confirmation

Both `171_pp2_state_probe_r11` (intrusive snapshots) and `011_pp2_host_r12` (ordinary scheduler, no diagnostic H2D completion fence) completed all 12 cases / 24 restored requests. Both passed every output comparison, Guard postcheck and verified cleanup: 48/48 combined. In r11 all 50 paired P/D GDN states remain bit-identical. This qualifies this repaired 35B PP2 path for the subsequent matched performance tests; it is not a universal claim about other models or arbitrary parallel layouts. Earlier E4 PP host-restore performance must remain excluded.
