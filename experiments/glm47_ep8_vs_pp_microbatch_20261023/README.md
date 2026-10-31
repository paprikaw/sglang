# EP8 versus qualified optimized PP2 × EP4

Run EP8 only, reusing the qualified PP candidate without restarting it.

- Same engine `9e48f876a8eb2d7460317bc713bfa8680d78aea5`, full GLM-4.7-AWQ CPU banks, nodes 065/167 and DeepEP NVLink-pair + RDMA policy.
- Attention TP4 × DP2, Expert EP8, PP1; local target and draft native Graph B4. Native deterministic inference and loaded Humming batch-invariant/Stream-K-off receipts required.
- Exactly the qualified eight semantic input files: P4096, Q1, 64 output tokens. Model-generated semantic KV, no synthetic cache. Native cold HiCache has 4032 CPU tokens and zero GPU tokens per request; first forward recomputes 65 tokens per request.
- Two unprofiled cohorts in one process, each starting cold. TTFT starts at the one B8 HTTP submission, TPOT is first-to-last output divided by 63, E2E ends at each request's last token. HiCache H2D is timed; model loading, semantic Prefill and warmup are excluded.
- Harness-only CPU admission control routes logical requests 0–3 to DP0 and 4–7 to DP1 and releases each complete local ordered B4. It uses stock request DTOs and handlers, adds no collectives or device operations, and leaves unknown/sanity requests on the stock path. Stock colocated input blocking is disabled because it adds steady-loop CPU collectives.
- All eight ranks must save actual admission, first-forward B4, local schedule, native cache, target/draft Graph, deterministic and loaded-Humming receipts. Timed witness counters must remain zero. No CUPTI or EP8 witness capture cohorts are run.

PP source: `../glm47_pp_microbatch_witness_run_20261023/runs/witness_micro_r1`.
Its stronger bounded oracle validation is not automatically attributed to EP8.
Comparison requires exact 64-token IDs/text within EP8 repeats and against the
saved PP production outputs; mismatches retain metrics as diagnostic data and
disable qualification. No model-quality or general correctness claim.

Prepare only (no GPU launch):

```sh
python3 prepare_measure.py --name ep8_r1 --engine-commit 9e48f876a8eb2d7460317bc713bfa8680d78aea5
```

Parent task launches only through canonical Experiment Guard and owns cleanup.
