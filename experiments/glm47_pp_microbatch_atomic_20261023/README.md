# Atomic admission PP2 MTP correctness diagnostic

Same engine f415858311c0d7646b368b30b64b371dca3427e4; only SGLANG_PP_SPEC_MICROBATCH changes between arms.
Two existing A100 nodes 065/167, full canonical CPU weight bank reuse-only, no synthetic KV.
B8 P4096 Q1 → 64, real semantic Prefill builds native process-local HiCache. Cached P4032, first forward 65 tokens.

Every preparation, warmup, unprofiled production, unprofiled repeat and separate profile submits one HTTP nested input_ids batch with an ordered list of eight RIDs. Stock pre-tokenized no-DP-attention handling automatically dispatches BatchTokenizedGenerateReqInput; no tokenizer-batch-encode flag is enabled. All ranks record the actual atomic scheduler admission and exact FCFS groups [0..3]/[4..7].

TTFT starts at that one HTTP submission and ends at each request's first received token; TPOT spans its first to last token; end-to-end ends at its last received token, not another request's stream close. Batch throughput includes the whole response close.

Both production cohorts start with native CPU HiCache ready, GPU empty, and profiler off. Repeat outputs are retained and compared exactly. Pair speedup is not validated unless same-process repeats AND cross-arm greedy outputs match.

Only after both unprofiled cohorts are saved, the scheduler idle cache command starts CUPTI and runs a one-element non-marker bootstrap fill with synchronization. This occurs before profile clients are released and outside any model forward. No inner-forward phase fences are added. A separate bounded 12-Verify profile preserves all-rank receipts and cleanup.
