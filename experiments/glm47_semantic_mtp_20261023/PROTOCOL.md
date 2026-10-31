# Semantic MTP Decode comparison

## Invalidated precursor

`../glm47_speculative_compare_20261023/runs/spec_r3` used synthetic KV and
random token IDs. Its internal mechanical checks do not make it a valid
semantic speculative-decoding experiment. The run was cooperatively stopped,
both nodes passed owned-process/GPU cleanup, and Guard released the lease.
The original artifacts and the explicit scientific disposition are retained.

## Inputs and cache preparation

- Model: complete GLM-4.7-AWQ checkpoint, native one-layer MTP via EAGLE 2/1/3.
- Source: isolated SGLang `a0940c4c011804cab0334c9ad37aa8f02445688a`.
- Nodes: booked A100 065/167, same ConnectX-6 Dx inventory classification.
- EP8: Attention TP4 x DP2; experts EP8 across both nodes.
- PP2 x EP4: Attention TP4 and EP4 within each stage; 46+46 target layers,
  two real B4 slots, tail-stage-only MTP draft.
- Eight distinct real source-code/documentation contexts, identically hashed
  for both configurations; no shared long prefix, random IDs, or injected KV.
- Historical input size retained: raw prefix P=81,920, Q=1, output=128.
  Native EAGLE bigram/page boundaries give 81,856 cached tokens and 65 tokens
  recomputed on the initial forward. Report both values explicitly.
- Short semantic prompts first check actual verification and positive draft
  acceptance. This is a functional sanity check, not numerical parity proof.
- The serving model performs actual Prefill of every long input. Native
  write-through copies target and MTP KV to CPU HiCache. Wait for all queues
  and acknowledgements, evict GPU only, and verify every rank's CPU prefix hit.

## Measurement

Actual Prefill preparation, weight loading and one full Decode warmup are
outside timing. The final eight-request measurement starts CPU-ready and
GPU-empty. TTFT therefore includes native HiCache H2D, the mandatory boundary
recomputation, scheduling and first output. TPOT is first-to-last streamed
output time divided by 127, including the effect of multi-token speculative
bursts. End-to-end and global output throughput use the complete cohort.
Per-request acceptance counts and real graph-replay/placement receipts are
retained. N=8 P99 is a nearest-rank sample maximum, not a stable tail estimate.

This is same-process native HiCache preparation, not a CPU checkpoint replay.
There was no compatible registry entry for this engine and semantic prompts.
No previous synthetic result is used as a matched non-speculative baseline.

## Separate component profile

`../glm47_semantic_mtp_profile_20261023` uses the same real input digests and
actual model Prefill, but profiles a bounded steady MTP window. Its instrumented
performance numbers are not substituted for the unprofiled performance run.
Classify exact CUDA kernel names using repeated untimed CPU-scope calibration;
retain unknown, overlapping, and idle intervals. Report target verification,
draft, and draft-extend separately. Local per-rank API elapsed partitions are
not a sum across ranks, not pure wire time, and not directly equal to TPOT.
