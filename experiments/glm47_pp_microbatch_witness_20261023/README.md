# PP2/EAGLE first-divergence witness — SOURCE ONLY / NOT RUN

This directory contains instrumentation and CPU tests only. It does not copy a
previous run tree, modify an engine/frozen harness, prepare a Guard spec or launch
GPUs. A future diagnostic must integrate it into a new frozen source root.

## Placement

Call `witness.install(output_directory)` from the scheduler startup hook before
constructing Scheduler, after other instrumentation is installed. It wraps the
actual `eagle_worker_common.eagle_sample` binding called by `run_eagle_verify`.
Only PP2 last-stage ranks record. On this CUDA engine the greedy branch does not
broadcast acceptance (that broadcast is HIP-only), so all four tail TP ranks
must be retained and compared, not just TP0.

The witness admits only the two `timed-` / `repeat-` cohorts, unpenalized greedy
sampling, no grammar mask, EAGLE topk1 and three-node verification. Bounds are 16
verify calls per `(cohort, PP slot)` and 768 MiB retained CUDA tensor bytes/rank.
For B4 × draft3 × vocab151k FP32, one cohort's two slots use about 232 MB/rank.
GPU snapshots are released on each idle flush between cohorts.

## What is captured

Immediately before native `eagle_sample`: cloned target logits; seq_lens and
seq_lens_cpu, input_ids, request pool indices and out_cache_loc; draft_token,
positions, custom_mask, retrieve_index/next_token/next_sibling. CPU request data
include RID, normalized request index, original-prompt digest, emitted output IDs
and committed-history digest. Immediately after native sampling: cloned predict,
accept_lens and accept_index. Snapshots occur on the producer stream before reuse
of a CUDA Graph output buffer. No `.cpu()`, `.item()`, `.tolist()`, CUDA sync or
file I/O is inserted into the model-forward wrapper.

Only `Scheduler.is_fully_idle()` triggers synchronization, D2H and mode-0600
`rank4..7_flushNNN.pt` binary writes. The future diagnostic controller must wait
for all four tail-rank flush artifacts before stopping model processes. These
instrumented runs are not performance measurements. Nothing dumps logits/tensors
to stdout. The test's tiny tensors are CPU data, not substitute model evidence.

## Offline comparison

`chain_reference(snapshot)` independently reconstructs CPU greedy chain
acceptance from captured logits/candidates and compares native accept_lens,
accept_index-selected token values. `compare_snapshot` distinguishes different
histories, different logical metadata, logit drift, and inconsistent acceptance.
`compare_requests(left_rows, right_rows)` matches by request index plus committed
history digest, not verify-round number. A sibling request's divergence cannot
hide still-matching requests. Missing common histories are reported, not forced
or interpolated. Physical allocation indices may differ without semantic change;
whole-batch custom-mask and sibling-history equality remain explicit evidence.

A changed argmax with small logit drift is not automatically harmless. Reports
set `benign_numerical_branching_proven=False`: actual GPU KV values and an
independent target-model oracle are not captured. If metadata already differs,
fix relay/state first. If acceptance disagrees with recorded logits, inspect the
verify/accept kernel or buffer ownership. If same observed metadata produces
changed logits, the next selective test must compare actual logical-prefix KV
values or teacher-forced same-history target verification before asserting a
numerical-only explanation.

## Source audit

Engine revision remains `f415858311c0d7646b368b30b64b371dca3427e4`.
`run_eagle_verify` obtains target logits before `eagle_sample`; sampling creates
new prediction/accept buffers. `_pp_launch_batch` snapshots proxy/output tensor
payloads on the forward producer stream before recording its ready event. Tail
next-tree parent/score buffers are cloned in the worker, and scheduler snapshots
next-chain payload before another batch can reuse draft Graph buffers. This is
source-level ordering evidence, not proof that live KV content or logits match.
