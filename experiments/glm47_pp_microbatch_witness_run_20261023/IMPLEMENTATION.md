# PP2 speculative Decode microbatch implementation

## Scope

The opt-in `SGLANG_PP_SPEC_MICROBATCH=1` path supports PP2, two slots,
`pp_async_batch_depth=0`, EAGLE top-k 1, disabled overlap scheduler, and no
DP Attention, elastic PP, or disaggregated serving. Each slot still waits for
its own accepted tokens and next draft before that slot is reused. B8 is split
into two independently tracked B4 request groups; this does not split one
request's dependent Decode steps into independent work.

## Scheduling change

The ready slot submits its computation and activation transfer before retiring
the other slot's feedback. Activations and feedback use distinct PP process
groups, so an activation receive cannot queue ahead of the feedback needed to
release the other slot. Graph-owned activation, feedback, and verification
location tensors are copied on their producer stream and retained until the
corresponding communication work completes. Existing request IDs, logical
microsteps, acceptance updates, and KV ownership remain in the original
protocol.

The implementation is commit `f415858311c0d7646b368b30b64b371dca3427e4`.
The follow-up commit `9e48f876a8eb2d7460317bc713bfa8680d78aea5` forwards native
deterministic mode to Humming dense and MoE compute configuration and both
explicit expert GEMM tuning calls. That option controls compute tuning, not
weight representation; existing canonical CPU weight banks remain reusable.

## Qualification protocol

The new baseline and candidate use the same Humming invariant configuration,
semantic corpus, native HiCache protocol, model, graph shapes, and transport.
Timing cohorts must have zero diagnostic snapshots. Separate untimed witness
cohorts capture bounded pre-argmax logits and post-acceptance state on all four
tail TP ranks; CPU readback and writing happen only at full scheduler idle.
An independent CPU tree traversal checks accepted token values, lengths, and
indices. Offline comparisons align request IDs and committed histories, not
wall clocks or arbitrary Verify ordinals.

Earlier default-mode measurements showed shorter PP waits, but both the
original and candidate failed exact output repeatability. Those measurements
remain diagnostic, not a correctness-qualified speedup. Native deterministic
mode alone also failed repeatability before the Humming forwarding fix. The
current paired runs must provide their own qualification evidence; this
document is not a completion receipt or a claim that the issue is resolved.
