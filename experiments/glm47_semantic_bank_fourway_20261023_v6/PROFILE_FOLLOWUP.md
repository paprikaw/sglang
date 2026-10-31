# Profile-only follow-up

This directory is an unfrozen candidate until `prepare_profile.py` succeeds.
It does not launch a GPU and does not replace the original production results.

## Protocol

1. Require all four v5 arms to contain two valid unprofiled production cohorts,
   exact within-arm output-ID/text repeatability, native Graph and scheduler
   proofs, and verified cleanup of both model nodes.
2. Bind the original configuration, original frozen source/model/native inputs,
   both production files, repeatability, cleanup and per-arm calibration traces
   by SHA-256. Require exactly the same engine commit, model identity, semantic
   Bank identity, topology, native serving arguments and kernel environment.
   Only listener/rendezvous ports and new run/scratch/runtime locations differ.
3. Reuse each arm's existing calibration **labels and exact launch geometry**
   offline. Copy its trace bytes without changing filenames; import no timing
   values. Require both warm0/warm1 on ranks 0/4 for every target/draft role,
   actual scoped GPU activity, and PP boundary witnesses for PP arms.
4. Start a fresh consumer with no startup profiler or startup calibration.
   Do not rerun either production64 cohort. Import the immutable semantic Bank
   and run one short B8/Q1→8 warm cohort without CUPTI, outside all timing.
   Require real per-request MTP mode, native Graph/scheduler and Humming proofs.
   Reset/reimport the same Bank again, proving CPU4032/GPU0 residency before
   the separate formal profile. This avoids first-request wrapper/JIT drift.
5. At confirmed scheduler idle start the process's first and only CPU+CUDA
   profiler. A priming fill and three explicitly scoped long marker proofs
   precede the client release. Keep that same collector running across the
   bounded formal cohort and stop only after complete scheduler/HiCache idle.
6. Resolve marker names from actual CPU scopes, runtime correlations and GPU
   events **after export**. Remove only those individually attested proof GPU
   events from the formal 2-markers-per-segment audit. Never infer names, drop
   an arbitrary prefix, accept missing markers, or import another rank's GPU
   timing. Preserve unknown classifications.

The earlier lifecycle-loss cause is not proven. This candidate removes the
startup collector lifecycles rather than claiming an environment flag fixed
them. A new arm passes only if its actual all-rank GPU activity audit succeeds
and cleanup passes. Missing profile metrics remain missing. New performance
point count is zero; four original performance points and four new profiles
remain separate provenance-linked records.

## Preparation (only after all source arms are complete)

Use `prepare_profile.py --name NAME --profile-source-run ABS_V5_RUN
--reuse-semantic-bank ABS_READY_BANK`, under the pinned CPU Python environment.
Preparation rejects incomplete source arms or incomplete calibration pairs.
It freezes all evidence and outputs a Guard specification; it does not launch.
