---
name: llm-experiment-guard
description: Mandatory preflight guard for running LLM experiments with vLLM, SGLang, or another serving engine in this Spartan workspace. Use before any benchmark, model serving, GPU validation, profiling, autoscaling, or multi-node experiment command; it selects the engine-specific preflight, pins source commits and tags, checks reproducibility controls, enforces exclusive one-model-experiment-per-node placement, parallelizes only across disjoint booked node sets, and never requests new nodes.
---

# LLM Experiment Guard

Apply this skill before running any LLM engine experiment command. Use the
common placement/provenance rules for every engine, then execute the matching
vLLM or SGLang preflight below.

## Hard Rules

1. Do not run an experiment until the selected engine's environment, imports,
   and required native artifact checks pass.
2. Do not run an experiment until the experiment config passes the fixed baseline check.
3. Do not request, allocate, or submit new compute nodes. Never use `salloc`, `sbatch`, or an `srun` pattern that creates a new allocation.
4. Use only nodes already booked by the user and visible from `squeue --me`.
5. When execution must happen on a booked remote node, run the command with `ssh -o ConnectTimeout=5 <node> "<command>"`.
6. Run one SSH probe at a time. Avoid parallel SSH fan-out on busy clusters.
7. If no suitable already-booked node is available, stop and report that the experiment cannot be run yet.
8. Before launching any KV-cache experiment, resolve whether the user wants native VMM, default KVCacheD, or KVCacheD layer stacking. If the wording is ambiguous, ask before running.
9. Keep pipeline concurrency at the implementation default unless the user
   explicitly makes the concurrency gap an experiment variable. Record every
   nonzero gap in the config, hypothesis, result, and experiment manifest.
10. Treat every node used by a model experiment as exclusive for that
    experiment. Never run two model-serving, benchmark, profiling, or
    reconfiguration experiments on the same node at the same time, even when
    they would use different GPUs.
11. Minimize total experiment time by running independent experiments
    concurrently on disjoint booked node sets. A multi-node experiment claims
    every node in its placement until all model processes and workers exit.
12. Pin the exact source commit before launching. For a reusable or validated
    implementation, create an immutable annotated source tag after tests pass;
    never use a floating branch name as the only provenance. Use the source-tag
    rules and helper in `spartan-model-serving` for model-serving runs.

## Source Tagging

Record worktree, branch, commit, dirty state, and existing tag in every preflight.
Commit compatibility/development changes before tagging. Use an annotated tag
such as `serving/vllm/<model-slug>/<purpose>-<YYYYMMDD>` and never force-move it.
Keep the Git source tag distinct from the deployment ID: one source commit may
support many placements and launch configurations. For vLLM/SGLang serving,
also apply `/home/bxb1/.codex/skills/spartan-model-serving/SKILL.md`.

## Step 1: Identify Engine And Checkout Type

Run:

```bash
git rev-parse --show-toplevel --git-common-dir --git-dir
git worktree list --porcelain
```

Classify the current checkout:

- Main checkout: the current top-level path is the primary repository worktree.
- Git worktree: `.git` is a file or `git rev-parse --git-dir` points under another checkout's `.git/worktrees/...`.

Classify the engine from its source tree and launch command. Record `engine` as
`vllm`, `sglang`, or an explicit third engine in the manifest. Do not apply a
vLLM build baseline to SGLang or infer the engine from the parent directory
name alone.

In this workspace, the known main vLLM checkout is:

```text
/data/gpfs/projects/punim2715/vllm_workbench/vllm
```

### Known-Good Qwen/Llama E2E Reconfiguration Tag

Use this tag when the user asks for the old implementation that was validated
by the Qwen and Llama end-to-end PP reconfiguration experiments, or when they
ask to avoid newer `xu`/BurstGPT profiling changes because the current
implementation may not be trusted:

```text
tag: legacy/qwen-llama-e2e-reconfig-good-20260704
commit: b6911b94d987c57787b6ccfdd66e341829104ade
branch/tag at tagging time: origin/kv_migration_v2, codex/burstgpt-pattern-switch
subject: update
```

This tag was previously named `legacy/qwen-e2e-reconfig-good-20260704`; use the
Qwen/Llama name above for new commands and reports.

For this tag, keep any launcher-only fixes such as using
`python -m vllm.entrypoints.cli.main serve` as uncommitted worktree fixes when
needed for the shared virtualenv, but do not treat those launcher fixes as a
different validated reconfiguration implementation.

This tag includes the Llama FlexiAttention path.  Use the tag-specific Llama
PP2 guard baseline for BurstGPT/Llama reruns instead of the global Qwen
baseline:

```bash
.venv/bin/python /home/bxb1/.codex/skills/llm-experiment-guard/scripts/verify_config_baseline.py \
  <config_path> \
  --baseline /home/bxb1/.codex/skills/llm-experiment-guard/legacy_llama_qwen_e2e_b691_config_baseline.yaml
```

For the current Llama-70B synthetic 128-input/256-output, 256-batched-token
profiling family, use:

```bash
.venv/bin/python /home/bxb1/.codex/skills/llm-experiment-guard/scripts/verify_config_baseline.py \
  <config_path> \
  --baseline /home/bxb1/.codex/skills/llm-experiment-guard/llama70b_synth128x256_config_baseline.yaml
```

This family baseline fixes the workload and runtime controls but intentionally
does not fix `pipeline_parallel_size`; the global baseline also leaves pipeline
size and max batched tokens as runtime experiment variables.

## Step 2A: Verify vLLM Build Artifacts

Required native artifacts:

```text
vllm/_C.abi3.so
vllm/_moe_C.abi3.so
vllm/cumem_allocator.abi3.so
vllm/kv_cache_allocator.cpython-312-x86_64-linux-gnu.so
vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
```

For the main checkout, verify `.venv` is a real environment and every required artifact exists:

```bash
test -x .venv/bin/python
artifacts=(
  vllm/_C.abi3.so
  vllm/_moe_C.abi3.so
  vllm/cumem_allocator.abi3.so
  vllm/kv_cache_allocator.cpython-312-x86_64-linux-gnu.so
  vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
  vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
)
for f in "${artifacts[@]}"; do
  test -e "$f" || echo "missing: $f"
done
```

For a worktree, verify `.venv` and every required artifact are symlinks that resolve to the main checkout:

```bash
main=/data/gpfs/projects/punim2715/vllm_workbench/vllm
test -L .venv && test "$(readlink -f .venv)" = "$main/.venv"
artifacts=(
  vllm/_C.abi3.so
  vllm/_moe_C.abi3.so
  vllm/cumem_allocator.abi3.so
  vllm/kv_cache_allocator.cpython-312-x86_64-linux-gnu.so
  vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
  vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
)
for f in "${artifacts[@]}"; do
  test -L "$f" && test "$(readlink -f "$f")" = "$main/$f" || echo "bad-link: $f"
done
```

If any artifact is missing or not linked correctly, stop before running the experiment. Either ask to repair links or repair them when the user has already requested that action.

## Step 2B: Verify SGLang Environment

For an SGLang checkout, record the exact worktree, commit/tag, Python
environment, CUDA stack, and launcher module. Verify imports and resolve their
source paths from the selected checkout:

```bash
python -c "import inspect, sglang; print(inspect.getfile(sglang))"
python -m sglang.launch_server --help >/dev/null
```

Verify any model-specific native dependencies required by the chosen attention,
quantization, MoE, or communication backend on the booked GPU node. Do not use
the vLLM `_C` artifact list or vLLM baseline YAML as an SGLang pass condition.
For the workspace layout, distinguish ordinary serving worktrees under
`engines/sglang/.worktrees/` from the autoscaling development checkout at
`engines/sglang/autoscaling`.

## Step 3: Verify vLLM Import Paths

Run from the checkout that will launch the experiment:

```bash
.venv/bin/python -c "import importlib.util; names=['vllm._C','vllm._moe_C','vllm.cumem_allocator','vllm.kv_cache_allocator','vllm.vllm_flash_attn._vllm_fa2_C','vllm.vllm_flash_attn._vllm_fa3_C']; [print(n, importlib.util.find_spec(n).origin if importlib.util.find_spec(n) else None) for n in names]"
```

It is acceptable for import-time warnings to mention missing `libcuda.so.1` on login nodes; this preflight checks path resolution, not GPU execution.

## Step 4: Resolve KV Backend And Layout Intent

Always identify the requested KV-cache mode before editing or launching a config. Do not infer from one overloaded phrase such as "physical block size" unless the user's intent is already clear.

### Two Layer-Stacking Control Planes

There are two separate layer-stacking configuration namespaces. Keep them distinct when explaining, editing, and validating configs:

- **Native VMM/VRM layer stacking** is controlled by vLLM config/env knobs, not by KVCacheD envs. The relevant knobs are `use_vmm`, the sweep `block_size`, and `VLLM_PAGE_ATTENTION_BLOCK_SIZE_KB`; the native code uses these values to derive layer grouping for VMM-backed combined-layer allocation.
- **KVCacheD layer stacking** is controlled by `KVCACHED_*` envs after selecting `VLLM_KVCACHE_BACKEND=kvcached`. The relevant knobs are `KVCACHED_LAYER_STACKING`, `KVCACHED_CONTIGUOUS_LAYOUT`, `KVCACHED_PAGE_SIZE_MB`, and the logical layer-stacked block-size env.

Do not use `KVCACHED_*` envs to describe native VMM/VRM layer stacking. Do not use vLLM `block_size` alone to describe KVCacheD's backing allocation page size.

### Conceptual Names

Use these meanings when discussing or designing configs:

- **PageAttention block size**: semantic vLLM attention block footprint, K+V total. Current env: `VLLM_PAGE_ATTENTION_BLOCK_SIZE_KB`.
- **KVCacheD physical allocation page/block size**: true KVCacheD/CUDA backing allocation granularity. Current env: `KVCACHED_PAGE_SIZE_MB`. Treat this as the real physical page size, even though the current env name says "page".
- **KVCacheD logical layer-stacked block size**: effective per-layer K+V block size after stacking layers into one backing allocation page. Preferred future env name: `KVCACHED_LOGICAL_BLOCK_SIZE_KB`. In the current code, this is still spelled `KVCACHED_PHYSICAL_BLOCK_SIZE_KB`; call out that this is legacy/misnamed when using it.

Do not describe `KVCACHED_PHYSICAL_BLOCK_SIZE_KB` as the true CUDA physical allocation size in the current code. In current layer-stacking code it controls the logical/effective per-layer stacked block size.

### Mode Matrix

Native VMM / no KVCacheD:

- Do not set `VLLM_KVCACHE_BACKEND=kvcached`.
- Do not set `KVCACHED_LAYER_STACKING`, `KVCACHED_PAGE_SIZE_MB`, or `KVCACHED_PHYSICAL_BLOCK_SIZE_KB`.
- Keep the stable baseline values such as `use_vmm: [true]`, `enable_kv_resize: [true]`, and the PageAttention block size unless the user explicitly changes the experiment family.
- Treat `use_vmm`, sweep `block_size`, and `VLLM_PAGE_ATTENTION_BLOCK_SIZE_KB` as the native VMM/VRM layer-stacking control surface.

Default KVCacheD, no layer stacking:

- Set `VLLM_KVCACHE_BACKEND=kvcached`.
- Set `KVCACHED_LAYER_STACKING=false`.
- For autoscaling/shrink experiments, use `KVCACHED_CONTIGUOUS_LAYOUT=false` unless the user explicitly asks for upstream contiguous layout.
- Set the physical allocation page with `KVCACHED_PAGE_SIZE_MB`, commonly `2`.
- Do not set `KVCACHED_PHYSICAL_BLOCK_SIZE_KB`; current code rejects it when layer stacking is disabled.

KVCacheD layer stacking:

- Set `VLLM_KVCACHE_BACKEND=kvcached`.
- Set `KVCACHED_LAYER_STACKING=true`.
- Set `KVCACHED_CONTIGUOUS_LAYOUT=false`; layer stacking is independent of upstream KVCacheD contiguous layout.
- Set the true backing allocation granularity with `KVCACHED_PAGE_SIZE_MB`, commonly `2`.
- Set the logical/effective layer-stacked block with `KVCACHED_LOGICAL_BLOCK_SIZE_KB` if the checkout supports it; otherwise set the current-code legacy env `KVCACHED_PHYSICAL_BLOCK_SIZE_KB`.
- Keep the native VMM/VRM knobs stable unless the experiment is explicitly changing both control planes.
- Validate: physical allocation page size must be divisible by logical layer-stacked block size; local layer count must be divisible by the derived stacking factor; logical layer-stacked block size must be at least and divisible by the PageAttention K+V block size.
- For block-aligned layer-stacking experiments, set logical layer-stacked block size equal to PageAttention block size, for example `128KB` and `128KB`.
- For slice-packed layer-stacking experiments, set logical layer-stacked block size larger than PageAttention block size, for example `512KB` logical block with `128KB` PageAttention block.

### Clarification Triggers

Ask the user before running if any of these are unclear:

- They say "KVCacheD" but do not specify default vs layer stacking.
- They say "physical block size" and it is not clear whether they mean true backing allocation page size or logical/effective layer-stacked block size.
- They ask for a KVCacheD vs VMM comparison but the config only sets one mode.
- They ask for an autoscaling experiment with `KVCACHED_CONTIGUOUS_LAYOUT=true`; confirm this is intentional because it is usually incompatible with dynamic per-rank layer ownership.

For the saved layer-stacking advantage pair, the intended interpretation is:

- Default KVCacheD: `VLLM_PAGE_ATTENTION_BLOCK_SIZE_KB=128`, `KVCACHED_PAGE_SIZE_MB=2`, `KVCACHED_LAYER_STACKING=false`.
- Layer stacking: `VLLM_PAGE_ATTENTION_BLOCK_SIZE_KB=128`, `KVCACHED_PAGE_SIZE_MB=2`, `KVCACHED_LAYER_STACKING=true`, and current-code `KVCACHED_PHYSICAL_BLOCK_SIZE_KB=128` meaning logical/effective stacked block size 128KB.

## Step 5: Verify Experiment Config Baseline

Before any run, read the target experiment config and compare it against the baseline:

```text
/home/bxb1/.codex/skills/llm-experiment-guard/current_experiment_config_baseline.yaml
```

Run this checker from the checkout that will launch the experiment:

```bash
.venv/bin/python /home/bxb1/.codex/skills/llm-experiment-guard/scripts/verify_config_baseline.py <config_path>
```

The baseline intentionally fixes only the values that should not drift during the current experiments, including:

- `envs` values for persistent NCCL/compiled-DAG/dev-mode behavior.
- Stable `static_config.vllm` settings such as max model length, chunked
  prefill, and CUDA graph mode. Pipeline size and max batched tokens are
  runtime experiment variables and are not fixed globally.
- `sweep_configs[*].vllm` values such as `attention_kernel: [direct]`, `use_vmm: [true]`, `enable_kv_resize: [true]`, `weight_loading_mode: [async]`, and `enable_cpu_weight_cache: [true]`.
- `VLLM_PAGE_ATTENTION_BLOCK_SIZE_KB: "128"`, from which the launcher derives
  the vLLM token `block_size` at runtime.

The model path/name and `static_config.vllm.gpu_memory_utilization` are documented defaults, not hard-fixed fields. The checker should accept changed values for these fields and report that they differ from the defaults.

Do not add shrink/expand-specific fields to the baseline. Runtime choices such
as model, GPU memory utilization, pipeline size, max batched tokens, node
placement, `rank_to_ip`, `pp_layer_config`, request schedules, and total
request count may be chosen per run.

If the checker reports any mismatch, stop before launching. Either fix the experiment config to match the baseline or ask the user whether the baseline itself should be updated.

### Profiling And High-Volume Traces: Default Off

Apply the central policy in:

```text
/home/bxb1/.codex/skills/llm-experiment-guard/profiling_controls.yaml
```

The normal baseline command rejects any enabled profiling or high-volume trace
flag. This applies to the Dynamic PP CUDA Event Profiler, PP lifetime/ref/NCCL
edge traces, NCCL debug, and the autoscaling KVCacheD debug flags that
implicitly enable PP traces. `static_config.vllm.enable_nsight: false` and
`static_config.benchmark.profile: false` remain required baseline values.

For ordinary performance experiments, omit the following variables or set only
the master flag to a falsy value:

```yaml
envs:
  VLLM_DYNAMIC_PP_CUDA_PROFILE: '0'
```

The CUDA profiler is enabled only when
`VLLM_DYNAMIC_PP_CUDA_PROFILE` is one of `1`, `true`, `yes`, or `on`.
The companion variables never enable it by themselves:

- `VLLM_DYNAMIC_PP_CUDA_PROFILE_DIR`
- `VLLM_DYNAMIC_PP_CUDA_PROFILE_FLUSH_RECORDS`
- `VLLM_DYNAMIC_PP_CUDA_PROFILE_SAMPLE_EVERY`
- `VLLM_DYNAMIC_PP_CUDA_PROFILE_KINDS`

For an intentional diagnostic run, set all five variables explicitly and run:

```bash
.venv/bin/python \
  /home/bxb1/.codex/skills/llm-experiment-guard/scripts/verify_config_baseline.py \
  <config_path> \
  --baseline <family_baseline_path> \
  --allow-profiling
```

With `--allow-profiling`, the guard requires a nonempty output directory,
positive `FLUSH_RECORDS` and `SAMPLE_EVERY`, and at least one explicit kind
from `stage`, `transformer`, `send`, or `recv`. Do not use
`--allow-profiling` for a normal performance run.

Record enabled diagnostics, sampling, kinds, and flush settings in the
experiment manifest. Treat an instrumented result as diagnostic; pair it with
a clean profiler-off control before using it as a performance frontier.

### Pipeline Concurrency Gap: Pending Exploration

Treat pipeline depth and the number of in-flight batches as separate controls:

```text
logical_pp = number of active, nonempty pipeline stages
effective_concurrent_batches =
    VLLM_DYNAMIC_PP_MAX_CONCURRENT_BATCHES, when explicitly set
    logical_pp, otherwise
concurrency_gap = logical_pp - effective_concurrent_batches
```

The current implementation default is gap `0`. In particular, leave
`VLLM_DYNAMIC_PP_MAX_CONCURRENT_BATCHES` unset for the default PP5 behavior:
five active stages use five concurrent batches. Do not make PP5 depth `4` the
guard default and do not add a value of `4` to the generic config baseline.

Before a run:

- Report logical PP, effective concurrent batches, and concurrency gap.
- Treat an explicit override as a first-class experimental variable; keep the
  workload, PP layout, placement, KV mode, and other serving controls fixed.
- Flag comparisons that change the gap, even when logical PP is unchanged.
- Store the configured and effective values in the experiment manifest and
  verify the server log reports the intended batch-queue size.

The saved PP5 experiment at
`/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/llama70b_a100_burstgpt_pp5_concurrent4_quickpivot_20260728`
is evidence that gap `1` can be useful: PP5 depth `4` recovered PP4-like
tokens per step and reduced scheduler steps and P99 TPOT while preserving
arrival-limited throughput. It does not establish gap `1` as universally
optimal or prove higher saturated throughput.

Keep the control policy unresolved for now. Both of these remain candidates:

- a static per-PP concurrency value selected before launch;
- a dynamic controller that changes effective concurrency according to
  runnable work, batching efficiency, and latency SLO pressure.

Do not choose, implement, or auto-tune either policy without an explicit
experiment plan. Until that decision is made, preserve the implementation
default and use nonzero gaps only in named, controlled experiments.

## Step 6: Discover Existing Booked Nodes

Run:

```bash
squeue --me --format="%.18i %.9P %.20j %.8u %.2t %.10M %.6D %.20R %b"
```

Use only nodes listed by this command. If no usable nodes are listed, stop.

For each candidate node, probe serially:

```bash
ssh -o ConnectTimeout=5 <node> "hostname; ray status 2>/dev/null; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

Prefer already-booked nodes that match the experiment config and are idle. Do not change the reservation, partition, QoS, or GPU count.

### Build An Exclusive Parallel Run Plan

Before launching more than one experiment, resolve the complete node set for
every run and assign runs to execution waves.

- Treat the node, not the individual GPU, as the scheduling unit. One active
  model experiment owns the whole node.
- Treat every node named by `rank_to_ip` or the equivalent placement as owned
  by that experiment. Node sets of concurrently running experiments must be
  pairwise disjoint.
- Maximize safe parallelism across the already-booked nodes. For example,
  three independent single-node profiles should run together on three idle
  nodes; a two-node PP run may run beside one single-node profile only when the
  third node is disjoint.
- Serialize experiments that need any common node. Never co-locate two model
  experiments merely because unused GPUs or memory appear available.
- Count a node as occupied until the server, Ray workers, benchmark client, and
  experiment finalization relevant to that run have exited. Verify no stale
  model process remains before assigning the node to the next wave.
- Record `experiment_id`, complete node set, wave number, start/end time, and
  checkout/config identity in the experiment manifest or orchestration notes.

Show the planned waves before a large profiling batch. Prefer the schedule
with the shortest critical path while preserving comparable experimental
conditions; do not sacrifice node exclusivity to reduce wall-clock time.

## Step 7: Run From the Correct Node

Read the experiment config before launching. If the config has `rank_to_ip`, launch from rank 0's booked node. If rank 0 is remote, SSH there and run the foreground experiment command.

Use the checkout-local shared environment:

```bash
source .venv/bin/activate
python -m vllm_exp.run sweep-test --config <config_path> --single-server
```

Keep each command in the foreground of its own tracked terminal/PTTY session
with full output tracking. Do not use untracked shell background jobs. Multiple
sessions may run concurrently only when their resolved node sets are disjoint
under the exclusive parallel run plan above.

## Failure Policy

Report and stop when:

- Build artifacts are absent in the main checkout.
- Worktree `.venv` or native `.so` symlinks do not resolve to the main checkout.
- The experiment config does not match the fixed baseline file.
- No already-booked node is available.
- The config references nodes outside the current booking.
- A required booked node is busy or unreachable.
- Another model experiment already owns any node required by the run.
- The proposed parallel run plan assigns the same node to multiple active
  experiments.

## CPU KV Checkpoint Preflight

For repeated or long-context SGLang Decode experiments, also apply
sglang-cpu-kv-checkpoint-cache before launch. Query its shared registry before
starting a model server. Reuse only a fingerprint-compatible checkpoint that
passes live node, boot-ID, path, and inventory validation. On a miss, build and
register the checkpoint, stop SGLang, and measure Decode with a fresh process.
A populated /dev/shm directory without a valid registry hit is not reusable
evidence.
