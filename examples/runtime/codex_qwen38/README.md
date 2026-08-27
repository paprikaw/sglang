# Qwen3.8-27B-Uncensored on Spartan

This directory records the Codex compatibility and deployment layer used by the
local Qwen service. It does not patch SGLang core source files. Normal model
serving uses the stable `sglang-serving` worktree on branch
`codex/sglang-serving`, based on the upstream SGLang commit recorded in
`deployment.json`.

Secrets and generated runtime artifacts are intentionally excluded. Create
`api.key` from `api.key.example` before launching, keep it mode `0600`, and do
not commit it. `codex-models-combined.json` is also excluded because it is a
generated snapshot of the active Codex catalog; merge `codex-models.json` into
the current catalog during installation instead.

## Use from Codex

Reconnect the Spartan host in the desktop app (or restart the app) so its
app-server reloads the startup-only model catalog. In a **new task**, select
**Qwen3.8-27B-Uncensored · A100** from the model picker. The original default is
still **GPT-5.6-Sol**; other GPT entries remain available.

On `spartan-login1`, including a terminal connected to that host:

```bash
codex --model orcarouter/Qwen3.8-27B-Uncensored
```

This uses the same router and freeform-tool compatibility as the desktop.
The separately configured `codex-qwen38` / `codex --profile qwen38` command is a
direct SGLang fallback with standard function tools, independent of the router.

The profile is `/home/bxb1/.codex/qwen38.config.toml`. It overlays the normal
configuration and does not change the existing `gpt-5.6-sol` default or execution
policy. The model has a 262,144-token context window; the profile sets the automatic
compaction threshold to 249,000 tokens. These are configured limits, not a long-context
benchmark result.

## Desktop integration and rollback

Only two global configuration keys were added to `/home/bxb1/.codex/config.toml`:
`openai_base_url` points to the loopback router at port 31038, and
`model_catalog_json` points to `codex-models-combined.json`. The existing default
model, reasoning settings, compaction threshold, and execution policy were not
changed. A fresh app-server, with no CLI overrides, returned both GPT and Qwen in
`checks/app-server-global-model-list.json`. The running desktop window itself
was not restarted or visually inspected by this task.

The router sends Qwen requests to SGLang using only the local deployment key.
For predictable single-A100 latency it caps local Qwen requests to `low`
reasoning effort by default, even if the global Codex setting is `max`/`xhigh`.
Set `QWEN_REASONING_EFFORT` on the router process to change that policy. Router
timing logs contain only input item types/counts and never prompt or tool text.
Codex remote-compaction items are bridged locally: the router asks Qwen for a
handoff summary, returns it as a `compaction` item, and decodes that item back
into local context on the next request. This avoids depending on OpenAI's
server-side encrypted compaction format.
Other models continue to the official ChatGPT/OpenAI upstream, with HTTP and
WebSocket forwarding. It accepts only this user's current Codex credential or
the private local key; a local key cannot access the OpenAI upstream. Namespace
and freeform tools are adapted to SGLang function calls and converted back on
the response stream. Native Codex `apply_patch` and terminal calls were both
tested through this route.

The combined catalog is a snapshot of the current Codex model catalog plus Qwen.
It may need refreshing when new official models become available. The local
model service is configured for the model's native 256K (262,144-token) context limit. Long-session desktop auto-compaction and
provider-hosted web/image-generation tools have not been validated; use a fresh
Qwen task and avoid carrying over a large GPT conversation.

To remove the routing/catalog overrides while preserving later unrelated
configuration changes:

```bash
python3 /data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored/rollback_desktop.py
```

Then reconnect the Spartan host. The full pre-change backup is
`/home/bxb1/.codex/config.toml.before-qwen38-20260827.bak`. Do not stop the router
before rolling back/reloading clients that depend on it.

## Deployment

| Setting | Value |
| --- | --- |
| Model | `orcarouter/Qwen3.8-27B-Uncensored` |
| Model revision | `9878936be9458522b5aeed0e13476bb8426f57f0` |
| Engine | SGLang 0.5.18, upstream source commit `ea48cb04cc33233e01674e218102746931f72128` on `codex/sglang-serving` |
| Node / GPU | `spartan-gpgpu142`, GPU 0, A100 80GB PCIe |
| Existing Slurm booking | `29526906` |
| Booking expiry | 2026-08-31 14:56:13 Australia/Melbourne |
| Weights | BF16; 18 shards; 55,563,006,776 bytes |
| Context / KV capacity | 262,144 tokens |
| Maximum concurrent requests | 4 |
| Direct API on login host | `http://127.0.0.1:30038/v1` |
| Codex router | `http://127.0.0.1:31038/v1` |

The service binds to loopback on the GPU node. A loopback-only SSH tunnel makes
it available on `spartan-login1`. A private local API key is read from `api.key`
using Codex's command-backed authentication. No Hugging Face or OpenAI token is
passed to the model server. Unauthenticated model-list requests return HTTP 401.
The service is for personal use and is not exposed as a public endpoint.

## Service management

Server, tunnel, and router run in named tmux terminals, on their respective hosts, using
the dedicated tmux socket `qwen38-codex`. The launcher refuses to start if its
booking is absent or another GPU compute process is already using this node.
It never requests nodes or changes reservations.
The server's verified cgroup is
`/system.slice/slurmstepd.scope/job_29526906/step_extern/user/task_0`, so it remains
inside the booked Slurm job rather than outliving that allocation.

```bash
python3 /data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored/manage.py status
python3 /data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored/manage.py stop
```

To start after stopping, while the same node is still booked and idle:

```bash
python3 /data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored/manage.py start
python3 /data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored/manage.py tunnel
python3 /data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored/manage.py router
```

The source and dependency environment are reused read-only. CUDA variables and
include paths are isolated in `environment.sh` so the login CUDA 12.4 module does
not contaminate CUDA 12.8 kernel builds. Kernel compilation runs on the booked
GPU node. The service depends on these existing source/environment paths staying
available; no shared packages or experimental source files were modified.

## Verification

- A fresh app-server model list shows the local Qwen entry as `256K context`.
- The live `/v1/loads` endpoint reports both `max_total_num_tokens` and
  `memory.token_capacity` as 262,144 after the latest-source restart.

- `checks/api-checks.json`: correct model identity, Chinese chat response,
  Responses SSE completion, function call arguments and function-result round trip.
- `checks/vision-check.json`: correctly identified a generated solid-red test image.
- `checks/codex-ready.jsonl`: actual Codex CLI received `READY` through this model.
- `checks/codex-tool-current-config.jsonl`: actual Codex executed exactly
  `printf CODEX_QWEN_TOOL_OK`, received exit code 0 and the expected output.
- `checks/router-patch.jsonl`: actual Codex used the native patch tool to create
  `checks/coding/probe.py`, ran it, and received `QWEN_PATCH_OK` with exit code 0.
- `checks/global-openai.jsonl` / `checks/global-qwen.jsonl`: both backends tested
  using the installed global routing configuration, without endpoint overrides.
- `test_router.py`: authentication, compression, fixed upstreams, namespace and
  freeform-tool transformations tested locally.
- The extra read-only sandbox test in `checks/codex-tool.jsonl` encountered the
  host's `bwrap: Creating new namespace failed: No space left on device` error.
  The successful test used the user's existing execution policy without changing
  sandbox configuration.

The direct SGLang API passes only standard function tools to this model. The
Codex router additionally supports the namespace/freeform tools verified above.

Main service log: `logs/server-service.log`. Pinned provenance and verification
paths are recorded in `deployment.json` and `model_snapshot.json`.
