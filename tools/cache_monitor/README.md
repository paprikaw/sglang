# CPU weight / KV cache monitor

Read-only, standard-library dashboard. No model launches, GPU allocations, Torch
imports, cache payload reads, registry updates, eviction, or cleanup operations.

```sh
python -B tools/cache_monitor/server.py --port 0 --interval 15 \
  --receipt /tmp/sglang-cache-monitor.json
```

The receipt prints the loopback URL and owned server PID. The monitor runs on
Spartan's login host and SSH-observes only this user's currently RUNNING booked
nodes. Node observations run in parallel. The collector waits 15 seconds after
each collection; the UI shows actual observation timestamps and collecting state.
Browser refreshes read the cached snapshot and do not launch extra SSH probes.

## What the page measures

- **Weights:** directly call the registered daemon's CPU-only `status` protocol;
  check UID, PID/start identity and source commit; stat and deduplicate its exact
  weight memfd inodes. Display used bank bytes separately from reserved capacity
  and daemon RSS. Canonical weights are a shared full-model copy per node, not
  four CPU copies for four GPUs. Missing registration/daemon is unknown, not zero.
- **KV checkpoints:** discover user-owned `/dev/shm/sglang-prefix-bank` directories
  and combine registry metadata without summing aliases of the same physical
  files. Observe actual regular `.bin` metadata and deduplicate inode identities;
  never open/read KV payloads. Validate the tmpfs device, ownership and paths.
  Unknown filename layouts are included as unclassified bytes.
- **Residency:** allocated backing is counted as RAM resident only when the node
  reports no used/cached swap. Otherwise resident bytes stay unknown. No page
  cache faulting, mmap reads, mincore sweep or payload checksum is performed.
- **Layout:** TP/PP rank geometry comes from file suffixes, layer ranges from
  compatible same-boot registry metadata. No unsupported EP/DP or physical GPU
  mapping is invented. Hardware inventory is read with `nvidia-smi` metadata only.
  Checkpoint validity is not established by a byte count; this is not a reuse gate.
- **Active engine HiCache pools:** shown separately as unknown until a component
  telemetry API exists; process RSS cannot isolate weights versus KV versus other
  allocations. The measured KV subtotal is persistent CPU PrefixBank checkpoints,
  not every anonymous in-process KV allocation.

Cancelled/unbooked/inaccessible nodes retain historical metadata, displayed only
on request and excluded from live totals. The server binds 127.0.0.1 only. JSON
contains allowlisted metadata, not authentication keys, bindings or cache data.
Only fixed HTML/JS/CSS and GET `/api/health`, `/api/status` are served; there are no
mutation endpoints. Existing experiments, caches and dirty worktrees are untouched.

## Tests

```sh
python -B tools/cache_monitor/test_monitor.py
node --check tools/cache_monitor/static/app.js
node tools/cache_monitor/test_frontend.mjs
```

Tests create only tiny disposable files in a unique `/dev/shm/cache-monitor-test-*`
directory. They do not access production cache payloads or GPU resources.
