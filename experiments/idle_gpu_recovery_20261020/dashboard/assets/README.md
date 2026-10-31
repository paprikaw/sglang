# GPU resource overview v1

Minimal read-only fleet view: booked node, foreground/background state, weight
bank status and GPU/runtime versus registered persistent KV cache.

Each node also shows its shared RDMA NIC model and negotiated PCIe generation /
width, with separate per-GPU PCIe links. Hover for the actual versus maximum
links and each GPU index. Hardware uses the same read-only sysfs/lspci method as
the node inventory, plus a read-only nvidia-smi PCIe query, in the existing node
SSH sample. It does not require a model cache or an active GPU lease. Unknown
links are never inferred from capabilities, node names or Ethernet speed. Failed
queries show unknown; samples older than 120 seconds are marked historical.
Only the existing latest dashboard status JSON is replaced, without hardware
history or benchmark result files.

GPU utilization is shown for every ownership state, including foreground leases
and experiments. It uses the independent read-only GPU sample, shows the node's
mean utilization, and exposes each GPU's value and sample time on hover. Samples
older than 90 seconds are explicitly labelled old; missing samples stay unknown.
A legacy utilization value without a timestamp is also labelled old.

Background model loading, missing/stale worker progress and confirmed worker
failure are separate states. A healthy server or heartbeat alone is never a
working benchmark. Failed owned workers are recovered by the background manager
through normal Guard cleanup and admission; the dashboard stays read-only.

Served by `skills/llm-experiment-guard/scripts/resource_overview.py --config ...`.
This CPU-only aggregate is independent of experiment dashboards and acquires no
GPU lease. Bind only127.0.0.1. Routes are an explicit asset allow-list plus
`GET /api/status` and `GET /api/health`; no mutation endpoints or filesystem
browsing. Runtime configuration/logs belong outside the Guard source repository.

Sources: RUNNING/PENDING GPU bookings from `squeue --me`; active Guard leases;
existing fleet telemetry; private admitted cache descriptors (only normalized
status is exported); authenticated daemon `status` RPC; registered KV locations
checked against current boot and directory existence. No Torch/CUDA imports,
inference/health-generation calls, allocations, evictions or prefix loads.

Ownership refreshes on every browser poll. Cache RPCs run sequentially on their
exact booked nodes; the table shows sampled cache state (hover for timestamp).
KV directory existence is not checkpoint compatibility/reusability. No registered
KV entry is labelled “no registration”, never proof that arbitrary unregistered
files or a live runtime cache are empty. Runtime KV is a point sample from the
active benchmark's metrics or scheduler log. Foreground endpoint discovery is
not guessed: unavailable foreground runtime KV is shown as not sampled.
