---
name: spartan-node-manager
description: Manage Spartan A100 whole-node bookings, save a persistent NIC and negotiated PCIe inventory, book known high-PCIe or exploratory nodes, and inspect or prune allocations under the user's retention policy. Use for FEIT, deeplearn and public A100 subscriptions, hardware classification and requested periodic checks; model experiments use Experiment Guard separately.
---

# Spartan node manager

Entrypoint: `/data/gpfs/projects/punim2715/bookings/A100-node-manager.sh`.
This is a one-shot tool. Do not restart the retired ConnectX-6 watcher or install
a background service. Scheduling periodic checks requires explicit scheduling
authorization; running a monitor once does not authorize a schedule.

## Evidence and state

Canonical state is `/data/gpfs/projects/punim2715/bookings/node_inventory/`.
`inventory.json` and `observations.jsonl` store NIC model/PCI IDs, actual speed
and width, parent PCIe links, port rates, GPU model, boot ID, timestamp and source.
`groups.json` and `inventory.csv` are derived views. An SSH failure keeps old
evidence as historical and makes the current check unknown. Never infer NICs
from `dlg5`, partition, hostname, rack, or port speed alone.

High means NIC PCIe >=16 GT/s and >=x8; low means valid measured PCIe below
that profile. Missing or inconsistent evidence is unknown. Keep hardware group
separate from active RDMA ports and measured throughput. Current cancellation
decisions require a fresh successful probe, never only a historical grouping.

## Modes

```bash
cd /data/gpfs/projects/punim2715/bookings
./A100-node-manager.sh status
./A100-node-manager.sh inventory
./A100-node-manager.sh monitor               # refresh + cancellation preview
./A100-node-manager.sh monitor --apply       # apply standing policy once
./A100-node-manager.sh book --mode high --count 1 --partition deeplearn
./A100-node-manager.sh book --mode discover --count 2 --partition feit-gpu-a100
```

Booking defaults to preview; `--submit` submits and `--test-only` only checks
Slurm admission/forecast. Explicitly distinguish preview from submitted jobs.
`--renew` allows successors to running high nodes when the user asks to renew.
High mode only targets individually verified high nodes in the selected
partition, skipping existing exact-node pending reservations. Discovery mode
requests arbitrary whole A100 nodes and classifies the actual node on allocation.
Every new batch payload records local hardware once, then holds the booking;
it does not poll, prune, or renew jobs.

Follow the user's `A100-fullnode.sh`: 4 A100s, 32 CPUs and 7 days. Private
partitions use 700000 MiB; public A100 uses 450000 MiB to fit smaller host RAM.
FEIT uses `punim2565` / QoS `feit`; public uses `punim2715` / `publicgpu`.
Pending commitments inform advice and duplicate avoidance; they do not count
toward the four-running-node limit. Do not silently replace pending reservations
or add duplicate exact-node jobs. Model work still requires Guard.

## Standing retention authorization

Read [references/policy.md](references/policy.md) before cancellation. The user
authorized cancelling idle low-PCIe full A100 nodes above two usable running
nodes, leaving at least two. Above four, idle high nodes may also be cancelled
down to four. At 3–4 high nodes, notify rather than cancelling high nodes.
Unknown hardware, active experiments, GPU workloads and CPU weight caches are
protected. Recheck ownership, queue, current hardware, survivors and Guard before
each cancellation. Never cancel H100/L40S/MIG, partial allocations, multi-node
jobs, or another user's work. `monitor --apply` is authorized under this scope.

## Periodic checks

The user explicitly requested one skill run per day on 2026-10-12. Count only
RUNNING nodes toward the four-node cap; show PENDING separately. Scheduling
intent is saved in `node_inventory/monitor_preferences.json`. Check the actual
scheduler before claiming it is enabled; saved intent alone is not a schedule.

First discover the app automation tool if the user requests scheduling. Use the
one-shot `monitor --apply --quiet` payload, updating inventory on every run.
Stay quiet if unchanged; notify on new classification, allocation start/end,
cancellation, high-node excess, errors or required action. Monitoring never books
fresh nodes independently. If scheduling tools are unavailable, disclose it and
provide the command; never claim a schedule or substitute a daemon.

Recommend four known high nodes as the target, with two usable nodes retained
during exploration. Check after each new allocation and about once daily while
exploring. Report RUNNING and PENDING separately, evidence dates, changes,
protected nodes, cancellations and whether any actual schedule exists.
