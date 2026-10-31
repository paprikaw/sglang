# A100 bookings with NIC admission

Run the wrapper with `bash`, not `sbatch`:

```bash
./A100-fast-nic.sh --nic connectx6-pcie4 --count 2 --time 1-00:00:00 --test-only
./A100-fast-nic.sh --nic connectx6-pcie4 --count 2 --time 1-00:00:00 --submit
./A100-fast-nic.sh --status outputs/fast_nic_<request>/request.json
./A100-fast-nic.sh --retry outputs/fast_nic_<request>/request.json --submit
```

Without `--submit` or `--test-only`, the script only prints a preview. Each slot
requests one A100 node / four GPUs, 32 CPUs and 700000 MiB, as in the existing
full-node booking. Default duration is 24 hours. Existing bookings are untouched.

Slurm currently has no NIC-specific feature in `deeplearn`; `dlg5` includes mixed
NIC hardware. This is therefore a candidate-pool selection plus an allocation-time
check, not a native scheduler NIC constraint. The wrapper excludes already-held
nodes and conservatively excludes the 132–143 chassis family for the fast policy;
remaining candidate NICs are unverified until admission succeeds. It uses an
exclusion complement, not `--nodelist` as if that were a list of alternatives.

`connectx6-pcie4` requires `mlx5_bond_0`, Mellanox PCI vendor/device 15b3:101d,
negotiated PCIe speed at least 16 GT/s and width at least x8, plus an ACTIVE RDMA
port. `rdma` only requires that HCA and an ACTIVE port. Dual 100 Gb/s bond slave
inventory is recorded as advisory evidence; this does not measure throughput.

The allocation writes `attestation_<jobid>.json`; a failed admission exits and
releases the allocation. Successful admission only holds the booking and does
not start Jupyter, a model, CUDA, or a bandwidth benchmark. Later experiments must
still obtain the usual global registration and Experiment Guard lease.

Retry is explicit, not an infinite resubmission daemon. It replaces only a job
with a failed admission record after `squeue` and `sacct` confirm it has exited,
excludes the rejected node, and permits at most three attempts per slot. Jobs
cancelled by the user, successfully admitted bookings, and timeouts are not retried.
The wrapper serializes submissions with a file lock and records every job ID in
`request.json`; use that same request on retries. Slurm start estimates can change
and are not confirmed allocations.
