# Whole-node A100 CPU memory correction, 2026-10-20

Provenance: running jobs 32715970 (065), 32715983 (167), and 32762791 (169) submitted A100-fullnode.sh; job 33042490 (069) submitted A100-node-managed-fullnode.sh. Explicit node-manager command-line --mem=700000 previously overrode script defaults. Both paths are corrected.

Policy: --mem=0 --exclusive requests all Slurm-configured RAM of whichever node is allocated. It does not remove the physical/node/system limit. Omitting --mem is different and can select partition/GPU defaults. The installed Slurm 25.11.8 man page /apps/slurm/latest/share/man/man1/sbatch.1 lines1466-1467 confirms zero grants all node memory. Current four node RealMemory=1031012 MiB, about1006.85 GiB; the exact limit for a future node depends on that node's configured memory and applicable policy.

Before modification: prechange_tests.json covers the four running-node scripts with command-line --mem=0 --exclusive. other_partition_prechange_tests.json covers FEIT/public/public-preempt. All seven accepted.
After modification: postchange_tests.json records ten successful sbatch --test-only validations, including scripts without command-line memory overrides, all four node-manager partition profiles, and the actual fast-NIC command builder. All returned zero and scheduling forecasts. Shell/Python syntax checks and ten existing node-manager tests passed. Test-only does not submit a booking or guarantee actual start time, future policy acceptance, or future allocation cgroup behavior.

Eight source/document files changed; changes.json records hashes and originals/ stores exact backups. changes.diff contains the reviewable diff. Partial-GPU A100.sh and other GPU-family scripts remain outside this whole-node change. Existing RUNNING and PENDING allocations were not updated or resubmitted; queue_after.txt contains the same ten jobs. Future new submissions through the corrected entrypoints use full node RAM.
