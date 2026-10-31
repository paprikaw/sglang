# First CUPTI collector after a pre-existing NCCL CUDA Graph

Two fresh two-rank Torch processes use the same booked single node, actual GPU0/1,
full-node four-GPU registration, same PyTorch build and same preserve-CUPTI flags.
No model weights, KV cache, Attention or Expert computation is involved.

Each case initializes and verifies real bidirectional NCCL P2P, captures one Graph,
replays twice without a profiler, then starts its **first** collector. After three
correlated long idle marker proofs and one eager P2P, it replays the existing Graph
three times. The two cases differ only in CPU+CUDA versus CUDA-only activities.
This isolates CPU activity collection, not CUDA runtime/driver callbacks: PyTorch's
CUDA-only defaults still collect CUDA runtime/driver activity.

Actual loaded libcupti/libcudart paths are taken from /proc/self/maps before capture,
after capture, after collector start and before replay. cuptiGetVersion is called
only on already mapped handles via RTLD_NOLOAD. Installed package versions alone
are never used to identify the active CUPTI library.

Every native-replay boundary is persisted before execution. A native crash leaves
its mode invalid. The supervisor verifies owned process/GPU cleanup before trying
the second mode. Overall diagnostic completion is separate from the number of
valid GPU activity traces; missing/crashed traces are never measured results.
Captured Graphs are reset before process-group teardown.

CPU tests qualify harness behavior only. They do not establish a CUPTI fix.
