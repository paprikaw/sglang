# A100 Qwen35B BF16: EP2 versus two-slot PP2

Fixed 2 x A100, same GPUs 0/1 sequentially, complete 40-layer model.
EP2: Attention DP2/TP1, MoE EP2; PP2: Attention TP1, MoE EP1, 20 layers/stage and two 4-request slots.

Eight concurrent requests: cold P0/Q4096/O1 Prefill; GPU-hot P4096/Q1/O128 Decode. Three formal rounds per cell. No CPU KV checkpoint exists for this model in the shared registry; real native GPU radix/GDN prefixes are built outside timing.

Both arms use native BF16 Triton Experts, Triton Attention, DeepEP normal for EP, CUDA Graph and overlap disabled. The private interface connects token-major DeepEP local IDs to unchanged native Triton math. Before serving, each rank validates actual loaded weights/whole-layer calls against an independent allgather/local-expert/allreduce reference, including skew/zero-receive routes. No quantization or model truncation.

Performance and profiler cohorts are separate; both rank gzip exports are joined and CRC/JSON checked before model cleanup. CPU weight cache and whole-node lease use the previously qualified unchanged Qwen Guard bundle.

Run prepare.py with a new absolute run directory, then execute preparation.json command with its required_environment. Never modify an admitted run.
