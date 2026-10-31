## 1 EP2 分专家、逐层交换；PP2 分层、阶段传输

Conceptual four-layer example, not a measured execution trace. EP2 partitions the routed experts of every layer over both GPUs; both GPUs participate in each successive layer of a given forward. Orange double arrows denote dispatch and combine exchange at every routed MoE layer. PP2 assigns layers L1/L2 to GPU0 and L3/L4 to GPU1, with the full expert set for each local layer. Forward activations cross the stage boundary. The A/B microbatch diagram illustrates the middle pipeline slot: GPU0 handles B at stage0 while GPU1 handles A at stage1. The EP layer sequence describes one microbatch, not a general prohibition on request concurrency or overlapped kernels. Arrows are schematic communication locations, not measured byte counts or time durations. Attention TP1 / DP2 EP versus Attention TP1 PP follows the saved deployment description.

## 2 GPU 不同，EP / PP 的优劣会反转

Compare saved native deployment baselines: A100 Marlin/eager and H100 DeepGEMM/Graph. E2E is the mean complete-request latency, output128. Expert is routed GEMM only, matched using CPU module/kernel correlation on A100 and native Graph on H100. D+C uses DeepEP GPU kernels; pipeline is NCCL GPU residency including wait. Per actual layer-call normalization avoids comparing 48-layer EP against 24-layer PP totals. Not an additive E2E breakdown.

## 3 模型不同，PP 额外付出的 Expert 代价不同

35B BF16 and 122B FP8 are the named native checkpoints, not a size-only ablation. E2E is mean complete-request latency across3 formal cohorts. Right chart is routed Expert module GPU kernel activity for the whole B8 Prefill cohort, rank mean; excludes Shared Expert. EP D+C and PP NCCL retained separately. Historical 122B Prefill uses ready-send enabled;35B baseline disabled. Small-model EP Prefill varies across3 rounds. Do not subtract GPU activities as exposed E2E latency.

## 4 阶段不同，EP / PP 的选择也会改变

Same122B H100 model; phase-specific native paths. Prefill mean E2E uses output1;Decode mean E2E includes TTFT plus128 outputs. GPU comparison selects Expert GEMM for both phases and DeepEP dispatch/combine only. Actual per-layer invocation units; Prefill has larger fresh-token work than Decode. PP communication is profiled NCCL residency including wait, not wire-only latency. Formal timing is independent of profiles.
