## 1 GPU 不同，EP / PP 的优劣会反转

Compare saved native deployment baselines: A100 Marlin/eager and H100 DeepGEMM/Graph. E2E is the mean complete-request latency, output128. Expert is routed GEMM only, matched using CPU module/kernel correlation on A100 and native Graph on H100. D+C uses DeepEP GPU kernels; pipeline is NCCL GPU residency including wait. Per actual layer-call normalization avoids comparing 48-layer EP against 24-layer PP totals. Not an additive E2E breakdown.

## 2 模型不同，PP 额外付出的 Expert 代价不同

35B BF16 and 122B FP8 are the named native checkpoints, not a size-only ablation. E2E is mean complete-request latency across3 formal cohorts. Right chart is routed Expert module GPU kernel activity for the whole B8 Prefill cohort, rank mean; excludes Shared Expert. EP D+C and PP NCCL retained separately. Historical 122B Prefill uses ready-send enabled;35B baseline disabled. Small-model EP Prefill varies across3 rounds. Do not subtract GPU activities as exposed E2E latency.

## 3 阶段不同，EP / PP 的选择也会改变

Same122B H100 model; phase-specific native paths. Prefill mean E2E uses output1;Decode mean E2E includes TTFT plus128 outputs. GPU comparison selects Expert GEMM for both phases and DeepEP dispatch/combine only. Actual per-layer invocation units; Prefill has larger fresh-token work than Decode. PP communication is profiled NCCL residency including wait, not wire-only latency. Formal timing is independent of profiles.
