# Expert T × E doubling comparison



One GPU (A100 GPU0, node065), one real expert layer per model. No Attention, shared experts, router, communication or model loading in timing.

T means actual received local token-expert rows, not unique user tokens or request batch size. E means balanced active local experts. Resident expert counts remain fixed at Hy3=48, GLM=40.

Tokens: 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192. Active experts: 1, 2, 4, 8, 16, 32. E>T cells are invalid and omitted.

Both eager and CUDA Graph are measured separately, 30 CUDA-event samples/cell in 3 rotating/reversed blocks. Immediate warmups: warm-cache replay, not a full-model cold-HBM simulation.



| Model | H | I | Parameters/expert | Checkpoint bytes/expert | Backend |

|---|---:|---:|---:|---:|---|

| Hy3 BF16 / Triton | 4096 | 1536 | 18,874,368 | 36.00000 MiB | HY3 BF16 production Triton compact expert core |

| GLM-4.7 AWQ INT4 / Humming | 5120 | 1536 | 23,592,960 | 11.68945 MiB | Humming AWQ W4A16 GROUPED_MASKED |



GLM has 25% more parameters per expert; at equal BF16 it would be45 MiB vs Hy3 36 MiB. The tested GLM checkpoint is INT4 AWQ including scales/zeros, so Hy3 source weights per expert are about3.08x larger. Source bytes are a footprint proxy, not measured device/HBM traffic. Different precisions, kernels, layouts and engine revisions are intentional deployment-path comparisons; this is not a precision-controlled weight-bandwidth experiment.



| T | E | GLM median ms (Graph) | Hy3 median ms (Graph) |

|---:|---:|---:|---:|

| 8 | 2 | 0.0563 | 0.0666 |

| 8 | 8 | 0.0922 | 0.1956 |

| 32 | 2 | 0.0635 | 0.0696 |

| 32 | 8 | 0.0932 | 0.1976 |

| 32 | 32 | 0.2826 | 0.7173 |

| 128 | 2 | 0.0819 | 0.1690 |

| 128 | 8 | 0.0973 | 0.2621 |

| 128 | 32 | 0.2857 | 0.9236 |

| 512 | 2 | 0.2775 | 0.1812 |

| 512 | 8 | 0.2488 | 0.3000 |

| 512 | 32 | 0.2959 | 0.9820 |

| 2048 | 2 | 0.6584 | 0.6579 |

| 2048 | 8 | 0.6523 | 0.6625 |

| 2048 | 32 | 0.6584 | 1.1981 |

| 8192 | 2 | 2.2968 | 2.3685 |

| 8192 | 8 | 2.2748 | 2.4776 |

| 8192 | 32 | 2.3066 | 2.5718 |



At fixed T, ideal useful GEMM FLOPs are 6 H I T. Increasing E changes the active weight working set E*W, per-expert rows T/E, padding, scheduling and reuse; it does not multiply ideal useful FLOPs by E. No HBM/L2/occupancy counters were measured, so the E penalty cannot be assigned wholly to weight reads.

Hy3 records the actual selected BLOCK_M and padded rows per cell in raw results and CSV. GLM uses the validated AWQ GROUPED_MASKED path, fixed capacity8192/expert and expected_m=ceil(T/40) within each E sweep.

This local compute test alone cannot rank PP2×EP4 vs EP8 end to end. PP must also compensate for increased local expert cost, PP boundaries, critical-path scheduling and capacity-dependent batch changes with communication/capacity gains.



Maximum within-run block-median drift: GLM 18.9%, Hy3 16.8%. Three blocks are not three independent process launches.



Artifacts: full_metrics.csv, validation.json, sources.json, expert_doubling_surface.png/pdf, expert_doubling_curves.png/pdf. Raw result source paths and pinned commits are in sources.json.


## Historical exact placement anchors (separate cohort)

Hy3 EP8 owns24 experts/card; PP2×EP4 owns48. With two equal pipeline microbatches and balanced top8 source rows, each has the same average local T at fixed global input work. Existing real-weight matched core measurements (not a new full-model run):

| Local T | EP8 core ms | PP2×EP4 core ms | PP/EP |
|---:|---:|---:|---:|
| 128 | 0.7593 | 1.4106 | 1.858 |
| 512 | 0.8079 | 1.4715 | 1.821 |
| 2048 | 1.0337 | 1.6927 | 1.637 |
| 8192 | 3.0413 | 3.2497 | 1.069 |

Source: /data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_expert_surface_20261022_r2/matched_comparison.csv

Do not add these core times to full-model curves: this is local compute only, not pipeline critical-path or communication accounting.
