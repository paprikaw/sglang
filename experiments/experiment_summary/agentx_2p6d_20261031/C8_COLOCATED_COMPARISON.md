# AgentX C8：混部、4P4D 与 2P6D

历史数值对照：2P6D 尚未在请求平均完成时间上超过最佳混部。相对旧 4P4D，2P6D 的 E2E 仅改善约 0.2%–1.5%，TTFT 增加约 3%–8%。这还不足以证明改变 P:D 配比有稳定收益。

均为 Qwen3.6-35B-A3B BF16、8 张 H100、HiCache 开启；C8 是同一批 8 个根会话族，共 171 次请求。14 个完整 episode、2394 次请求已重新检查原始 SHA、独立完成 marker、请求身份／输入输出长度以及延迟均值。没有新增 GPU 实验。

| 部署 | 引擎／回放 | 完整轮次 | 平均 TTFT ms | 请求平均 TPOT ms | 平均 E2E ms |
|---|---|---:|---:|---:|---:|
| 混部：2×EP4 | E5/V3 | 3 | 431.71 | 9.640 | 3542.97 |
| 混部：4×PP2 | E5/V3 | 3 | 277.15 | 7.137 | 3266.10 |
| 混部：8×单卡 | E5/V3 | 3 | 348.46 | 4.990 | 2558.71 |
| 4P4D：EP4 P→EP4 D（历史完整 C8） | E5/V3 | 3 | 572.04 | 6.923 | 3590.81 |
| 2P6D：PP2 P→EP4＋EP2 D | E8/V4 | 1 | 616.39 | 6.670 | 3536.62 |
| 2P6D：PP2 P→3×EP2 D | E8/V4 | 1 | 591.09 | 6.789 | 3585.36 |

延迟越低越好。TTFT／E2E 从 HTTP 发起计时；TPOT 先逐请求计算首 token 后平均间隔，再对请求等权取均值。短输出会影响这一均值。

## Decode 与缓存补充

| 部署 | 按输出 token 间隔加权的 TPOT ms | P 前缀命中 | 实际新增 P token／请求 |
|---|---:|---:|---:|
| 混部：2×EP4 | 7.157 | 91.4619% | 7,245.10 |
| 混部：4×PP2 | 6.876 | 91.4619% | 7,245.10 |
| 混部：8×单卡 | 5.084 | 91.4619% | 7,245.10 |
| 4P4D：EP4 P→EP4 D（历史完整 C8） | 6.944 | 91.4584% | 7,248.09 |
| 2P6D：PP2 P→EP4＋EP2 D | 6.718 | 91.4584% | 7,248.09 |
| 2P6D：PP2 P→3×EP2 D | 6.888 | 91.4584% | 7,248.09 |

PD 缓存只取 P 端 prefill_meta，不能用 D 端的命中数代替。未保存 P 端字段时标缺失。相近缓存命中仍不能消除引擎、预算和到达时序差异。

## 如何解读

- 相对混部单卡：2P6D 的请求 E2E 约慢 38%–40%，现有数据没有显示 PD 分离的整体优势。
- 相对混部 PP2：2P6D 请求 E2E 约慢 8%–10%，改善部分 Decode 间隔没有抵消首 token 成本。
- 相对混部 EP4：请求 E2E 接近；请求等权 TPOT 的较大变化不等于纯 Decode 算力等比例提升，应同时看间隔加权列。
- 相对旧 4P4D：差别很小，且跨引擎／回放，不能归因为 2P6D 比例更优。旧同 E5/V3 的 EP4 混部与 4P4D 配对，也没有看到平均 E2E 改善。

## 可比范围

混部使用 E5/V3，旧 4P4D 同为 E5/V3；新 2P6D 使用 E8/V4。V4 修正分支触发／返回时序，GDN/KV 预算及第二节点也有变化。整轮 wall_s 留在 CSV 中供追溯，不用其跨版本差异声称吞吐提升。表中的旧 4P4D 是 E5 下三轮完整 C8 历史结果，其整组 C128 失败；并非后来 E7/V3、仅有 C128 的三轮压力点。旧 E7 C128 不混入本表。

下一步若验证 PD 收益，应补同一 E8/V4 的最佳混部；若验证 P:D 比例，应补同一版本的 4P4D。保持同样工作负载和缓存协议，记录 P 计算等待、D 接收预分配及完整完成情况。

[绝对数 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/c8_comparison.csv) · [逐轮来源、哈希与差值](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/c8_comparison_evidence.json) · [复算脚本](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/compare_colocated_c8.py) · [当前 2P6D 状态](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/STATUS.md)

[单卡与PD的差距分解](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/agentx_2p6d_20261031/SINGLE_VS_PD_DIAGNOSIS.md)：进一步分开首token前后时间、P排队、低batch布局成本及同输出对子集。
