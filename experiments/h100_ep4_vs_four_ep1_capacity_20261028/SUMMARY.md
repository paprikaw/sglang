# 35B：EP4 与四个 EP1，简单结论

同一四卡 H100 NVLink 节点，Qwen3.6-35B-A3B，完整 BF16；EP4 的 Attention 为 TP1×DP4，四个 EP1 是四个独立单卡副本。每卡16个活动请求，FP32 GDN状态；没有使用CPU KV恢复。

| 指标 | 四个 EP1 合计 | EP4 |
|---|---:|---:|
| 本配置下 KV token 池 | 640,768 | 8,692,992（13.57×） |
| 同负载热 Prefill 吞吐 | 125,543 tokens/s | 94,356 tokens/s（低24.8%） |
| 同负载 Decode 吞吐 | 5,327 tokens/s | 5,600 tokens/s（高5.1%） |

上表性能负载为64请求、每请求缓存8,256 tokens；热Prefill各新增512 tokens、输出1 token，Decode各新增1 token、输出128 tokens。各3轮正式测量，profiler另跑。

接近EP4容量上限：64请求各缓存130,304 tokens，合计8,339,456 tokens，占token池95.9%。EP4热Prefill为32,242 tokens/s；Decode为2,237 tokens/s，平均TPOT 23.89 ms。

四个EP1处理同样长上下文时，每卡只能完整驻留1条请求；Decode整体吞吐92.6 tokens/s，EP4约24.1×。这个比较包含四个EP1的排队和前缀重算，不能当作纯Expert算子加速倍数。容量压力对照只有1轮。

独立GPU采样显示，同负载下EP4每步Expert GEMM约减少41.7%，同时增加约1.51 ms的DeepEP kernel开销。本轮容量优势很强；Expert增效没有让所有阶段都变快，Prefill整体仍有损失。

数据与收尾：11项指标均已完成并独立核验。包装脚本末尾误捕获正常SystemExit(0)，Guard总状态保留failed；缓存和清理后检通过，自有模型已退出、租约已释放。修复版另存，未改原始冻结记录。

[详细结果、原始数据与核验说明](REPORT.md)
