# 计时回放的输出一致程度

参考为E5合并EP4；同一源请求、repeat命名空间和输入构造器，模型revision与图文件哈希匹配。每行是一轮完整episode，不合并重复或部分运行。不同布局面对不同实时batch和缓存历史。

| 候选 | 负载/轮次 | 调用数 | 全输出相同 | 首token相同 | 共同输出前缀token | 总输出token | 双方均无缓存时首token不同/调用数 |
|---|---|---:|---:|---:|---:|---:|---:|
| col_pp2 (eight_gpu_col_pp2_durable_m16_r3) | agentx_c128/r1 | 2736 | 865 | 2462 | 432255 | 1192128 | 56/434 |
| col_pp2 (eight_gpu_col_pp2_durable_m16_r3) | agentx_c128/r2 | 2736 | 887 | 2450 | 421982 | 1192128 | 39/429 |
| col_pp2 (eight_gpu_col_pp2_durable_m16_r3) | agentx_c128/r3 | 2736 | 827 | 2449 | 412980 | 1192128 | 38/427 |
| col_pp2 (eight_gpu_col_pp2_durable_m16_r3) | agentx_c8/r1 | 171 | 39 | 144 | 15514 | 74508 | 3/23 |
| col_pp2 (eight_gpu_col_pp2_durable_m16_r3) | agentx_c8/r2 | 171 | 84 | 153 | 42184 | 74508 | 2/23 |
| col_pp2 (eight_gpu_col_pp2_durable_m16_r3) | agentx_c8/r3 | 171 | 62 | 157 | 34842 | 74508 | 1/23 |
| col_single (eight_gpu_col_single_durable_kv40_r3) | agentx_c128/r1 | 2736 | 872 | 2441 | 440305 | 1192128 | 57/475 |
| col_single (eight_gpu_col_single_durable_kv40_r3) | agentx_c128/r2 | 2736 | 886 | 2421 | 424726 | 1192128 | 70/518 |
| col_single (eight_gpu_col_single_durable_kv40_r3) | agentx_c128/r3 | 2736 | 812 | 2450 | 399346 | 1192128 | 46/434 |
| col_single (eight_gpu_col_single_durable_kv40_r3) | agentx_c8/r1 | 171 | 35 | 144 | 15836 | 74508 | 4/23 |
| col_single (eight_gpu_col_single_durable_kv40_r3) | agentx_c8/r2 | 171 | 87 | 153 | 42516 | 74508 | 2/23 |
| col_single (eight_gpu_col_single_durable_kv40_r3) | agentx_c8/r3 | 171 | 64 | 151 | 36420 | 74508 | 2/23 |
| pd_ep4 (eight_gpu_pd_ep4_durable_slots8_r4) | agentx_c8/r1 | 171 | 57 | 158 | 31475 | 74508 | 4/23 |
| pd_ep4 (eight_gpu_pd_ep4_durable_slots8_r4) | agentx_c8/r2 | 171 | 98 | 155 | 49366 | 74508 | 2/23 |
| pd_ep4 (eight_gpu_pd_ep4_durable_slots8_r4) | agentx_c8/r3 | 171 | 90 | 164 | 48345 | 74508 | 1/23 |

计时输出并非逐请求完全相同；不能把受控串行参考及host恢复检查的通过，表述成整个动态batch回放逐token一致。无缓存请求也存在差异，所以这些配对本身不能定位为缓存错误；同样尚未通过logits等控制将差异归因为浮点归约。

回放固定来源记录的输入/输出长度，后续请求使用来源hash合成输入，不拼接本轮实际生成文本。因此布局间提供的源请求长度一致，但生成内容、实际专家路由和实际缓存工作量不保证一致。性能结论限定于该派生回放；受控KV/GDN交接、恢复和压力修复仍各自需要对应验证证据。

[逐轮数据与来源哈希](output_agreement.json)
