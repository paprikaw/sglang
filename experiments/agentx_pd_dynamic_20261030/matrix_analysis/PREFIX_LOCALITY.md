# 历史前缀位置核查（CPU派生指标）

仅分析合并实例。按实际请求到达客户端HTTP发送时刻，检查同一root/copy家族里此前请求的hash前缀；“同owner”是同实例及Attention-DP rank，“任意owner”是该家族其他位置的历史。分别以响应结束和首token时刻加入历史，不使用未来请求。

| 布局 | 负载 | 历史加入时刻 | 轮数 | 实测cached均值 | 同owner历史hash前缀均值 | 任意owner历史hash前缀均值 | 其他owner额外前缀均值 |
|---|---|---|---:|---:|---:|---:|---:|
| col_ep4 | agentx_c128 | response_end | 3/3 | 71318.24 | 77798.36 | 77849.45 | 51.09 |
| col_ep4 | agentx_c128 | first_token | 3/3 | 71318.24 | 77798.36 | 77849.45 | 51.09 |
| col_ep4 | agentx_c8 | response_end | 3/3 | 77611.04 | 77791.06 | 77849.45 | 58.39 |
| col_ep4 | agentx_c8 | first_token | 3/3 | 77611.04 | 77791.06 | 77849.45 | 58.39 |
| col_pp2 | agentx_c128 | response_end | 3/3 | 75197.17 | 77805.66 | 77849.45 | 43.79 |
| col_pp2 | agentx_c128 | first_token | 3/3 | 75197.17 | 77805.66 | 77849.45 | 43.79 |
| col_pp2 | agentx_c8 | response_end | 3/3 | 77611.04 | 77791.06 | 77849.45 | 58.39 |
| col_pp2 | agentx_c8 | first_token | 3/3 | 77611.04 | 77791.06 | 77849.45 | 58.39 |
| col_single | agentx_c128 | response_end | 3/3 | 44698.56 | 77798.36 | 77849.45 | 51.09 |
| col_single | agentx_c128 | first_token | 3/3 | 44698.56 | 77798.36 | 77849.45 | 51.09 |
| col_single | agentx_c8 | response_end | 3/3 | 77611.04 | 77791.06 | 77849.45 | 58.39 |
| col_single | agentx_c8 | first_token | 3/3 | 77611.04 | 77791.06 | 77849.45 | 58.39 |

**这是hash前缀机会，不是可立即复用的真实KV/GDN。** 不模拟容量、驱逐、GDN checkpoint、chunk写回或服务端排队。首token不是精确cache commit时刻；GPU实际查询也可能晚于HTTP发送，因此上述值与实测cached的差额不能直接解释为驱逐损失。两个历史时刻也不是严格的上下界。

“任意owner”不代表免费传输，也不是family-sticky策略实际执行的结果。改变路由会改变排队、batch和缓存历史；该表只用于决定是否值得做缓存亲和的实测控制，不能换算成动态路由加速。

[逐请求指标](prefix_locality.csv) · [来源哈希](prefix_locality_sources.json)
