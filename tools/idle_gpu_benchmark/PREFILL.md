# 持续并发 Prefill benchmark

空闲 GPU 的默认后台策略是 `retention: none`，使用 `prefill_worker_ephemeral.py`。各节点按配置的并发数持续发送请求，每一路完成后立即发送下一条；不设 RPS、不重放到达间隔、不用队列限流。节点独立运行，一个节点被抢占不会阻塞其他节点。GPU 利用率取实际采样值，不同输入长度不保证达到硬件理论峰值。

输入长度来自原始 BurstGPT CSV，按原始 `source_index` 对固定节点取模分配。保留原始输入 token 数，每条只生成 1 个 token；跳过输入加 1 超过模型上下文上限的行。原始输出长度仅作为内存中的元数据，不再作为生成长度。原始文本不在数据集中，合成 token IDs、输出内容和每请求测量均只存在于内存。

不保存后台 benchmark 的 request JSONL、workload、CSV、批摘要、历史目录或 cursor 文件。仅原子替换一个当前状态文件：`<runtime_root>/nodes/<node>.json`。该文件不超过 32 KiB，含当前状态、部署绑定、计数、吞吐和最新一批的聚合延迟；字段为 `storage_layout: ephemeral_live_v1`、`retention: none`，`stage_dir` 和 `trace_cursor` 均为 null。每次更新替换旧状态，不累积历史。临时替换文件在成功或失败后清理；遥测写入因配额失败不会主动中止负载。

TTFT、E2E 和实际排队时间仍按现有传输协议测量，在内存中聚合为最新分位数。TPOT 无定义，因为每条只生成一个输出 token。最新聚合值仅用于实时状态展示，随下一批替换，不形成实验结果文件。Radix cache 和 HiCache 的开关由模型部署配置控制。

负载是模型 Guard 的子进程，只能访问自己节点的 endpoint。每 50 ms 检查自己的 lease 与抢占信号；失效时取消在途请求，随后由 Guard 清理模型。抢占结束后，管理器恢复符合当前节点权重与容量的部署并启动新一轮负载。Trace cursor 仅存在于进程内存；全部成功才推进，失败重试当前批。进程重启或新部署从原始 trace 开头开始，不恢复旧 cursor。

旧的 `prefill_worker.py`、`prefill_worker_append.py` 和持久化 trace 实现继续保留，用于识别冻结的历史协议；`retention: none` 下 coordinator 拒绝启动这些保留结果的旧 worker。后台策略不改变前台实验保存结果、模型权重或 KV checkpoint 的规则。

CPU 验证不启动 GPU：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s tools/idle_gpu_benchmark -p test_prefill_worker_ephemeral.py -v
```

原 RPS 搜索与持久化结果协议见 [README.md](README.md) 的历史说明。
