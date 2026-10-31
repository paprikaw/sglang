# AgentX时间口径：源数据长间隔、压缩与分支触发

2026-10-31。当前所有已测布局使用相同、冻结的自定义回放协议；没有改动运行中的图或时序。它是AgentX派生实验，不是官方harness运行时的等价复现。

## 时间单位与长间隔来源

时间单位核对通过。原始256k数据第340行、trace `d876a71a07454e7499469e864f89cf6a3345`：

| 外层请求下标 | 原始t，秒 | api_time，秒 | 原始think_time，秒 |
|---:|---:|---:|---:|
| 0 | 0 | 11.568 | 未给出 |
| 2 | 145004.210 | 29.674 | 144875.644 |
| 3 | 145034.409 | 40.138 | 0.525 |
| 4 | 145075.511 | 5.604 | 0.964 |

[原始数据](../../datasets/agentx_20261030/cc-traces-weka-062126-256k/traces.jsonl:340)本身含约40小时的间隔，不是回放把毫秒当作秒。官方loader将秒乘1000转换为timestamp/delay毫秒，导出保留这些字段，当前回放再除1000转回秒。

所选该家族只有一个主请求、26个嵌套子代理请求及3个后续flat请求。长间隔主要位于后续flat分支的开始处，而不是普通相邻turn的delay，因此只检查turn.delay的最大值会漏掉它。

## 实际跳过多少时间

当前规则只在全局没有在途请求时，将下一次事件之前超过10秒的空闲压缩至10秒。记录了每轮`idle_compressed_s`，未删除请求或改写输入／输出长度。

- EP4 C8三轮跳过144606.858、144613.495、144613.625秒。
- PP2 C8三轮跳过144598.180、144607.484、144608.185秒。
- EP4 C128三轮跳过143973.080、143899.483、143969.759秒。

因此表中的数百秒整轮／根会话完成时间是**压缩回放的墙钟**。达标请求/s也以此为分母，既不是原始40小时会话的生产吞吐，也不是饱和服务容量。全局空闲出现时机随布局和负载变化，跳过量也随之变化；请求延迟、SLO网格和压缩回放完成时间必须一起看。`wall + skipped`只是回放虚拟时钟推进量，不等于实际未压缩重跑。

## 与官方分支运行时的差别

当前`agentx_replay_v3.py`对post分支加入“父请求完成＋子首请求的源timestamp减父源结束timestamp”的条件。官方loader构造的节点、源长度和join/predecessor关系得到保留，但运行时触发规则不是直接复用官方实现。

固定官方harness提交`e4bb2f835ffedbbe6b76252ea5c93caec68c5faf`的`branch_orchestrator.py`在父返回路径中，用**子首请求相对branch start**的偏移调度子分支；此外，它会对源记录中与父请求重叠的SPAWN，在父发出时启动分支。当前自定义回放采用post完成条件。

例如上述flat分支，当前两个子会话的post延迟分别为144992.642秒、145022.841秒，官方父返回路径的相对偏移为0秒、30.199秒。所选8个家族共10条子会话触发关系，其中一条源记录与父结束重叠32毫秒，当前回放将其串行到父结束。其余不同偏移包含0.455、0.675、0.977、7.712、14.215、59.868秒等。这个差别影响请求重叠与batch，不能声称完全复现官方调度。

源码：[当前触发](agentx_replay_v3.py:215)；[官方父返回路径](../../datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8/src/aiperf/timing/branch_orchestrator.py:856)；[官方相对偏移](../../datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8/src/aiperf/timing/branch_orchestrator.py:959)；[官方源时间重叠处理](../../datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8/src/aiperf/timing/branch_orchestrator.py:447)；[loader毫秒转换](../../datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8/src/aiperf/dataset/loader/weka_trace.py:2114)。

## 如何使用当前结果

当前矩阵仍能比较相同自定义负载、同GPU预算、同缓存策略下的布局；完整原值保留。结论限定于该协议，不能仅凭压缩整轮耗时就认定生产AgentX中动态分配有／无必要，也不能把无已就绪请求的时段全算成真实工具耗时。

若要据此提出官方AgentX或真实会话层面的强结论，须在明确复用官方branch dispatch／trajectory sampling的独立协议下做关键对照。该验证目前尚未完成；后续协议与本矩阵分开保存，不能改写已测数据来合并排名。

## 独立 V4 时间协议准备（未用于 GPU 结果）

2026-10-31 新增 `agentx_branch_timing.py` 与 `agentx_replay_v4.py`，保留全部 V3 源码和运行不动。V4 令根会话中源时间重叠的 SPAWN 在父请求获得发出资格时启动子分支；其余 SPAWN 按官方 branch-start 相对偏移、父返回路径触发，FORK 仍等待返回，保留后续 turn 延迟与 join 依赖。

[9项CPU检查](branch_timing_tests_v2.log)执行固定官方源码中的三个分支方法 AST 与两个时间 helper，核对全部24个导出候选图；真实线程测试检查父完成前子请求已经发出、分支仅发出一次及 join 等待。最初完整 harness 导入缺少 `orjson` 的失败日志保留，并未安装新依赖。V4显式拒绝pre-session分支，因为有限根会话准入不等价于官方每阶段一次的后台分支调度。

这是针对已定位分支规则的准备与CPU验证，尚无V4 GPU测量，也不实现官方轨迹抽样、chat渲染或评分。E6压力修复运行仍冻结V3回放；今后V4结果需单独建组，不能静默拼进当前均值。

2026-10-31 09:50更新：后续三个尚未启动的V3控制已取消，改为四组同源V4控制，共24轮C128，加入8个单卡副本基线。当前E7 V3运行只承担缓存压力资格验证。`prepare_cluster_v20.py`已冻结分支helper、官方分支源码hash、7200秒整轮上限与失败捕获；网络、token构造和路由八个函数／类的AST与V3一致。四组的回放、helper、runner、捕获代码及来源图hash相同，前三组启动配置与对应V3准备完全一致（运行ID／端口除外）。

[16项检查](v4_bundle_tests_native_ast.log)针对实际冻结副本，覆盖9项分支测试、4项失败捕获及3项前缀／路由／环检测。官方包初始化在两种本地Python环境中分别缺依赖或插件注册失败；未安装依赖或改变服务环境。验证器直接执行官方三个方法、两个时间helper及原始枚举／基类AST，枚举源码hash也保存。这里验证的是这些明确规则，不是完整harness等价性。[检查证明](runs/eight_gpu_ep4_queue32_e7_v4_b7200_r1/v4_cpu_validation.json) · [四组一致性校验](e7_v4_cohort_preflight.json) · [当前计划](E7_V4_RUN_PLAN.json)
