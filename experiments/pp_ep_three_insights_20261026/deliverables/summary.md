# PP / EP：三组 insight 与数据

新版PPT仅3页，每页一条insight、平均请求端到端延迟图、Expert/通信分项图。

1. GPU：122B Decode，A100 PP 的 E2E 延迟降低26.0%，H100 EP降低19.8%。
2. 模型：H100 Prefill，35B/122B 的 PP额外路由Expert模块活动66.4/117.7 ms/卡。
3. 阶段：H100 122B，Prefill PP延迟降低25.9%；Decode EP延迟降低19.8%。

GPU/阶段图采用实际路由Expert GEMM（每层2个GEMM，原始trace核验），单位GPU毫秒/卡/实际层调用；模型图采用整批Prefill路由Expert模块GPU活动。D+C为专家交换；PP NCCL为流水通信驻留。各部分用并排柱状图分别展示。正文仅保留数据所需的配置、单位和统计量；详细来源在备注、sources.json及profile_data。

数据来自已有正式计时及独立profile，没有启动新GPU实验。PPT文字可编辑，图表附PNG/矢量PDF及CSV。PDF为同一布局的页面预览；本环境无Office渲染器。原12页版本及原始实验资料保持不变。
