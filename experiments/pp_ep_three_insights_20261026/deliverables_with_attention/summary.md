# Attention 放置与 Expert / 通信权衡

共5页：部署对比 → Attention放置与共同计算 → GPU数据 → 模型数据 → 阶段数据。

第二页以4层、两组等量请求A/B示意：EP2使用Attention TP1/DP2，各卡保留全部Attention层、处理一组请求；PP2使用Attention TP1，各卡保留半数层，两组请求均经过两阶段。同一模型、token和上下文下，每个token的逐层Attention算子与总计算量相同，因而主要关注Expert本地执行效率与Dispatch/Combine专家交换。

算子与计算量的不变性不等同于GPU执行时间严格相等。批形状、缓存和调度的影响，以及阶段通信和流水填充/排空，保留在详细备注与原始端到端数据中。此说明不放在幻灯正文。

第一张部署图、三组数据表和六张数据图均核验保留；未启动新GPU实验。文字和两张示意图使用原生可编辑PPT形状。重建命令：build.py --with-attention --out <output_directory>。PDF和PNG来自同一布局。
