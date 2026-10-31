# GLM/H100：EP8 与 PP2×EP4 的差距来自哪里

## 结论

PP 将每层 Expert 通信保留在单节点，跨节点只传阶段边界；纯通信实测确认这条路径便宜得多。真实层 profile 进一步检查计算与暴露的等待，但局部占比不能直接解释端到端加速的百分比。

## 历史端到端结果（同 8 张 H100）

| workload | TP8 | EP8 | PP2×EP4 | 单位 |
|---|---:|---:|---:|---|
|prefill_c4|4248.41|3706.71|12336.44|输入 tok/s|
|prefill_c16|4297.91|3840.99|12480.09|输入 tok/s|
|decode_c4|168.83|130.60|166.37|输出 tok/s|
|decode_c16|530.78|429.39|518.09|输出 tok/s|

Prefill：输入8192/输出1；Decode：输入128/输出512；每格3轮，C4为12个、C16为48个正式完成请求。三配置各120个正式请求全部完成。历史数字重用，未重新进行完整性能测试。

Attention：TP8/EP8 都是 TP4×DP2；PP 每个阶段 TP4。Expert：TP8为160个宽192切片，EP8为20个完整专家，PP为每卡40个完整专家，完整专家宽1536。三者都使用 private Humming，但 TP indexed 与 EP/PP grouped-masked 计算布局仍不同。

## 新增真实 Prefill 层样本

C4，8192→1；全局chunk1024，EP每个Attention组Q512，PP阶段Q1024；采样prefix2048，层20/50，Eager。每配置只取一轮plain和一轮observed；不是服务尾延迟实验。

|层|配置|rank / forward|T|layer ms|Attention ms|routed Expert ms|DeepEP ms|TP通信 ms|
|---|---|---|---:|---:|---:|---:|---:|---:|
|20|ep8|2 / 80|512|2.9355|0.2870|0.2052|2.1796|0.1422|
|20|pp2ep4|3 / 77|1024|2.4490|0.4723|0.4708|0.9756|0.4008|
|50|ep8|6 / 81|512|3.1435|0.2886|0.2299|2.4048|0.0975|
|50|pp2ep4|5 / 77|1024|2.8156|0.7705|0.7323|0.3221|0.1032|

本轮plain输入吞吐 EP8=3900.27、PP=12109.71 tok/s；observed/plain整批wall比率 EP=1.030、PP=1.007。只各一轮，不代表局部插桩无扰动。

各行只使用同一rank/forward的exclusive分区，其他router/shared/gap见图及CSV。不同rank的捕获不是同一个global tick，不能拼接成全局关键路径；全部rank数据保留。

## 纯通信验证

|1024逻辑token，Eager|节点174 EP4|节点175 EP4|双节点EP8|
|---|---:|---:|---:|
|dispatch+combine|0.1874 ms|0.1932 ms|1.8847 ms|
|RS4→dispatch/combine→AG4|0.3116 ms|0.3085 ms|2.4634 ms|

PP activation GPU边界T1024中位数 0.9386 ms：两次源AG4→四对GPU分片P2P→两次目标AG4。总跨节点payload20MiB，排除CPU元数据和调度。

EP8 Prefill完整通信链三次样本中位数2.4634ms、均值3.4512ms、范围2.2737–5.6166ms；异常样本未丢弃。Decode Graph通信结果及所有原始三次样本保存在pure_communication；未用本轮Prefill profile推断Decode计算占比。

## 证据与限制

- EP真实profile来自r3独立完成并获接受的EP arm；r3整体Guard失败仍明确保留。PP来自r4独立成功Guard，不能声称两arm来自一次共同通过的运行。
- deployed EP使用custom DeepEP，deployed PP使用installed DeepEP；纯通信测试已统一custom库，但不能消除真实部署里的这一差异。
- 独立通信中位数不能相加；PP边界不加入真实层堆叠，避免重复或遗漏计时边界。
- profile只测Prefill层；不把局部比例乘89层去解释完整模型差距，不推断PP已达最佳并行度或批次。

![独立rank层分区与纯通信](attribution.png)

原始历史、通信、真实rank profile、配置、冻结源哈希及Guard证据均保留或指向原始canonical/runtime路径。
