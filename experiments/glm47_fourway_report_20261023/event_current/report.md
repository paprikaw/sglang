# GLM-4.7-AWQ four-way：生产性能与独立 CUDA Event 诊断

065／167，8×A100。B8，P4096，原生 CPU4032／GPU0，Q1→64；首次 forward 为65 token。生产计时包含 HiCache→GPU，排除权重加载、源 Bank 导入与诊断。

## 未加仪器的生产结果（原 v5 两轮，未重测）

两轮均值 ± 半极差，不是置信区间。

| 配置 | 输出 tok/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|
| ep8_mtp_on | 96.502 ± 0.838 | 553.127 ± 3.627 | 69.721 ± 0.757 | 4945.539 ± 44.086 |
| ep8_mtp_off | 84.939 ± 0.137 | 533.294 ± 5.554 | 87.213 ± 0.066 | 6027.734 ± 9.701 |
| pp2ep4_mtp_on | 88.284 ± 0.056 | 501.549 ± 0.139 | 77.182 ± 0.050 | 5364.026 ± 3.310 |
| pp2ep4_mtp_off | 83.527 ± 0.136 | 496.253 ± 1.311 | 89.057 ± 0.137 | 6106.836 ± 9.935 |

同配置两轮 exact token/text parity 通过；不同配置完整输出历史并不相同，不能解释为等 token 历史的纯通信 overhead 对照。

| 跨配置 | Timed 全输出相同 / 8 | Repeat 全输出相同 / 8 | 同语义 Bank |
|---|---:|---:|---|
| ep8_mtp_on / ep8_mtp_off | 2 | 2 | True |
| ep8_mtp_on / pp2ep4_mtp_on | 2 | 2 | True |
| ep8_mtp_on / pp2ep4_mtp_off | 3 | 3 | True |
| ep8_mtp_off / pp2ep4_mtp_on | 0 | 0 | True |
| ep8_mtp_off / pp2ep4_mtp_off | 2 | 2 | True |
| pp2ep4_mtp_on / pp2ep4_mtp_off | 2 | 2 | True |

## CUDA Event 诊断

状态：complete；有效 profile：4/4。缺失数据均为 null（表中 —）。

以下是每个 rank 的实际 Graph 平均绝对 stream span，单位 ms。不是 CUPTI kernel busy，也不是 TPOT 分解。Draft ordinal 仅标记最近 target，不能把 target+draft 均值相加当成完整 MTP cycle。

### 紧凑分组汇总

EP 对八 rank 求均值；PP 分别对每 stage 四 rank 求均值。均为单次实际 Graph-role 平均 span，不能跨 rank/stage 相加。target 的实际 MoE 层数为 EP89、PP stage0 43／stage1 46；draft 为1。

| 配置 | Rank组 | Role | Graph ms | A | RoutedE | SharedE | Dense | Dispatch+Combine | D+C / MoE层 ms | NCCL集体通信 | Other | Overlap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ep8_mtp_on | ranks0_7 | target | 163.0798 | 78.2681 | 9.1405 | 15.3882 | 0.2384 | 26.7215 | 0.3002 | 7.2067 | 26.1163 | 0.0000 |
| ep8_mtp_on | ranks0_7 | draft | 2.6873 | 0.3709 | 0.1046 | 0.1726 | 0.0000 | 0.3228 | 0.3228 | 0.8009 | 0.9155 | 0.0000 |
| ep8_mtp_off | ranks0_7 | target | 89.9074 | 15.6454 | 7.0348 | 15.4944 | 0.2365 | 19.8154 | 0.2226 | 5.7902 | 25.8907 | 0.0000 |
| ep8_mtp_off | ranks0_7 | draft | — | — | — | — | — | — | — | — | — | — |
| pp2ep4_mtp_on | ranks0_3 | target | 76.8292 | 39.1450 | 5.1070 | 7.2435 | 0.2908 | 8.7442 | 0.2034 | 3.5032 | 12.5149 | 0.2807 |
| pp2ep4_mtp_on | ranks0_3 | draft | — | — | — | — | — | — | — | — | — | — |
| pp2ep4_mtp_on | ranks4_7 | target | 79.3072 | 39.1588 | 5.3207 | 8.1591 | 0.0000 | 9.3803 | 0.2039 | 3.4932 | 13.7951 | 0.0000 |
| pp2ep4_mtp_on | ranks4_7 | draft | 2.3115 | 0.3712 | 0.1172 | 0.1751 | 0.0000 | 0.1762 | 0.1762 | 0.3720 | 1.0998 | 0.0000 |
| pp2ep4_mtp_off | ranks0_3 | target | 40.7895 | 7.7836 | 3.6841 | 7.4639 | 0.2887 | 6.5502 | 0.1523 | 2.5040 | 12.4016 | 0.1134 |
| pp2ep4_mtp_off | ranks0_3 | draft | — | — | — | — | — | — | — | — | — | — |
| pp2ep4_mtp_off | ranks4_7 | target | 43.4636 | 7.8125 | 3.9741 | 8.3415 | 0.0000 | 7.0228 | 0.1527 | 2.6670 | 13.6456 | 0.0000 |
| pp2ep4_mtp_off | ranks4_7 | draft | — | — | — | — | — | — | — | — | — | — |

D+C 归一化是部署配置的 span 对照；两组 peer 数、路由域与 request-group 组合同时变化，不能据此只归因于节点内／节点间链路。12 个 target 是局部 scheduler forwards：EP 每个 DP 组12次，PP 两个 B4 slot 各6次，不是每请求12次。

### ep8_mtp_on

状态：valid_external_event_profile。

| Rank | Role / phase | Graphs | Graph span | Attention | RoutedE | SharedE | Dense | Dispatch | Combine | NCCL集体通信 | Other | Overlap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | target | 12 | 163.1513 | 78.4763 | 8.9967 | 15.0173 | 0.2389 | 13.3545 | 13.5990 | 7.2786 | 26.1900 | 0.0000 |
| 0 | draft | 24 | 2.6834 | 0.3700 | 0.0994 | 0.1666 | 0.0000 | 0.2215 | 0.1347 | 0.7973 | 0.8939 | 0.0000 |
| 0 | draft | 12 | 2.1705 | 0.1683 | 0.0897 | 0.1669 | 0.0000 | 0.1221 | 0.1167 | 0.5972 | 0.9097 | 0.0000 |
| 0 | draft_extend | 12 | 3.1963 | 0.5717 | 0.1091 | 0.1662 | 0.0000 | 0.3209 | 0.1527 | 0.9975 | 0.8782 | 0.0000 |
| 0 | target_verify | 12 | 163.1513 | 78.4763 | 8.9967 | 15.0173 | 0.2389 | 13.3545 | 13.5990 | 7.2786 | 26.1900 | 0.0000 |
| 1 | target | 12 | 163.1299 | 78.1119 | 9.0982 | 15.4232 | 0.2377 | 13.3278 | 13.4759 | 7.3588 | 26.0962 | 0.0000 |
| 1 | draft | 24 | 2.8242 | 0.3714 | 0.1137 | 0.1769 | 0.0000 | 0.2097 | 0.1228 | 0.9007 | 0.9289 | 0.0000 |
| 1 | draft | 12 | 2.1494 | 0.1683 | 0.0950 | 0.1791 | 0.0000 | 0.1091 | 0.1128 | 0.5356 | 0.9494 | 0.0000 |
| 1 | draft_extend | 12 | 3.4990 | 0.5745 | 0.1324 | 0.1747 | 0.0000 | 0.3104 | 0.1328 | 1.2658 | 0.9084 | 0.0000 |
| 1 | target_verify | 12 | 163.1299 | 78.1119 | 9.0982 | 15.4232 | 0.2377 | 13.3278 | 13.4759 | 7.3588 | 26.0962 | 0.0000 |
| 2 | target | 12 | 163.0734 | 78.3338 | 9.2281 | 14.8696 | 0.2383 | 13.8322 | 13.2537 | 7.2744 | 26.0432 | 0.0000 |
| 2 | draft | 24 | 2.7194 | 0.3716 | 0.0955 | 0.1693 | 0.0000 | 0.2194 | 0.1373 | 0.8026 | 0.9236 | 0.0000 |
| 2 | draft | 12 | 2.0719 | 0.1689 | 0.0838 | 0.1702 | 0.0000 | 0.1205 | 0.1201 | 0.4669 | 0.9416 | 0.0000 |
| 2 | draft_extend | 12 | 3.3668 | 0.5744 | 0.1073 | 0.1684 | 0.0000 | 0.3183 | 0.1544 | 1.1383 | 0.9057 | 0.0000 |
| 2 | target_verify | 12 | 163.0734 | 78.3338 | 9.2281 | 14.8696 | 0.2383 | 13.8322 | 13.2537 | 7.2744 | 26.0432 | 0.0000 |
| 3 | target | 12 | 163.0282 | 78.1950 | 9.1640 | 14.8335 | 0.2375 | 14.0250 | 13.4705 | 7.0957 | 26.0070 | 0.0000 |
| 3 | draft | 24 | 2.7116 | 0.3702 | 0.0989 | 0.1649 | 0.0000 | 0.2250 | 0.1338 | 0.8238 | 0.8948 | 0.0000 |
| 3 | draft | 12 | 2.0730 | 0.1679 | 0.0870 | 0.1645 | 0.0000 | 0.1282 | 0.1161 | 0.5027 | 0.9066 | 0.0000 |
| 3 | draft_extend | 12 | 3.3501 | 0.5725 | 0.1108 | 0.1653 | 0.0000 | 0.3219 | 0.1516 | 1.1448 | 0.8831 | 0.0000 |
| 3 | target_verify | 12 | 163.0282 | 78.1950 | 9.1640 | 14.8335 | 0.2375 | 14.0250 | 13.4705 | 7.0957 | 26.0070 | 0.0000 |
| 4 | target | 12 | 163.1001 | 78.1890 | 8.9378 | 15.4249 | 0.2374 | 13.1507 | 13.6823 | 7.3329 | 26.1451 | 0.0000 |
| 4 | draft | 24 | 2.5774 | 0.3699 | 0.1055 | 0.1705 | 0.0000 | 0.1709 | 0.1301 | 0.7237 | 0.9068 | 0.0000 |
| 4 | draft | 12 | 2.1478 | 0.1681 | 0.0928 | 0.1698 | 0.0000 | 0.1198 | 0.1092 | 0.5703 | 0.9177 | 0.0000 |
| 4 | draft_extend | 12 | 3.0070 | 0.5717 | 0.1182 | 0.1712 | 0.0000 | 0.2220 | 0.1510 | 0.8771 | 0.8958 | 0.0000 |
| 4 | target_verify | 12 | 163.1001 | 78.1890 | 8.9378 | 15.4249 | 0.2374 | 13.1507 | 13.6823 | 7.3329 | 26.1451 | 0.0000 |
| 5 | target | 12 | 163.0730 | 78.2978 | 9.0745 | 15.9361 | 0.2394 | 12.8000 | 13.5673 | 7.0520 | 26.1059 | 0.0000 |
| 5 | draft | 24 | 2.6682 | 0.3729 | 0.1032 | 0.1831 | 0.0000 | 0.1622 | 0.1320 | 0.7820 | 0.9329 | 0.0000 |
| 5 | draft | 12 | 2.1227 | 0.1690 | 0.0911 | 0.1864 | 0.0000 | 0.1073 | 0.1104 | 0.5030 | 0.9553 | 0.0000 |
| 5 | draft_extend | 12 | 3.2138 | 0.5767 | 0.1152 | 0.1799 | 0.0000 | 0.2170 | 0.1536 | 1.0610 | 0.9104 | 0.0000 |
| 5 | target_verify | 12 | 163.0730 | 78.2978 | 9.0745 | 15.9361 | 0.2394 | 12.8000 | 13.5673 | 7.0520 | 26.1059 | 0.0000 |
| 6 | target | 12 | 163.0112 | 78.1638 | 9.2777 | 15.3730 | 0.2390 | 13.3400 | 13.1767 | 7.3408 | 26.1003 | 0.0000 |
| 6 | draft | 24 | 2.6366 | 0.3702 | 0.0909 | 0.1700 | 0.0000 | 0.1721 | 0.1416 | 0.7793 | 0.9126 | 0.0000 |
| 6 | draft | 12 | 2.0579 | 0.1684 | 0.0881 | 0.1696 | 0.0000 | 0.1248 | 0.1106 | 0.4665 | 0.9298 | 0.0000 |
| 6 | draft_extend | 12 | 3.2154 | 0.5720 | 0.0936 | 0.1703 | 0.0000 | 0.2194 | 0.1725 | 1.0920 | 0.8955 | 0.0000 |
| 6 | target_verify | 12 | 163.0112 | 78.1638 | 9.2777 | 15.3730 | 0.2390 | 13.3400 | 13.1767 | 7.3408 | 26.1003 | 0.0000 |
| 7 | target | 12 | 163.0714 | 78.3774 | 9.3470 | 16.2283 | 0.2393 | 12.4976 | 13.2189 | 6.9202 | 26.2428 | 0.0000 |
| 7 | draft | 24 | 2.6774 | 0.3712 | 0.1294 | 0.1794 | 0.0000 | 0.1648 | 0.1045 | 0.7980 | 0.9301 | 0.0000 |
| 7 | draft | 12 | 2.0797 | 0.1690 | 0.1091 | 0.1799 | 0.0000 | 0.1166 | 0.0923 | 0.4704 | 0.9423 | 0.0000 |
| 7 | draft_extend | 12 | 3.2752 | 0.5734 | 0.1497 | 0.1789 | 0.0000 | 0.2130 | 0.1167 | 1.1256 | 0.9179 | 0.0000 |
| 7 | target_verify | 12 | 163.0714 | 78.3774 | 9.3470 | 16.2283 | 0.2393 | 12.4976 | 13.2189 | 6.9202 | 26.2428 | 0.0000 |

| Rank | PP boundary API calls | API stream span mean ms |
|---:|---:|---:|
| 0 | 0 | — |
| 1 | 0 | — |
| 2 | 0 | — |
| 3 | 0 | — |
| 4 | 0 | — |
| 5 | 0 | — |
| 6 | 0 | — |
| 7 | 0 | — |

PP boundary 是 current-stream API dependency/enqueue span；不表示异步发送完成或纯 NIC 传输延迟。

| 仪器影响（不是性能新结果） | Profile | 未加仪器两轮均值 | 倍数 |
|---|---:|---:|---:|
| wall_s | 6.2844 | 5.3060 | 1.1844 |
| ttft_ms | 535.3069 | 553.1268 | 0.9678 |
| tpot_ms | 78.9875 | 69.7208 | 1.1329 |
| end_to_end_ms | 5511.5204 | 4945.5387 | 1.1144 |

### ep8_mtp_off

状态：valid_external_event_profile。

| Rank | Role / phase | Graphs | Graph span | Attention | RoutedE | SharedE | Dense | Dispatch | Combine | NCCL集体通信 | Other | Overlap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | target | 12 | 89.9644 | 15.6004 | 6.9065 | 15.1205 | 0.2357 | 9.7079 | 10.5539 | 5.9598 | 25.8798 | 0.0000 |
| 0 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 0 | target_decode | 12 | 89.9644 | 15.6004 | 6.9065 | 15.1205 | 0.2357 | 9.7079 | 10.5539 | 5.9598 | 25.8798 | 0.0000 |
| 1 | target | 12 | 89.9362 | 15.6428 | 6.9570 | 15.5210 | 0.2369 | 9.5418 | 10.4748 | 5.7109 | 25.8510 | 0.0000 |
| 1 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 1 | target_decode | 12 | 89.9362 | 15.6428 | 6.9570 | 15.5210 | 0.2369 | 9.5418 | 10.4748 | 5.7109 | 25.8510 | 0.0000 |
| 2 | target | 12 | 89.9522 | 15.6918 | 7.2487 | 14.9710 | 0.2365 | 9.8798 | 10.1647 | 5.9609 | 25.7988 | 0.0000 |
| 2 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 2 | target_decode | 12 | 89.9522 | 15.6918 | 7.2487 | 14.9710 | 0.2365 | 9.8798 | 10.1647 | 5.9609 | 25.7988 | 0.0000 |
| 3 | target | 12 | 89.9494 | 15.6437 | 7.1193 | 14.9338 | 0.2366 | 10.1688 | 10.3721 | 5.7363 | 25.7389 | 0.0000 |
| 3 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 3 | target_decode | 12 | 89.9494 | 15.6437 | 7.1193 | 14.9338 | 0.2366 | 10.1688 | 10.3721 | 5.7363 | 25.7389 | 0.0000 |
| 4 | target | 12 | 89.9181 | 15.6084 | 6.8372 | 15.5127 | 0.2356 | 9.2735 | 10.5025 | 5.9631 | 25.9852 | 0.0000 |
| 4 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 4 | target_decode | 12 | 89.9181 | 15.6084 | 6.8372 | 15.5127 | 0.2356 | 9.2735 | 10.5025 | 5.9631 | 25.9852 | 0.0000 |
| 5 | target | 12 | 89.8957 | 15.6399 | 6.9239 | 16.0499 | 0.2370 | 9.0001 | 10.4129 | 5.7295 | 25.9025 | 0.0000 |
| 5 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 5 | target_decode | 12 | 89.8957 | 15.6399 | 6.9239 | 16.0499 | 0.2370 | 9.0001 | 10.4129 | 5.7295 | 25.9025 | 0.0000 |
| 6 | target | 12 | 89.8342 | 15.6527 | 7.1796 | 15.5119 | 0.2364 | 9.4668 | 10.1246 | 5.7261 | 25.9361 | 0.0000 |
| 6 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 6 | target_decode | 12 | 89.8342 | 15.6527 | 7.1796 | 15.5119 | 0.2364 | 9.4668 | 10.1246 | 5.7261 | 25.9361 | 0.0000 |
| 7 | target | 12 | 89.8094 | 15.6838 | 7.1064 | 16.3347 | 0.2371 | 8.7546 | 10.1242 | 5.5353 | 26.0332 | 0.0000 |
| 7 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 7 | target_decode | 12 | 89.8094 | 15.6838 | 7.1064 | 16.3347 | 0.2371 | 8.7546 | 10.1242 | 5.5353 | 26.0332 | 0.0000 |

| Rank | PP boundary API calls | API stream span mean ms |
|---:|---:|---:|
| 0 | 0 | — |
| 1 | 0 | — |
| 2 | 0 | — |
| 3 | 0 | — |
| 4 | 0 | — |
| 5 | 0 | — |
| 6 | 0 | — |
| 7 | 0 | — |

PP boundary 是 current-stream API dependency/enqueue span；不表示异步发送完成或纯 NIC 传输延迟。

| 仪器影响（不是性能新结果） | Profile | 未加仪器两轮均值 | 倍数 |
|---|---:|---:|---:|
| wall_s | 6.4607 | 6.0279 | 1.0718 |
| ttft_ms | 499.4119 | 533.2938 | 0.9365 |
| tpot_ms | 94.6213 | 87.2133 | 1.0849 |
| end_to_end_ms | 6460.5548 | 6027.7341 | 1.0718 |

### pp2ep4_mtp_on

状态：valid_external_event_profile。

| Rank | Role / phase | Graphs | Graph span | Attention | RoutedE | SharedE | Dense | Dispatch | Combine | NCCL集体通信 | Other | Overlap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | target | 12 | 76.9044 | 38.9989 | 5.0694 | 7.2452 | 0.2901 | 4.4569 | 4.2161 | 3.7415 | 12.6065 | 0.2798 |
| 0 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 0 | target_verify | 12 | 76.9044 | 38.9989 | 5.0694 | 7.2452 | 0.2901 | 4.4569 | 4.2161 | 3.7415 | 12.6065 | 0.2798 |
| 1 | target | 12 | 76.9028 | 39.4480 | 5.2797 | 7.4050 | 0.2924 | 4.5011 | 4.0419 | 3.1420 | 12.5128 | 0.2801 |
| 1 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 1 | target_verify | 12 | 76.9028 | 39.4480 | 5.2797 | 7.4050 | 0.2924 | 4.5011 | 4.0419 | 3.1420 | 12.5128 | 0.2801 |
| 2 | target | 12 | 76.7179 | 39.0996 | 4.9249 | 7.1680 | 0.2909 | 4.5716 | 4.3366 | 3.5673 | 12.4768 | 0.2821 |
| 2 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 2 | target_verify | 12 | 76.7179 | 39.0996 | 4.9249 | 7.1680 | 0.2909 | 4.5716 | 4.3366 | 3.5673 | 12.4768 | 0.2821 |
| 3 | target | 12 | 76.7916 | 39.0333 | 5.1540 | 7.1556 | 0.2897 | 4.6772 | 4.1755 | 3.5622 | 12.4634 | 0.2807 |
| 3 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 3 | target_verify | 12 | 76.7916 | 39.0333 | 5.1540 | 7.1556 | 0.2897 | 4.6772 | 4.1755 | 3.5622 | 12.4634 | 0.2807 |
| 4 | target | 12 | 79.3454 | 39.3737 | 5.1591 | 7.9939 | 0.0000 | 4.9393 | 4.7468 | 3.2947 | 13.8378 | 0.0000 |
| 4 | draft | 24 | 2.2882 | 0.3696 | 0.1227 | 0.1706 | 0.0000 | 0.0948 | 0.0779 | 0.3728 | 1.0798 | 0.0000 |
| 4 | draft | 12 | 1.9287 | 0.1683 | 0.0982 | 0.1702 | 0.0000 | 0.0795 | 0.0708 | 0.2558 | 1.0859 | 0.0000 |
| 4 | draft_extend | 12 | 2.6477 | 0.5709 | 0.1471 | 0.1711 | 0.0000 | 0.1101 | 0.0850 | 0.4898 | 1.0737 | 0.0000 |
| 4 | target_verify | 12 | 79.3454 | 39.3737 | 5.1591 | 7.9939 | 0.0000 | 4.9393 | 4.7468 | 3.2947 | 13.8378 | 0.0000 |
| 5 | target | 12 | 79.3198 | 39.4599 | 5.2131 | 8.2451 | 0.0000 | 4.8799 | 4.6829 | 3.1090 | 13.7300 | 0.0000 |
| 5 | draft | 24 | 2.3460 | 0.3724 | 0.1117 | 0.1775 | 0.0000 | 0.0883 | 0.0896 | 0.4176 | 1.0889 | 0.0000 |
| 5 | draft | 12 | 1.9336 | 0.1687 | 0.0884 | 0.1782 | 0.0000 | 0.0709 | 0.0826 | 0.2452 | 1.0996 | 0.0000 |
| 5 | draft_extend | 12 | 2.7585 | 0.5760 | 0.1351 | 0.1768 | 0.0000 | 0.1057 | 0.0965 | 0.5901 | 1.0783 | 0.0000 |
| 5 | target_verify | 12 | 79.3198 | 39.4599 | 5.2131 | 8.2451 | 0.0000 | 4.8799 | 4.6829 | 3.1090 | 13.7300 | 0.0000 |
| 6 | target | 12 | 79.3152 | 38.7682 | 5.4433 | 7.9958 | 0.0000 | 4.8736 | 4.3819 | 4.0357 | 13.8167 | 0.0000 |
| 6 | draft | 24 | 2.3005 | 0.3698 | 0.1161 | 0.1717 | 0.0000 | 0.0942 | 0.0871 | 0.3673 | 1.0943 | 0.0000 |
| 6 | draft | 12 | 1.9216 | 0.1687 | 0.0989 | 0.1717 | 0.0000 | 0.0782 | 0.0710 | 0.2372 | 1.0959 | 0.0000 |
| 6 | draft_extend | 12 | 2.6794 | 0.5709 | 0.1332 | 0.1718 | 0.0000 | 0.1103 | 0.1032 | 0.4974 | 1.0927 | 0.0000 |
| 6 | target_verify | 12 | 79.3152 | 38.7682 | 5.4433 | 7.9958 | 0.0000 | 4.8736 | 4.3819 | 4.0357 | 13.8167 | 0.0000 |
| 7 | target | 12 | 79.2484 | 39.0332 | 5.4672 | 8.4017 | 0.0000 | 4.6386 | 4.3781 | 3.5337 | 13.7958 | 0.0000 |
| 7 | draft | 24 | 2.3113 | 0.3730 | 0.1184 | 0.1805 | 0.0000 | 0.0883 | 0.0845 | 0.3304 | 1.1362 | 0.0000 |
| 7 | draft | 12 | 1.9192 | 0.1691 | 0.0938 | 0.1799 | 0.0000 | 0.0722 | 0.0748 | 0.1970 | 1.1325 | 0.0000 |
| 7 | draft_extend | 12 | 2.7034 | 0.5769 | 0.1430 | 0.1812 | 0.0000 | 0.1044 | 0.0942 | 0.4637 | 1.1399 | 0.0000 |
| 7 | target_verify | 12 | 79.2484 | 39.0332 | 5.4672 | 8.4017 | 0.0000 | 4.6386 | 4.3781 | 3.5337 | 13.7958 | 0.0000 |

| Rank | PP boundary API calls | API stream span mean ms |
|---:|---:|---:|
| 0 | 36 | 7.2811 |
| 1 | 36 | 7.3074 |
| 2 | 36 | 7.2527 |
| 3 | 36 | 7.2845 |
| 4 | 36 | 1.1724 |
| 5 | 36 | 1.1862 |
| 6 | 36 | 1.1879 |
| 7 | 36 | 1.1906 |

PP boundary 是 current-stream API dependency/enqueue span；不表示异步发送完成或纯 NIC 传输延迟。

| 仪器影响（不是性能新结果） | Profile | 未加仪器两轮均值 | 倍数 |
|---|---:|---:|---:|
| wall_s | 6.0344 | 5.7995 | 1.0405 |
| ttft_ms | 504.4229 | 501.5491 | 1.0057 |
| tpot_ms | 80.6200 | 77.1822 | 1.0445 |
| end_to_end_ms | 5583.4818 | 5364.0257 | 1.0409 |

### pp2ep4_mtp_off

状态：valid_external_event_profile。

| Rank | Role / phase | Graphs | Graph span | Attention | RoutedE | SharedE | Dense | Dispatch | Combine | NCCL集体通信 | Other | Overlap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | target | 12 | 40.8194 | 7.7765 | 3.7692 | 7.4487 | 0.2892 | 3.1756 | 3.2522 | 2.5265 | 12.4698 | 0.1118 |
| 0 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 0 | target_decode | 12 | 40.8194 | 7.7765 | 3.7692 | 7.4487 | 0.2892 | 3.1756 | 3.2522 | 2.5265 | 12.4698 | 0.1118 |
| 1 | target | 12 | 40.7805 | 7.7888 | 3.7767 | 7.6359 | 0.2891 | 3.0799 | 3.2618 | 2.4115 | 12.4215 | 0.1155 |
| 1 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 1 | target_decode | 12 | 40.7805 | 7.7888 | 3.7767 | 7.6359 | 0.2891 | 3.0799 | 3.2618 | 2.4115 | 12.4215 | 0.1155 |
| 2 | target | 12 | 40.7811 | 7.8006 | 3.5667 | 7.3972 | 0.2892 | 3.2482 | 3.3875 | 2.5847 | 12.3937 | 0.1132 |
| 2 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 2 | target_decode | 12 | 40.7811 | 7.8006 | 3.5667 | 7.3972 | 0.2892 | 3.2482 | 3.3875 | 2.5847 | 12.3937 | 0.1132 |
| 3 | target | 12 | 40.7770 | 7.7687 | 3.6240 | 7.3740 | 0.2871 | 3.4242 | 3.3714 | 2.4932 | 12.3214 | 0.1131 |
| 3 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 3 | target_decode | 12 | 40.7770 | 7.7687 | 3.6240 | 7.3740 | 0.2871 | 3.4242 | 3.3714 | 2.4932 | 12.3214 | 0.1131 |
| 4 | target | 12 | 43.4998 | 7.7954 | 3.8367 | 8.1504 | 0.0000 | 3.5843 | 3.7280 | 2.7624 | 13.6427 | 0.0000 |
| 4 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 4 | target_decode | 12 | 43.4998 | 7.7954 | 3.8367 | 8.1504 | 0.0000 | 3.5843 | 3.7280 | 2.7624 | 13.6427 | 0.0000 |
| 5 | target | 12 | 43.5086 | 7.8292 | 4.0149 | 8.4537 | 0.0000 | 3.4081 | 3.5212 | 2.6733 | 13.6081 | 0.0000 |
| 5 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 5 | target_decode | 12 | 43.5086 | 7.8292 | 4.0149 | 8.4537 | 0.0000 | 3.4081 | 3.5212 | 2.6733 | 13.6081 | 0.0000 |
| 6 | target | 12 | 43.4521 | 7.7914 | 3.9937 | 8.1603 | 0.0000 | 3.5387 | 3.5731 | 2.7414 | 13.6534 | 0.0000 |
| 6 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 6 | target_decode | 12 | 43.4521 | 7.7914 | 3.9937 | 8.1603 | 0.0000 | 3.5387 | 3.5731 | 2.7414 | 13.6534 | 0.0000 |
| 7 | target | 12 | 43.3939 | 7.8341 | 4.0513 | 8.6016 | 0.0000 | 3.2389 | 3.4988 | 2.4909 | 13.6783 | 0.0000 |
| 7 | draft | — | — | — | — | — | — | — | — | — | — | — |
| 7 | target_decode | 12 | 43.3939 | 7.8341 | 4.0513 | 8.6016 | 0.0000 | 3.2389 | 3.4988 | 2.4909 | 13.6783 | 0.0000 |

| Rank | PP boundary API calls | API stream span mean ms |
|---:|---:|---:|
| 0 | 36 | 5.3508 |
| 1 | 36 | 5.3593 |
| 2 | 36 | 5.3310 |
| 3 | 36 | 5.3493 |
| 4 | 36 | 0.4771 |
| 5 | 36 | 0.5061 |
| 6 | 36 | 0.4699 |
| 7 | 36 | 0.4507 |

PP boundary 是 current-stream API dependency/enqueue span；不表示异步发送完成或纯 NIC 传输延迟。

| 仪器影响（不是性能新结果） | Profile | 未加仪器两轮均值 | 倍数 |
|---|---:|---:|---:|
| wall_s | 6.5940 | 6.1298 | 1.0757 |
| ttft_ms | 491.2257 | 496.2529 | 0.9899 |
| tpot_ms | 96.4779 | 89.0569 | 1.0833 |
| end_to_end_ms | 6569.3325 | 6106.8363 | 1.0757 |

## Verify 与普通 Decode 的源码路径

- Pinned serving arguments choose Triton Attention. On A100 the gfx95-only verify split-KV/shared-KV gates are false; target Verify falls back to extend_attention_fwd.
- Ordinary MTP-off Q1 Decode uses forward_decode/decode_attention_fwd. MTP-on target Verify has Q3 per request and follows a different path.
- The different query width/kernel path supports a possible Attention-cost explanation, but does not by itself attribute the observed difference to the kernel: scopes include projections, normalization, cache operations and stream waits.

这是已固定源码支持的路径区别，不是对 Attention 差距的单因证明；源码 SHA 与位置保存在 summary.json。

## 机制资格与证据

Tiny external-event qualification：qualified（八 rank × 三次 Graph replay；只证明测量机制，不替代完整模型 profile）。

生产源：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3`

Profile 源：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v8/runs/event_profile_followup_r1`

summary.json 保存逐文件 SHA、源绑定、Graph/cache/cleanup/输出 parity gates 以及完整生产统计。

## 边界

- Measured per-device, per-actual-Graph external CUDA Event stream spans in milliseconds; includes stream waits and event nodes, not CUPTI kernel busy time.
- The one completion/read after every actual Graph changes diagnostic cadence. Instrumented wall/TPOT are shown separately from unprofiled production, never used to correct it.
- Target, draft and actual phases stay separate. A draft target_ordinal is the latest observed target, NOT proof of a complete same-request MTP cycle.
- Do not add target and draft means, ranks or PP stage maxima into request TPOT. PP boundary API stream spans are separate and do not measure asynchronous send/NIC completion.
- Attention covers the wrapped attention forward, including QKV/O projections, QK normalization, RoPE and KV store, but excludes the outer input RMSNorm. Expert totals include routed, shared and dense FFN scopes.
- The historical schema key nccl_tp is the wrapped GroupCoordinator collective API stream span. Raw data omit group ranks, so it cannot be asserted to be only Attention TP4 or split further into TP/DP groups.
- Other is uncovered Graph stream time, not GPU idle. Cross-category overlap is retained; nested NCCL collective API spans are removed from Attention by the audited interval partition.
- Twelve targets are local scheduler forwards, not twelve per request: each EP Attention DP group contributes twelve B4 targets; PP has two B4 request slots, six sampled targets per slot.
- Dispatch+Combine per MoE layer uses 89 EP target MoE layers or PP stage counts 43/46 (92 target layers, first three dense). This compares actual deployment spans; peer count, routing domain and request-group composition also change, so it is not a pure link-only causal comparison.
- Diagnostic committed tokens per actual target use bounded local event receipts; they are not the full 64-output production native accept_rate/accept_length denominators. Keep these populations separate.
- Missing, partial, mismatched or failed evidence produces null diagnostic data, never estimated component times.
