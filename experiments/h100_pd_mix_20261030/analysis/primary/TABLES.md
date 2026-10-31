# P/D 混合负载：完整原始轮次的算术平均

每节点分别比较；单位均为绝对毫秒。未通过客户端到达偏差检查的点不作为受控到达结论。原始请求、全部轮次和各自源目录保留。

## spartan-gpgpu011

| workload | 布局／路由 | PP policy | 平均 TTFT | 平均 TPOT | 平均 E2E | 完整流量完成时间 | 到达有效 |
|---|---|---:|---:|---:|---:|---:|---|
| balanced_rps16 | ep4/least_outstanding | 0 | 308.33 | 71.35 | 3272.80 | 6302.63 | True |
| balanced_rps16 | four_ep1/least_outstanding | 0 | 145.97 | 10.65 | 1567.96 | 5777.13 | True |
| balanced_rps16 | mixed/least_outstanding | 0 | 336.99 | 26.57 | 2560.01 | 7592.85 | True |
| balanced_rps16 | mixed/p_to_ep | 0 | 664.07 | 72.29 | 3368.45 | 8391.25 | True |
| balanced_rps16 | mixed/p_to_pp | 0 | 156.24 | 26.33 | 2471.40 | 6731.46 | True |
| balanced_rps16 | two_ep2/least_outstanding | 0 | 267.34 | 39.67 | 2797.60 | 6485.73 | True |
| balanced_rps16 | two_pp2/least_outstanding | 0 | 168.92 | 23.29 | 2486.88 | 7151.70 | True |
| balanced_rps32 | ep4/least_outstanding | 0 | 444.58 | 40.12 | 2579.28 | 4751.82 | True |
| balanced_rps32 | four_ep1/least_outstanding | 0 | 250.11 | 25.44 | 2048.80 | 4573.69 | True |
| balanced_rps32 | mixed/least_outstanding | 0 | 508.73 | 31.57 | 2686.09 | 6924.77 | True |
| balanced_rps32 | mixed/p_to_ep | 0 | 1178.51 | 55.12 | 3577.66 | 7443.66 | True |
| balanced_rps32 | mixed/p_to_pp | 0 | 327.48 | 74.48 | 3234.89 | 5052.05 | True |
| balanced_rps32 | two_ep2/least_outstanding | 0 | 349.14 | 37.38 | 2444.17 | 4659.31 | True |
| balanced_rps32 | two_pp2/least_outstanding | 0 | 360.60 | 24.64 | 2543.61 | 5798.35 | True |
| cold_b16_q8192 | ep4/least_outstanding | 0 | 720.38 | — | 720.38 | 1030.50 | True |
| cold_b16_q8192 | four_ep1/least_outstanding | 0 | 501.32 | — | 501.32 | 706.53 | True |
| cold_b16_q8192 | mixed/least_outstanding | 0 | 587.97 | — | 587.97 | 1029.40 | True |
| cold_b16_q8192 | two_ep2/least_outstanding | 0 | 687.48 | — | 687.48 | 1034.23 | True |
| cold_b16_q8192 | two_pp2/least_outstanding | 0 | 424.94 | — | 424.94 | 721.86 | True |
| cold_b32_q512 | ep4/least_outstanding | 0 | 219.11 | — | 219.11 | 223.47 | True |
| cold_b32_q512 | four_ep1/least_outstanding | 0 | 163.39 | — | 163.39 | 169.76 | True |
| cold_b32_q512 | mixed/least_outstanding | 0 | 189.82 | — | 189.82 | 229.86 | True |
| cold_b32_q512 | two_ep2/least_outstanding | 0 | 205.88 | — | 205.88 | 214.19 | True |
| cold_b32_q512 | two_pp2/least_outstanding | 0 | 146.06 | — | 146.06 | 176.43 | True |
| cold_b8_q16384 | ep4/least_outstanding | 0 | 834.77 | — | 834.77 | 1033.45 | True |
| cold_b8_q16384 | four_ep1/least_outstanding | 0 | 567.40 | — | 567.40 | 707.10 | True |
| cold_b8_q16384 | mixed/least_outstanding | 0 | 628.88 | — | 628.88 | 977.92 | True |
| cold_b8_q16384 | two_ep2/least_outstanding | 0 | 786.91 | — | 786.91 | 987.57 | True |
| cold_b8_q16384 | two_pp2/least_outstanding | 0 | 483.94 | — | 483.94 | 750.16 | True |
| dheavy_rps16 | ep4/least_outstanding | 0 | 273.85 | 41.47 | 3854.36 | 6494.27 | True |
| dheavy_rps16 | four_ep1/least_outstanding | 0 | 113.17 | 10.90 | 2244.42 | 6034.89 | True |
| dheavy_rps16 | mixed/least_outstanding | 0 | 511.03 | 20.60 | 3618.43 | 8264.61 | True |
| dheavy_rps16 | mixed/p_to_ep | 0 | 1820.01 | 14.46 | 4383.44 | 10982.69 | True |
| dheavy_rps16 | mixed/p_to_pp | 0 | 793.33 | 14.01 | 3695.42 | 8539.36 | True |
| dheavy_rps16 | two_ep2/least_outstanding | 0 | 217.82 | 25.40 | 3574.56 | 6661.44 | True |
| dheavy_rps16 | two_pp2/least_outstanding | 0 | 757.25 | 17.29 | 3742.36 | 8586.95 | True |
| dheavy_rps32 | ep4/least_outstanding | 0 | 299.18 | 25.72 | 2987.43 | 4633.24 | True |
| dheavy_rps32 | four_ep1/least_outstanding | 0 | 155.14 | 14.01 | 2639.24 | 4736.90 | True |
| dheavy_rps32 | mixed/least_outstanding | 0 | 825.54 | 22.33 | 3619.16 | 7767.14 | True |
| dheavy_rps32 | mixed/p_to_ep | 0 | 2074.97 | 25.12 | 4581.69 | 9729.09 | True |
| dheavy_rps32 | mixed/p_to_pp | 0 | 830.58 | 18.02 | 3242.68 | 6730.72 | True |
| dheavy_rps32 | two_ep2/least_outstanding | 0 | 281.69 | 25.70 | 3127.61 | 4859.45 | True |
| dheavy_rps32 | two_pp2/least_outstanding | 0 | 1072.79 | 18.01 | 3756.68 | 7479.93 | True |
| pheavy_rps16 | ep4/least_outstanding | 0 | 348.19 | 101.46 | 2753.32 | 6018.49 | True |
| pheavy_rps16 | four_ep1/least_outstanding | 0 | 190.27 | 9.93 | 875.40 | 5389.64 | True |
| pheavy_rps16 | mixed/least_outstanding | 0 | 215.94 | 47.45 | 1886.47 | 6643.81 | True |
| pheavy_rps16 | mixed/p_to_ep | 0 | 694.15 | 67.20 | 2439.84 | 6330.07 | True |
| pheavy_rps16 | mixed/p_to_pp | 0 | 364.73 | 83.40 | 2239.66 | 5794.65 | True |
| pheavy_rps16 | two_ep2/least_outstanding | 0 | 307.65 | 88.64 | 2538.76 | 6153.25 | True |
| pheavy_rps16 | two_pp2/least_outstanding | 0 | 123.15 | 29.93 | 1492.53 | 6333.27 | True |
| pheavy_rps32 | ep4/least_outstanding | 0 | 764.81 | 63.35 | 2484.50 | 5036.95 | True |
| pheavy_rps32 | four_ep1/least_outstanding | 0 | 385.85 | 49.83 | 1719.50 | 4003.97 | True |
| pheavy_rps32 | mixed/least_outstanding | 0 | 533.42 | 41.81 | 1988.35 | 5711.01 | True |
| pheavy_rps32 | mixed/p_to_ep | 0 | 1421.89 | 69.32 | 3240.01 | 5504.60 | True |
| pheavy_rps32 | mixed/p_to_pp | 0 | 1079.62 | 82.60 | 3006.99 | 4984.30 | True |
| pheavy_rps32 | two_ep2/least_outstanding | 0 | 624.94 | 59.44 | 2272.95 | 4840.17 | True |
| pheavy_rps32 | two_pp2/least_outstanding | 0 | 331.98 | 35.92 | 1735.80 | 5124.12 | True |

## spartan-gpgpu171

| workload | 布局／路由 | PP policy | 平均 TTFT | 平均 TPOT | 平均 E2E | 完整流量完成时间 | 到达有效 |
|---|---|---:|---:|---:|---:|---:|---|
| balanced_rps16 | ep4/least_outstanding | 0 | 301.35 | 69.91 | 3242.25 | 6300.56 | True |
| balanced_rps16 | four_ep1/least_outstanding | 0 | 141.00 | 10.50 | 1607.01 | 5728.86 | True |
| balanced_rps16 | mixed/least_outstanding | 0 | 341.02 | 30.93 | 2725.25 | 7966.53 | True |
| balanced_rps16 | mixed/p_to_ep | 0 | 668.83 | 70.76 | 3367.57 | 8466.79 | True |
| balanced_rps16 | mixed/p_to_pp | 0 | 157.92 | 28.72 | 2529.42 | 6787.19 | True |
| balanced_rps16 | two_ep2/least_outstanding | 0 | 269.71 | 36.95 | 2772.95 | 6551.80 | True |
| balanced_rps16 | two_pp2/least_outstanding | 0 | 153.05 | 23.33 | 2497.83 | 6993.53 | True |
| balanced_rps32 | ep4/least_outstanding | 0 | 493.90 | 34.71 | 2559.53 | 4822.15 | True |
| balanced_rps32 | four_ep1/least_outstanding | 0 | 233.90 | 30.60 | 2230.73 | 4713.79 | True |
| balanced_rps32 | mixed/least_outstanding | 0 | 515.93 | 31.15 | 2704.47 | 7008.69 | True |
| balanced_rps32 | mixed/p_to_ep | 0 | 1187.75 | 55.21 | 3608.01 | 7530.20 | True |
| balanced_rps32 | mixed/p_to_pp | 0 | 352.83 | 77.08 | 3329.08 | 5122.54 | True |
| balanced_rps32 | two_ep2/least_outstanding | 0 | 351.19 | 37.81 | 2456.20 | 4653.36 | True |
| balanced_rps32 | two_pp2/least_outstanding | 0 | 394.32 | 25.39 | 2606.00 | 6269.65 | True |
| cold_b16_q8192 | ep4/least_outstanding | 0 | 715.72 | — | 715.72 | 1005.49 | True |
| cold_b16_q8192 | four_ep1/least_outstanding | 0 | 504.93 | — | 504.93 | 714.92 | True |
| cold_b16_q8192 | mixed/least_outstanding | 0 | 580.16 | — | 580.16 | 985.78 | True |
| cold_b16_q8192 | two_ep2/least_outstanding | 0 | 665.32 | — | 665.32 | 972.93 | True |
| cold_b16_q8192 | two_pp2/least_outstanding | 0 | 430.30 | — | 430.30 | 732.25 | True |
| cold_b32_q512 | ep4/least_outstanding | 0 | 220.66 | — | 220.66 | 225.12 | True |
| cold_b32_q512 | four_ep1/least_outstanding | 0 | 163.80 | — | 163.80 | 170.09 | True |
| cold_b32_q512 | mixed/least_outstanding | 0 | 193.52 | — | 193.52 | 235.71 | True |
| cold_b32_q512 | two_ep2/least_outstanding | 0 | 205.32 | — | 205.32 | 210.87 | True |
| cold_b32_q512 | two_pp2/least_outstanding | 0 | 145.80 | — | 145.80 | 174.33 | True |
| cold_b8_q16384 | ep4/least_outstanding | 0 | 826.57 | — | 826.57 | 1019.55 | True |
| cold_b8_q16384 | four_ep1/least_outstanding | 0 | 570.81 | — | 570.81 | 722.84 | True |
| cold_b8_q16384 | mixed/least_outstanding | 0 | 633.66 | — | 633.66 | 968.90 | True |
| cold_b8_q16384 | two_ep2/least_outstanding | 0 | 772.77 | — | 772.77 | 995.48 | True |
| cold_b8_q16384 | two_pp2/least_outstanding | 0 | 506.35 | — | 506.35 | 773.99 | True |
| dheavy_rps16 | ep4/least_outstanding | 0 | 269.93 | 41.09 | 3863.54 | 6519.83 | True |
| dheavy_rps16 | four_ep1/least_outstanding | 0 | 113.78 | 11.06 | 2289.92 | 6109.64 | True |
| dheavy_rps16 | mixed/least_outstanding | 0 | 505.24 | 21.45 | 3653.48 | 8305.92 | True |
| dheavy_rps16 | mixed/p_to_ep | 0 | 1862.37 | 14.57 | 4458.20 | 11118.81 | True |
| dheavy_rps16 | mixed/p_to_pp | 0 | 849.62 | 14.53 | 3867.29 | 8879.22 | True |
| dheavy_rps16 | two_ep2/least_outstanding | 0 | 221.65 | 28.99 | 3721.97 | 6802.51 | True |
| dheavy_rps16 | two_pp2/least_outstanding | 0 | 759.00 | 17.37 | 3756.23 | 8593.16 | True |
| dheavy_rps32 | ep4/least_outstanding | 0 | 297.54 | 25.77 | 2997.73 | 4611.67 | True |
| dheavy_rps32 | four_ep1/least_outstanding | 0 | 154.66 | 14.06 | 2646.66 | 4727.44 | True |
| dheavy_rps32 | mixed/least_outstanding | 0 | 846.48 | 22.59 | 3653.42 | 7780.31 | True |
| dheavy_rps32 | mixed/p_to_ep | 0 | 2150.40 | 25.06 | 4728.35 | 10035.63 | True |
| dheavy_rps32 | mixed/p_to_pp | 0 | 852.14 | 18.38 | 3351.94 | 6892.65 | True |
| dheavy_rps32 | two_ep2/least_outstanding | 0 | 283.35 | 26.41 | 3194.07 | 4931.57 | True |
| dheavy_rps32 | two_pp2/least_outstanding | 0 | 1114.10 | 18.77 | 3842.30 | 7520.24 | True |
| pheavy_rps16 | ep4/least_outstanding | 0 | 340.07 | 102.85 | 2786.98 | 6097.75 | True |
| pheavy_rps16 | four_ep1/least_outstanding | 0 | 182.47 | 10.46 | 903.49 | 5468.70 | True |
| pheavy_rps16 | mixed/least_outstanding | 0 | 209.18 | 51.85 | 1988.98 | 6843.88 | True |
| pheavy_rps16 | mixed/p_to_ep | 0 | 677.14 | 65.85 | 2401.14 | 6330.15 | True |
| pheavy_rps16 | mixed/p_to_pp | 0 | 382.52 | 84.38 | 2282.88 | 5839.82 | True |
| pheavy_rps16 | two_ep2/least_outstanding | 0 | 317.09 | 92.84 | 2633.79 | 6265.20 | True |
| pheavy_rps16 | two_pp2/least_outstanding | 0 | 125.04 | 31.59 | 1529.99 | 6447.07 | True |
| pheavy_rps32 | ep4/least_outstanding | 0 | 754.72 | 64.43 | 2512.01 | 5118.63 | True |
| pheavy_rps32 | four_ep1/least_outstanding | 0 | 364.21 | 51.19 | 1732.41 | 4000.57 | True |
| pheavy_rps32 | mixed/least_outstanding | 0 | 520.79 | 45.08 | 2050.00 | 5715.47 | True |
| pheavy_rps32 | mixed/p_to_ep | 0 | 1431.12 | 69.52 | 3264.47 | 5519.40 | True |
| pheavy_rps32 | mixed/p_to_pp | 0 | 1114.30 | 84.39 | 3080.43 | 5094.62 | True |
| pheavy_rps32 | two_ep2/least_outstanding | 0 | 624.24 | 59.40 | 2284.61 | 4888.44 | True |
| pheavy_rps32 | two_pp2/least_outstanding | 0 | 327.33 | 37.35 | 1754.24 | 5170.66 | True |

