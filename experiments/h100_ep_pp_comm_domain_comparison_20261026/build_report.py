"""Offline, provenance-preserving comparison of saved H100 communication domains."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MANAGER = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
TWO = ROOT / 'experiments/h100_ep2_pp2_model_weight_analysis_20261026'
EIGHT = MANAGER / 'glm47_h100_expert_tp8_ep8_pp2ep4_20261007'
COMM = MANAGER / 'h100_ep8_ep4_comm_locality_20261008_r3'
RECENT = ROOT / 'experiments/h100_decode_pp_opt_20261026'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_csv(name, rows):
    with (HERE / name).open('w') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def save_figure(fig, stem):
    metadata = {
        'png': {'Software': 'H100 historical communication-domain analysis'},
        'svg': {'Date': None, 'Creator': 'H100 historical communication-domain analysis'},
        'pdf': {'CreationDate': None, 'ModDate': None,
                'Creator': 'H100 historical communication-domain analysis'},
    }
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(HERE / (stem + '.' + ext), dpi=180, metadata=metadata[ext])


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'sources').mkdir(exist_ok=True)
    sources = []
    inputs = {
        'two_gpu_matched.csv': TWO / 'matched_workloads.csv',
        'two_gpu_manifest.json': TWO / 'manifest.json',
        'eight_gpu_metrics.csv': EIGHT / 'metrics.csv',
        'eight_gpu_manifest.json': EIGHT / 'manifest.json',
        'communication_metrics.csv': COMM / 'metrics.csv',
        'communication_manifest.json': COMM / 'manifest.json',
        'communication_verification.json': COMM / 'verification.json',
        'recent_decode_metrics.csv': RECENT / 'analysis/fixed_groups/metrics.csv',
    }
    for name, path in inputs.items():
        target = HERE / 'sources' / name
        shutil.copy2(path, target)
        assert sha(target) == sha(path)
        sources.append(dict(path=str(path), snapshot='sources/' + name, sha256=sha(path)))
    small = load_csv(HERE / 'sources/two_gpu_matched.csv')
    large = load_csv(HERE / 'sources/eight_gpu_metrics.csv')
    comm = load_csv(HERE / 'sources/communication_metrics.csv')
    eight_manifest = json.loads((HERE / 'sources/eight_gpu_manifest.json').read_text())
    comm_manifest = json.loads((HERE / 'sources/communication_manifest.json').read_text())
    verification = json.loads((HERE / 'sources/communication_verification.json').read_text())
    assert verification['passed'] and verification['aligned_iterations_per_span'] == 3
    assert len({comm_manifest['implementation']['native_sha256'],
                eight_manifest['deepep_custom']['sha256']}) == 1
    assert comm_manifest['workload']['route_padding'] is False
    assert comm_manifest['model']['loaded'] is False
    accepted = {arm for run in eight_manifest['verification']['guard']
                if run['scientific_success'] and run['cleanup_verified']
                for arm in run['accepted_arms']}
    assert {'ep8', 'pp2ep4'} <= accepted

    comparisons = []
    metrics = []
    for phase in ('prefill', 'decode'):
        s = next(r for r in small if r['model'] == 'Qwen3.5-122B-A10B-FP8' and r['phase'] == phase)
        cases = [('2gpu_b8', 'Qwen3.5-122B-A10B FP8', 2, 8,
                  float(s['ep_rate']), float(s['pp_rate']), s, None)]
        for batch in (4, 16):
            e = next(r for r in large if r['arm'] == 'ep8' and r['workload'] == f'{phase}_c{batch}')
            p = next(r for r in large if r['arm'] == 'pp2ep4' and r['workload'] == f'{phase}_c{batch}')
            key = 'input_tok_s' if phase == 'prefill' else 'output_tok_s'
            cases.append((f'8gpu_b{batch}', 'GLM-4.7-AWQ INT4', 8, batch,
                          float(e[key]), float(p[key]), e, p))
        for case, model, gpu, batch, er, pr, e, p in cases:
            comparisons.append(dict(case=case, model=model, gpus=gpu, requests=batch, phase=phase,
                ep_throughput=er, pp_throughput=pr, ep_over_pp=er / pr,
                ep_vs_pp_pct=100 * (er / pr - 1),
                throughput_unit='input tok/s' if phase == 'prefill' else 'output tok/s'))
            for layout in ('ep', 'pp'):
                if gpu == 2:
                    ttft = float(e[layout + '_ttft_s'])
                    tpot = float(e[layout + '_tpot_ms']) if phase == 'decode' else ''
                    end = float(e[layout + '_e2e_s'])
                else:
                    row = e if layout == 'ep' else p
                    assert int(float(row['trials'])) == 3
                    ttft = float(row['mean_ttft_s'])
                    tpot = float(row['mean_tpot_s']) * 1000 if phase == 'decode' else ''
                    end = float(row['mean_e2e_s'])
                metrics.append(dict(case=case, phase=phase, model=model, gpus=gpu,
                    requests=batch, layout=('EP2' if layout == 'ep' else 'PP2') if gpu == 2
                    else ('EP8' if layout == 'ep' else 'PP2xEP4'),
                    throughput=er if layout == 'ep' else pr,
                    throughput_unit='input tok/s' if phase == 'prefill' else 'output tok/s',
                    mean_ttft_s=ttft, mean_tpot_ms=tpot, mean_e2e_s=end,
                    preexisting_cached_tokens_per_request=4096 if gpu == 2 and phase == 'decode' else 0,
                    fresh_input_tokens_per_request=(4096 if phase == 'prefill' else 1) if gpu == 2
                    else (8192 if phase == 'prefill' else 128),
                    output_tokens_per_request=(1 if phase == 'prefill' else 128) if gpu == 2
                    else (1 if phase == 'prefill' else 512)))
    assert len(metrics) == 12 and len(comparisons) == 6
    # Preserve each local node separately; do not invent a pooled local median.
    locality = []
    for case in ('decode4', 'decode16', 'prefill1024'):
        selected = [r for r in comm if r['case'] == case and r['operation'] == 'dc']
        assert {r['phase'] for r in selected} == {'local174', 'local175', 'cross8'}
        cross = next(r for r in selected if r['phase'] == 'cross8')
        assert all(int(r['n']) == 3 for r in selected)
        for local in (r for r in selected if r['phase'].startswith('local')):
            locality.append(dict(case=case, global_logical_tokens=int(cross['tokens']),
                mode=cross['mode'], local_node=local['phase'],
                ep4_local_median_ms=float(local['median_ms']),
                ep8_cross_median_ms=float(cross['median_ms']),
                cross_over_local=float(cross['median_ms']) / float(local['median_ms']),
                extra_ms=float(cross['median_ms']) - float(local['median_ms'])))
    save_csv('full_metrics.csv', metrics)
    save_csv('e2e_comparisons.csv', comparisons)
    save_csv('communication_comparisons.csv', locality)

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': HERE.name})
    colors = {'ep': '#159989', 'pp': '#d19139'}
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.7), sharey=True)
    labels = ['2 GPUs / NVLink\nQwen3.5-122B FP8 / 8 requests',
              '8 GPUs / 2 nodes\nGLM-4.7 INT4 / 4 requests',
              '8 GPUs / 2 nodes\nGLM-4.7 INT4 / 16 requests']
    for ax, phase in zip(axes, ('prefill', 'decode')):
        rows = [r for r in comparisons if r['phase'] == phase]
        vals = [r['ep_over_pp'] for r in rows]
        ax.barh(range(3), vals, height=.57,
                color=[colors['ep'] if v >= 1 else colors['pp'] for v in vals])
        ax.axvline(1, color='#555', linestyle='--', linewidth=1.4)
        ax.set_yticks(range(3), labels)
        ax.set_xlim(0, 1.62)
        ax.set_xlabel('EP throughput / PP throughput (PP = 1)')
        ax.set_title(('Prefill: new input token/s' if phase == 'prefill' else 'Decode: output token/s'), pad=16)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        for j, r in enumerate(rows):
            ax.text(r['ep_over_pp'] + .025, j - .045,
                    f"{r['ep_over_pp']:.3f}x  ({r['ep_vs_pp_pct']:+.1f}%)", va='center', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=.5))
            ax.text(.025, j + .40,
                    f"EP {r['ep_throughput']:,.1f} / PP {r['pp_throughput']:,.1f} tok/s",
                    fontsize=9.5, color='#444')
        ax.text(1.015, 1.012, 'EP faster >', transform=ax.get_xaxis_transform(), fontsize=9, color='#137b70')
    axes[0].set_ylim(2.65, -.55)
    fig.suptitle('H100 EP/PP outcomes: NVLink-local vs cross-node deployments', fontsize=17, y=.98)
    fig.text(.5, .095, '2 GPUs: EP2 vs PP2.  8 GPUs: EP8 vs PP2 x EP4.  Each bar is normalized within its own saved cohort.',
             ha='center', fontsize=10)
    fig.text(.5, .055, '2-GPU Prefill: no cache, input4096 -> output1; Decode: GPUcached4096 + new1 -> output128, 8 requests.',
             ha='center', fontsize=9.5)
    fig.text(.5, .022, '8-GPU Prefill: no cache, input8192 -> output1; Decode: cold input128 -> output512. Models/backends/chunks differ: not a network-only ablation.',
             ha='center', fontsize=9.5, color='#9b4e26')
    fig.subplots_adjust(left=.22, right=.985, top=.84, bottom=.22, wspace=.20)
    save_figure(fig, 'e2e_communication_domain')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={'width_ratios': [1.6, 1]})
    cases = [('decode4', '4 logical tokens'), ('decode16', '16 logical tokens')]
    for ax, columns, title in ((axes[0], cases, 'Decode-like small messages: CUDA Graph'),
                               (axes[1], [('prefill1024', '1024 logical tokens')], 'Prefill-like large message: eager')):
        width = .23
        for i, (phase, label, color) in enumerate((('local174', 'EP4: node174 NVLink', '#159989'),
                ('local175', 'EP4: node175 NVLink', '#75bdb2'), ('cross8', 'EP8: cross-node RDMA', '#9464ca'))):
            rows = [next(r for r in comm if r['case'] == c and r['operation'] == 'dc' and r['phase'] == phase)
                    for c, _ in columns]
            vals = np.array([float(r['median_ms']) for r in rows])
            low = np.array([float(r['min_ms']) for r in rows])
            high = np.array([float(r['max_ms']) for r in rows])
            x = np.arange(len(columns)) + (i - 1) * width
            ax.bar(x, vals, width=width, label=label, color=color,
                   yerr=np.vstack([vals - low, high - vals]), capsize=3,
                   error_kw={'linewidth': .9})
            for xx, val, hi in zip(x, vals, high):
                ax.text(xx, hi + (.007 if len(columns) == 2 else .045), f'{val:.4f}',
                        ha='center', va='bottom', fontsize=10)
        ax.set_xticks(np.arange(len(columns)), [v for _, v in columns])
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_ylabel('Dispatch + Combine (ms)')
        ax.set_ylim(0, .42 if len(columns) == 2 else 2.55)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[1].text(.5, .88, 'Cross/local: 9.76-10.06x', transform=axes[1].transAxes, ha='center', fontsize=11)
    fig.suptitle('Mechanism check: same DeepEP binary, same logical token count, no model compute', fontsize=15, y=.98)
    fig.text(.5, .085, 'H100174/175; BF16 hidden5120, 160 experts, top8; 3 warmups + 3 samples. Bars: rank-max median; whiskers: observed min/max.',
             ha='center', fontsize=9.5)
    fig.text(.5, .043, 'Local and cross-node EP partition/rank count differ. Event spans include waits; these are not request TPOT or measured NIC wire times.',
             ha='center', fontsize=9.5)
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.20, wspace=.26)
    save_figure(fig, 'communication_locality')
    plt.close(fig)

    comp_table = [[r['model'] + f"／{r['gpus']}卡／{r['requests']}请求", r['phase'],
                   f"{r['ep_throughput']:,.2f}", f"{r['pp_throughput']:,.2f}",
                   f"{r['ep_over_pp']:.3f}×", f"{r['ep_vs_pp_pct']:+.2f}%"] for r in comparisons]
    full_table = [[r['case'], r['phase'], r['layout'], f"{r['throughput']:,.2f}",
                   f"{r['mean_ttft_s']:.4f}", f"{r['mean_tpot_ms']:.3f}" if r['mean_tpot_ms'] != '' else '不适用',
                   f"{r['mean_e2e_s']:.4f}"] for r in metrics]
    comm_table = []
    for case in ('decode4', 'decode16', 'prefill1024'):
        rows = [r for r in locality if r['case'] == case]
        comm_table.append([rows[0]['global_logical_tokens'], rows[0]['mode'],
                           f"{rows[0]['ep4_local_median_ms']:.6f}", f"{rows[1]['ep4_local_median_ms']:.6f}",
                           f"{rows[0]['ep8_cross_median_ms']:.6f}",
                           f"{min(r['cross_over_local'] for r in rows):.2f}–{max(r['cross_over_local'] for r in rows):.2f}×"])
    recent = load_csv(HERE / 'sources/recent_decode_metrics.csv')
    ep = next(r for r in recent if r['model'] == '122b' and r['arm'] == 'ep2')
    pp = next(r for r in recent if r['model'] == '122b' and r['arm'] == 'pp2' and r['mode'] == 'early_activation')
    sensitivity = 100 * (float(ep['output_tokens_per_s']) / float(pp['output_tokens_per_s']) - 1)
    assert sensitivity > 0
    report = '''# H100：通信域改变时，EP 与 PP 的选择是否变化？

## 术语、资源和计时边界

这是现有结果的离线汇总，没有申请GPU、启动引擎或增加性能样本。不是同一模型从2卡扩到8卡的扩容实验，也不是严格相同workload的网络消融。

- **两卡组：** Qwen3.5-122B-A10B FP8，单节点H100 178的NVLink卡对。EP2为Attention两个数据并行实例、每卡Attention不做张量并行，专家跨两卡；PP2为两stage、每stage本地全部专家，两个流水槽位各4请求。历史EP用GPU0/1、PP用GPU2/3，同节点两组并行。
- **八卡组：** GLM-4.7-AWQ INT4/Humming，H100 174/175各4卡。EP8为Attention节点内TP4×DP2、专家跨两节点；PP2×EP4为每节点一个stage、Attention TP4、专家EP4只用节点内NVLink。跨节点EP使用NVSHMEM RDMA/RoCE，PP仍有NCCL阶段传输、反馈和调度。
- **Prefill负载：** 两卡为8请求，每条无旧缓存、新输入4096、输出1；八卡为4或16请求，每条无旧缓存、新输入8192、输出1。
- **Decode负载：** 两卡为8请求，每条已有4096个真实模型生成的前缀token驻留GPU，本轮新增输入1、生成128；八卡为4或16请求，每条从无缓存的128个输入开始、生成512。八卡Decode的TTFT包含这128个输入的Prefill。两组都没有CPU HiCache换入或MTP；Prefill eager、Decode CUDA Graph。
- **指标：** Prefill吞吐只计新输入token/s，Decode吞吐计输出token/s，分母为整批完成时间；每个点3轮。TTFT为提交至首token、TPOT为首末token间隔除以其间输出token数、端到端为提交至末token；延迟表为请求均值。不同模型的绝对吞吐不作高低排名。
- **图中归一化：** 同一历史cohort的EP吞吐除以其PP吞吐；PP为1。大于1表示EP更快，小于1表示PP更快。它展示配置选择，不等于纯网络造成的倍率。

## 1. 端到端：Decode的部署偏好确实反转

'''+table(['模型／资源／请求数', '阶段', 'EP吞吐', 'PP吞吐', 'EP÷PP', 'EP相对PP'], comp_table)+'''

![按各自cohort的PP归一化，附绝对吞吐](e2e_communication_domain.png)

两卡NVLink组，Decode的EP2比PP2快约24.87%；跨节点八卡组，EP8比PP2×EP4慢约21.50%（4请求）或17.12%（16请求）。Prefill两组均偏向PP，但跨节点组的差距更大。**这是符合通信域影响部署选择的现象证据，不是网络因果效应的精确估计。**

## 2. 更直接的机制证据：同库、固定全局逻辑数据量的纯通信

使用独立归档`h100_ep8_ep4_comm_locality_20261008_r3`：同一个自定义DeepEP SM90二进制，SHA256为`d9b26ac86748517800ce9f9734d865be20d089d473d9d4f6d66dde98c673d76b`。隐藏维度5120、BF16、160个专家、每token命中8个专家、确定性均衡路由，无padding。每个相同token数的local/cross实验使用相同逻辑token及路由，不加载模型、不计算Attention/Expert，也不运行PP调度。这里的token数指本轮参与通信的全局逻辑输入token数，不是已缓存前缀长度，也不直接等于完整服务的请求数。

三次预热、三次采样；每次取对齐rank中的最大CUDA事件跨度，再取三次中位数。两个本地节点的中位数分别保留，没有伪造合并中位数。下表单位ms，一次Dispatch＋Combine：

'''+table(['全局逻辑token数', '执行模式', '节点174内EP4', '节点175内EP4', '跨节点EP8', '跨节点÷节点内'], comm_table)+'''

![纯通信绝对耗时，误差线为三次样本的观察范围](communication_locality.png)

小消息通信代价增加约5.7–6.1倍，1024-token大消息约9.8–10.1倍。这支持“慢通信域给宽EP引入更高成本”的机制；**它测的是完成时间增长，不是总通信字节量同比增长**。跨域同时改变EP4/EP8的rank数、每rank行数及专家/QP分区；官方QP规则为40→20，因此还不是固定rank布局、只改网线的硬件消融。事件跨度含等待，不能解释成NIC纯传输时间。

## 3. 可以提炼的insight及证据限制

**PP/EP的选择取决于专家并行组跨越哪一种通信域。** 同一快速NVLink域内，EP减少每卡每层专家工作集的收益可能超过Dispatch/Combine成本；跨慢域做宽EP时，每个MoE层都要支付跨域交换，PP可以把专家交换留在快域，把跨域传输集中到stage边界。最终仍需计入PP额外Expert计算、阶段反馈、流水利用率与KV容量。

现有数据足以展示这个方向和一次Decode胜负反转，但不能把端到端差距全部归因于网络：

- 模型、精度、专家矩阵、层数、Attention布局、请求数、缓存起点和输出长度不同；没有找到严格同workload的这两组H100完整推理对照。
- 八卡早期EP8使用自定义DeepEP，PP使用预装本地NVLink库；Expert内核布局和共享专家组织不同。Prefill实际本地chunk为EP512、PP1024；不能把其约3.3倍差距当作纯跨节点代价。
- 两卡组历史结果不是后续PP提前发送优化版本。后续同卡对固定分组补测，122B的EP仍比优化PP快约'''+f'{sensitivity:.2f}%'+'''，所以两卡Decode偏向EP并未因这项调度修正消失；不同轮次不合并。
- 纯通信是在相同库与逻辑路由下的更强机制对照，但不含Expert、实际PP边界或重叠，不能用其倍率直接预测完整模型速度，也不能把独立通信中位数相加成TPOT。
- 每格只有3次正式测量；图中没有统计显著性或稳定服务尾延迟的声明。历史Guard/缓存资格边界保持原样，没有重写失败记录。

## 4. 全部请求指标

两卡`2gpu_b8`；八卡`8gpu_b4/b16`。Prefill吞吐为input token/s，Decode为output token/s；TTFT/端到端单位秒，TPOT单位毫秒。

'''+table(['cohort', '阶段', '配置', '吞吐', 'TTFT s', 'TPOT ms', '端到端 s'], full_table)+'''

## 5. 可追踪来源

[完整指标](full_metrics.csv) · [六组归一化](e2e_comparisons.csv) · [六组本地/跨节点通信比较](communication_comparisons.csv) · [原始路径与SHA256](source_manifest.json) · [离线校验](validation.json)。原始CSV及小型元数据在`sources/`，原始trace、权重和KV数据均未复制。

两卡来源为Q03/Q07；八卡来源为D01/D02/P01/P02；纯通信来自同库EP4/EP8 locality r3。重生成使用`build_report.py`，不会启动GPU实验。
'''
    (HERE / 'REPORT.md').write_text(report)
    (HERE / 'source_manifest.json').write_text(json.dumps(sources, indent=2, ensure_ascii=False)+'\n')
    validation = dict(passed=True, offline_only=True, new_gpu_samples=0, e2e_rows=len(metrics),
        e2e_comparisons=len(comparisons), communication_comparisons=len(locality),
        source_copies_byte_verified=len(sources), same_comm_native_verified=True,
        historical_comm_validation=verification, e2e_exact_workload_match=False,
        pure_network_causal_claim=False, newer_optimized_two_gpu_ep_advantage_pct=sensitivity)
    (HERE / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    manifest = dict(id=HERE.name, title='H100 EP/PP communication-domain comparison: two vs eight GPUs',
        created_at='2026-10-26', status='complete_offline_analysis',
        implementation=dict(engine='SGLang / DeepEP historical artifacts',
            engine_commit=eight_manifest['implementation']['commit'],
            source_records='source_manifest.json', launched_new_engine=False),
        workload=dict(exact_cross_scale_match=False, e2e='full_metrics.csv',
            equal_logical_tokens_comm=[4,16,1024]),
        deployment=dict(new_gpu_launch=False), testbed=dict(gpu='H100 80GB',
            two_gpu_nodes=[178], eight_gpu_nodes=[174,175],
            local='NVLink', cross_node='NVSHMEM RDMA/RoCE'),
        model=['Qwen3.5-122B-A10B FP8', 'GLM-4.7-AWQ INT4'],
        engine_parameters=dict(two_gpu='EP2 AttentionTP1/DP2 vs two-slot PP2 AttentionTP1/EP1',
            eight_gpu='EP8 AttentionTP4/DP2 vs PP2xEP4 AttentionTP4 per stage',
            graph='E2E Prefill eager, Decode Graph; communication modes separated',
            eight_gpu_native_libraries_differ=True,
            eight_gpu_local_prefill_chunk=dict(ep=512, pp=1024)),
        key_results=dict(e2e=comparisons, communication=locality,
            newer_optimized_two_gpu_ep_advantage_pct=sensitivity),
        artifacts=dict(report='REPORT.md', e2e='e2e_communication_domain.png',
            communication='communication_locality.png', validation='validation.json'),
        limits=['different models/workloads/backends/chunks',
                'No network-only E2E causality or extrapolated GPU speedup'])
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
