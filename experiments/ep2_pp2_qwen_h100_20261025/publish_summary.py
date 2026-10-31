"""Idempotently index one completed comparison through a human-readable report."""
import argparse
import csv
import json
from pathlib import Path


def publish(run):
    analysis = run / 'analysis'
    verified = json.loads((analysis / 'validated_metrics.json').read_text())
    assert verified['passed'] and len(verified['rows']) == 14
    post = json.loads(Path(verified['guard_postcheck']).read_text())
    assert post['passed'] and post['cleanup_verified']
    project = run.parents[2]
    summary = project / 'experiment_summary'
    report = summary / 'component_reports/Q03.md'
    rows = list(csv.DictReader((analysis / 'full_metrics.csv').open()))
    lookup = {(r['case'],r['phase'],r['arm']):r for r in rows}
    capacities = verified['logical_capacities']
    lines = ['# Q03：H100 双卡 EP2 与两槽 PP2，中等负载及接近容量上限', '',
        '## 测的是什么', '',
        '**请求数**指整批同时提交的独立请求；**已缓存上下文**指计时前已经由真实模型生成并驻留 GPU 的前缀，'
        '不是本轮重新 Prefill 的输入。Prefill 测新增输入、输出 1 token；Decode 每条新增 1 token、生成 128 token。'
        '本轮没有 CPU HiCache 换入、合成 KV 或 MTP；完整 CPU 权重 Bank 只用于加快模型加载。', '',
        '**TTFT** 从发送请求到收到首 token；**TPOT** 是首 token 到最后 token 的时间除以后续 127 个输出；'
        '**端到端延迟** 到最后 token 为止；**吞吐**按整批完成时间计算，只计新输入或实际输出。'
        '形状预热、旧上下文准备和独立 profile 不计入正式计时。每格 3 轮；下表延迟均为请求均值，'
        '中位数和 P99 见完整报告。', '',
        '模型 **Qwen3.5-122B-A10B-FP8，完整 48 层**。同一已预约 H100 节点 178、四张 80GB GPU：'
        'GPU0/1 跑 EP2，GPU2/3 跑 PP2，两对均为 NVLink NV6，同步并行测量，没有跨节点通信。'
        'EP2：每卡 Attention TP1、两个 Attention 数据并行副本，专家跨两卡 EP2；'
        'PP2：每卡 24 层，每层全部专家，每阶段 Attention TP1、Expert EP1。PP 始终两个等大槽位，'
        '8 请求拆为两组 4，16 请求拆为两组 8。', '',
        '两组均使用 H100 原生 FP8 W8A8 DeepGEMM、BF16 KV、FP32 线性 Attention 状态，'
        'FA3 全 Attention、Triton 线性 Attention，实际 Decode CUDA Graph 已核验。'
        'EP2 Decode 使用低延迟 DeepEP 和原生 overlap；该引擎版本对 PP 自动关闭 overlap。', '',
        '## 同一负载的 Prefill', '',
        '| 请求组成 | EP2 新输入吞吐，token/s | PP2 新输入吞吐，token/s | PP2 吞吐变化 | EP2／PP2 平均 TTFT，秒 |',
        '|---|---:|---:|---:|---:|']
    labels = dict(moderate='8 请求，每条无旧缓存、新增 4096',
                  near_capacity_b8='8 请求，每条已缓存 26752、新增 512',
                  near_capacity_b16='16 请求，每条已缓存 13376、新增 512')
    for case, label in labels.items():
        e, p = (lookup[case,'prefill',a] for a in ('ep2','pp2'))
        delta = (float(p['tokens_per_s'])/float(e['tokens_per_s'])-1)*100
        lines.append(f"| {label} | {float(e['tokens_per_s']):.2f} | {float(p['tokens_per_s']):.2f} | "
                     f"{delta:+.2f}% | {float(e['ttft_s_mean']):.3f}／{float(p['ttft_s_mean']):.3f} |")
    lines += ['', '## 同一负载的 Decode', '',
        '三行每条均是新增 1、生成 128；共同长缓存负载总计 214016 token，约占 EP2 实测 KV 容量的 90%。', '',
        '| 请求组成 | EP2 输出吞吐，token/s | PP2 输出吞吐，token/s | EP2／PP2 平均 TTFT，秒 | EP2／PP2 平均 TPOT，毫秒 | EP2／PP2 平均端到端，秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for case in labels:
        e, p = (lookup[case,'decode',a] for a in ('ep2','pp2'))
        length = e['cached_context_per_request']
        lines.append(f"| {e['requests']} 请求，每条已缓存 {length} | {float(e['tokens_per_s']):.2f} | "
            f"{float(p['tokens_per_s']):.2f} | {float(e['ttft_s_mean']):.3f}／{float(p['ttft_s_mean']):.3f} | "
            f"{float(e['tpot_ms_mean']):.3f}／{float(p['tpot_ms_mean']):.3f} | "
            f"{float(e['e2e_s_mean']):.3f}／{float(p['e2e_s_mean']):.3f} |")
    own = lookup['pp_own_capacity_b16','decode','pp2']
    own_pre = lookup['pp_own_capacity_b16','prefill','pp2']
    lines += ['', '## PP 自身容量边界，单列，不作同工作量排名', '',
        f"实测逻辑 KV 容量：EP2 **{capacities['ep2']:,} token**，PP2 **{capacities['pp2']:,} token**，"
        f"约 **{capacities['pp2']/capacities['ep2']:.2f} 倍**。PP 两个阶段存不同层，容量不能相加。", '',
        f"PP2 另测 16 请求、每条已缓存 **{int(own['cached_context_per_request']):,} token**，"
        f"总计 **{16*int(own['cached_context_per_request']):,} token**，约占其自身容量 90%。"
        f"新增 512、输出 1 的 Prefill 吞吐 **{float(own_pre['tokens_per_s']):.2f} token/s**；"
        f"新增 1、输出 128 的 Decode 吞吐 **{float(own['tokens_per_s']):.2f} token/s**，"
        f"平均 TPOT **{float(own['tpot_ms_mean']):.3f} 毫秒**。", '',
        '该工作量超过 EP2 的 GPU 全驻留容量；本轮没有替 EP2 运行 CPU HiCache／排队策略，'
        '因此只能证明 PP 能全驻留跑通，不能声称这个大工作量下 PP 比 EP 快多少。', '',
        '## 优化和核验', '',
        '- 通信缓冲区先初始化，再实测 KV；EP 的 Graph 和线性状态池按 DP 后的每卡 batch 配置。'
        'EP 全局状态池 88、每卡 44；PP 每阶段 84。',
        '- 启用原生 `--mamba-max-states-per-path 1`，清掉沿途不再需要的中间 chunk 状态，'
        '保留末端、分叉和锁定状态。避免仅靠扩大状态池挤占 KV；没有改模型计算或状态精度。'
        '两组最长前缀驻留资格检查及全部正式请求缓存命中检查通过。',
        '- PP Prefill 使用短 A/B/B/A 选出的发送就绪策略，本轮局部派生吞吐约 +1.08%，'
        '首 token 输出检查通过；Decode 保留原生产发送路径。7 项本轮协议测试与 7 项原生剪枝 CPU 测试通过。',
        '- 仅计时外建缓存时每组最多 4 请求；正式计时中所有 8／16 请求仍同时提交，未按 GPU 容量客户端串行化。'
        '每轮缓存 Prefill 都从相同基础前缀开始，不把上一轮分支挤出缓存造成的重算混入结果。',
        '- Guard 科学结果、完整 CPU 权重 Bank 使用、退出清理检查均通过，GPU 租约已释放。'
        '失败尝试原样保留，不混入正式表。', '',
        '## 该怎样解释', '',
        '**A100 上两槽 PP 的 Decode 吞吐优势，没有在本轮 H100 原生路径重现。** '
        '三个同工作量 Decode 点里，PP 吞吐低约 20%–23%，尽管其首 token 更早。'
        'Prefill 不是固定胜负：8 条长缓存请求时 PP 吞吐高 19.69%；相同总缓存拆成 16 条请求时，'
        '本轮均值低 5.61%，但 PP 的平均 TTFT 仍较低。', '',
        '独立 Decode Graph profile 中，中等负载的 EP 每卡完整 48 层 GPU 跨度约 9.93–10.04 毫秒；'
        'PP 两个 24 层阶段分别约 5.05／5.37 毫秒。EP Graph 中 DeepEP kernel 合计约 1.73–2.10 毫秒；'
        'PP Graph 中没有 DeepEP，但仍有 Graph 外的阶段传输与反馈。'
        'PP 每个阶段的 routed GEMM 合计约 1.84–2.09 毫秒，EP 完整层为 2.12–2.61 毫秒，'
        '显示 PP 集中专家并非没有计算代价。**这些是独立 profiler 样本，不能相加冒充请求 TPOT 或精确剩余归因。**', '',
        '本轮不是“硬件型号”单变量实验：此前 A100 使用 W8A16 Marlin、关闭 Graph／overlap、Decode 走 normal 通信；'
        'H100 使用原生 W8A8、Graph，EP2 还启用了低延迟通信和 overlap。'
        'PP 在本引擎仍受 overlap 支持限制。本次验证了关键配置与短局部优化，未证明所有参数下的全局最优。'
        '每格仅 3 轮，PP 的 16 请求 Prefill 吞吐有波动，不能把小差异推广为普遍结论。', '',
        f'[完整 14 行均值／中位数／P99 报告]({analysis / "REPORT.md"}) · '
        f'[CSV]({analysis / "full_metrics.csv"}) · [逐请求重算与哈希]({analysis / "validated_metrics.json"}) · '
        f'[实际 Graph 分段诊断]({analysis / "GRAPH_PROFILE.md"}) · '
        f'[运行协议]({run.parents[1] / "README.md"})。', '',
        f'![Prefill 对比]({analysis / "prefill_comparison.png"})', '',
        f'![Decode 对比]({analysis / "decode_comparison.png"})', '']
    followup = analysis / 'component_followup/REPORT.md'
    if followup.exists():
        lines += ['## Decode 分项补分析', '',
            f'[各负载的 Attention 核心／路由专家 GEMM／Dispatch／Combine 绝对耗时]({followup})。'
            '仅使用现有 trace，不重跑 GPU；保留未归属投影、共享专家，以及 PP Graph 外传输的测量缺口，'
            '不把 kernel 累计时间当成独占占比或 TPOT。', '']
    report.write_text('\n'.join(lines))
    readme = summary / 'README.md'
    text = readme.read_text()
    row = (f'| [Q03：H100 双卡 EP2／两槽 PP2]({report}) | 单节点 178 四张 H100，两组双卡并行。'
        '中等 8 请求；长缓存总计 214016 token，分 8／16 请求；另有 PP 自身 674816 token 容量点。'
        'Prefill 和 Decode 分开，旧前缀在 GPU，准备和 profile 不计时。 | '
        '同工作量 Decode：PP 吞吐低约 **20%–23%**；8 请求长缓存 Prefill 高 **19.69%**，'
        '16 请求本轮均值低 **5.61%**。KV 逻辑容量 EP **237952**／PP **750848** token。'
        '不是 A100 到 H100 的硬件单变量比较。 | 原生 FP8 DeepGEMM、FA3、实际 Decode Graph；'
        '已核验缓存命中及独立逐卡 GPU trace；完整 TTFT／TPOT／端到端三种统计见报告。 |')
    if followup.exists():
        row = row[:-1] + f' [Decode 分项补分析]({followup})。 |'
    existing = [line for line in text.splitlines() if line.startswith('| [Q03：H100')]
    if existing:
        assert len(existing) == 1
        text = text.replace(existing[0], row)
    else:
        anchor = next(line for line in text.splitlines() if line.startswith('| [Q02：'))
        text = text.replace(anchor, anchor + '\n' + row)
    readme.write_text(text)
    optimizations = summary / 'OPTIMIZATIONS.md'
    text = optimizations.read_text()
    start, end = '<!-- BEGIN Q03 H100 -->', '<!-- END Q03 H100 -->'
    section = '\n'.join([start, '### 5.3 H100：原生 Graph 与线性 Attention 状态缓存预算', '',
        'Qwen3.5-122B 的 H100 对照启用原生 FP8 W8A8 DeepGEMM、FA3、Decode Graph；'
        'EP2 Decode 使用低延迟 DeepEP。PP 始终两个等大槽位，Prefill 发送就绪局部 A/B/B/A 约 +1.08%；'
        '没有把未验收的 Decode 候选打开。', '',
        '排查发现 KV 尚有空间时，沿途保存的中间线性状态也会挤掉完整前缀。'
        '仅将 PP 状态池增至 128 时，KV 逻辑容量为 475712 token，且自身容量点仍不能完整命中；'
        '最终用原生每路径状态剪枝、PP 每阶段 84 个状态，KV 为 750848 token，'
        '16×42176 的基础前缀及全部正式请求命中通过。此处同时调整了状态保留策略和预算，'
        '是缓存容量／公平性修正，不是 DeepEP 内核吞吐提速。', '',
        '同工作量 Decode 仍是 EP2 更快，PP 吞吐低约 20%–23%；'
        '不能因 PP 的层内通信减少就认定端到端一定占优。PP 在该引擎版本不支持 overlap，'
        '跨代还同时改变 FP8 计算和 Graph 路径，不能全部归因于 GPU。', '',
        f'[协议、完整指标、profile 和测试证据]({report})。', end, '', ''])
    if start in text:
        before, rest = text.split(start,1)
        _, after = rest.split(end,1)
        text = before + section.rstrip() + after
    else:
        anchor = '## 6. 权重／KV Bank'
        assert anchor in text
        text = text.replace(anchor, section + anchor, 1)
    optimizations.write_text(text)
    print(json.dumps(dict(published=True, report=str(report), index=str(readme))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    publish(parser.parse_args().run.resolve())
