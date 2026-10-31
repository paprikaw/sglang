"""Join independent node-local comparisons without averaging across nodes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def desc(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write_csv(p, rows):
    with p.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    means = []; comparisons = []; manifests = []; audits = []
    for path in a.runs:
        run = path.resolve(); cfg = json.loads((run/'config.json').read_text())
        dest = out/run.name
        subprocess.run([sys.executable, str(ROOT/'analyze.py'), '--run', str(run), '--output', str(dest)],
                       check=True, stdout=subprocess.DEVNULL)
        manifest = json.loads((dest/'manifest.json').read_text())
        assert manifest['completed_points'] == cfg['expected_points'], run
        checks = list((run/'dashboard_guard/checks').glob('*-post.json'))
        post = max(checks, key=lambda p: p.stat().st_mtime)
        audit = json.loads(post.read_text())
        assert audit['passed'] and audit['cleanup_verified'] and audit['model_cache']['passed'], audit
        audits.append(dict(run=str(run), node=cfg['node'], postcheck=desc(post),
                           passed=True, points=cfg['expected_points']))
        manifests.append(desc(dest/'manifest.json'))
        with (dest/'mean_metrics.csv').open() as f:
            for r in csv.DictReader(f):
                means.append(dict(node=cfg['node'], run=run.name, **r))
        with (dest/'comparisons.csv').open() as f:
            for r in csv.DictReader(f):
                comparisons.append(dict(node=cfg['node'], run=run.name, **r))
        for case in cfg['workloads']:
            point = run/'results/two_pp2'/case['id']
            if any(x['id']=='two_pp2' for x in cfg['layouts']):
                native = json.loads((point/'native_fixed_group_validation.json').read_text())['result']
                ref = [r['output_ids'] for r in native['requests']]
                for i in range(1,6):
                    formal = json.loads((point/f'formal.r{i}.json').read_text())['result']
                    assert [r['output_ids'] for r in formal['requests']] == ref
                    assert len(formal['fixed_group_receipts']) == 2
                    assert len(formal['scheduling_receipt']['deltas']) == 4
                    assert all(d['ahead']>0 and d['ahead']==d['consumed'] and d['fast_sent']>0
                               for d in formal['scheduling_receipt']['deltas'])
    write_csv(out/'mean_metrics.csv', means); write_csv(out/'comparisons.csv', comparisons)
    labels = dict(ep4='EP4', two_pp2='2×PP2（lookahead）', four_ep1='4×单卡', pp4='PP4（原生）')
    pp = {(r['metric'],int(r['N'])):r for r in comparisons if r['layout']=='two_pp2'}
    def pp_advantage(metric,n):
        row=pp[metric,n];ep=float(row['ep4_ms']);other=float(row['candidate_ms'])
        return ('PP2',100*(1-other/ep)) if other<ep else ('EP4',100*(1-ep/other))
    findings=[]
    if pp:
        for metric,label in [('mean_request_tpot_ms','平均 TPOT'),('mean_request_e2e_ms','平均请求 E2E')]:
            parts=[]
            for n in (8,16,32):
                winner,gain=pp_advantage(metric,n);parts.append(f'N{n}：{winner} 延迟低 {gain:.1f}%')
            findings.append('- '+label+'：'+'；'.join(parts)+'。')
    text = ['# 35B 小 batch：四张 H100 的整模型验证', '',
            '问题：固定 GPU 缓存、相同输入和四卡预算，在全组 N=8、16、32 的 Decode 中，'
            'PP 是否反超 EP4？两个 H100 节点并行实验，每个节点都有自己的 EP4 对照。', '',
            '所有数值都是 5 轮算术平均。TPOT 先逐请求计算 `(最后 token − 首 token)/127`，'
            '再对请求和轮次求平均；请求 E2E 包括 TTFT。没有选择最大值或最快样本。', '',
            '## 判断', '', *findings, '',
            '优化 PP2 与 EP4 在同一 Decode 阶段内出现了可重复的 TPOT 换边，双向幅度约 10%。'
            '但含 TTFT 的整请求结果中，较大 N 的 EP4 收益很小；'
            '当前数据不足以支持“动态切换能获得双向很大的整请求收益”。'
            '小 N 时，四个单卡副本相对同节点 EP4 的收益更大，也必须进入候选集。'
            '尚未测量实例分流、状态迁移或布局切换开销。', '']
    for node in dict.fromkeys(r['node'] for r in means):
        rows = [r for r in means if r['node']==node]
        text += [f'## {node}：绝对延迟', '',
                 '| 全组 N | 四卡布局 | 平均 TPOT ms/token | 平均请求 E2E ms | 平均 TTFT ms | 整批完成 ms |',
                 '|---:|---|---:|---:|---:|---:|']
        for n in (8,16,32):
            for r in rows:
                if int(r['N']) != n: continue
                text.append(f'| {n} | {labels[r["layout"]]} | {float(r["mean_request_tpot_ms"]):.3f} | '
                            f'{float(r["mean_request_e2e_ms"]):.2f} | {float(r["mean_request_ttft_ms"]):.2f} | '
                            f'{float(r["cohort_makespan_ms"]):.2f} |')
        text.append('')
    text += ['## 同一指标内是否反转', '',
             '下表是候选相对同节点 EP4 的延迟降低百分比；正数表示候选更快，负数表示 EP4 更快。', '',
             '| 节点 | 候选 | 指标 | N8 | N16 | N32 | 是否换边 |',
             '|---|---|---|---:|---:|---:|---|']
    for node, run, layout in dict.fromkeys((r['node'],r['run'],r['layout']) for r in comparisons):
        for metric, label in [('mean_request_tpot_ms','平均 TPOT'), ('mean_request_e2e_ms','平均请求 E2E')]:
            values = [next(float(r['candidate_latency_reduction_percent']) for r in comparisons
                       if (r['node'],r['run'],r['layout'],r['metric'],int(r['N']))==(node,run,layout,metric,n))
                      for n in (8,16,32)]
            reversal = min(values)<0<max(values)
            ep_wins=[-v/(1-v/100) for v in values if v<0]
            strong = reversal and max(ep_wins)>=10 and max(values)>=10
            verdict = '双向均超过10%' if strong else ('有换边，幅度较小' if reversal else '未换边')
            text.append(f'| {node.removeprefix("spartan-gpgpu")} | {labels[layout]} | {label} | '
                        + ' | '.join(f'{v:+.1f}%' for v in values) + f' | {verdict} |')
    text += ['', '10% 只是本次预先记录的实用幅度门槛，不代表统计显著性。'
             '逐轮值及轮均值标准差保留在各运行的 CSV 中。', '',
             '## 固定条件与边界', '',
             '- 完整 Qwen3.6-35B-A3B BF16，40 层、256 个 routed experts、top-8。'
             '每条请求通过真实 Prefill 建立 P=4096 token 的 GPU KV/GDN 缓存，Q=1，O=128，忽略 EOS；'
             '每个请求严格检查 device cached_tokens=4096、完整输出 128。没有 CPU HiCache/KV 恢复。',
             '- 输入为固定 repeated 模板及每请求独立标记，贪心生成；不是在线生产流量或任务质量评估。',
             '- EP4 是 Attention DP4/TP1、MoE EP4、PP1；2×PP2 是两个独立实例，每实例两阶段、两微批；'
             '4×单卡是四个完整模型副本；PP4 为四阶段原生调度。',
             '- EP4 每 rank 的请求数为 N/4；2×PP2 每实例 N/2、每微批 N/4；'
             '固定请求分组与 EP4 对齐，等待合批保留在延迟里。'
             '新增 B2 通信形状保留原生预热 schema 检查；每个 PP2 点逐请求核对原生固定分组与 lookahead 的输出，'
             '并检查四个 stage 的 lookahead 执行及退休计数。',
             '- 所有布局使用同一小 token Expert Triton 内核家族、原生 Attention/GDN；'
             '完整模型和真实权重资格检查通过。实际 rank/stage batch 与 Graph 资格计数见各运行 forward_histograms.csv。',
             '- 两个节点各自顺序测候选布局；跨节点 EP4 不求平均，也不把跨节点数值当作因果对照。'
             '尚未做布局交替复测、其他上下文或其他模型；PP4 未应用仅支持 PP2 的 lookahead。',
             '- 这里是同一 Decode workload 类别内的结果。Q1 的 TTFT 不能代替冷 Prefill 性能；'
             '自然生成的路由随 batch/数值变化，不能把差异全部拆成通信和 Expert 的因果贡献。', '',
             '## 如何理解旧组件预测', '',
             '旧图的小 N 正预算，只表示省下的专家通信能否支付额外 Expert 计算；'
             '它假设流水充分填充，未计入阶段交接、反馈、调度等待和 Attention/GDN。'
             '旧图中相同的 EP/PP Expert 数值来自相同的合成微批 lookup，'
             '也没有独立测量 EP 每 rank 64 个常驻专家与 PP 单层 256 个常驻专家的实际执行。'
             '因此它是候选区域，不能视为 PP 整模型胜出的预测。', '',
             '## 数据与复现', '',
             '- [全部平均值](mean_metrics.csv) · [同节点比较](comparisons.csv) · [来源与校验](manifest.json)',
             '- 原始运行保留在实验目录；早期准备、失败和容量不合格的尝试不纳入本报告。'
             '处理详情见 [运行与资格审计](RUN_AUDIT.md)。', '']
    for audit in audits:
        text.append(f'- [{Path(audit["run"]).name}：逐轮与 batch 表]({Path(audit["run"]).name}/REPORT.md)；'
                    f'Guard postcheck、CPU 权重使用证据、拥有进程清理均通过。')
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    manifest = dict(experiment_id='qwen35b_h100_small_batch_layout_verify_20261029',
                    new_gpu_measurement=True, runs=audits, analyses=manifests,
                    statistic='arithmetic mean; independent node-local EP4 controls; every formal round retained',
                    files=[desc(out/n) for n in ('mean_metrics.csv','comparisons.csv','REPORT.md')])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(out), points=len(means), eligible_runs=len(audits))))


if __name__=='__main__':
    main()
