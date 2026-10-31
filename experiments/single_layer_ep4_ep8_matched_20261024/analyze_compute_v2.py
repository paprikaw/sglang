"""CPU-only analysis of authenticated, real single-layer Expert measurements."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

BASE = Path(__file__).resolve().parent
OUT = BASE / 'compute_v2'
TOKENS = [2**n for n in range(3, 13)]
RUNS = [('glm', 'run_a', 36), ('glm', 'run_b', 36), ('hy3', 'run_a', 112)]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    flat, samples, evidence, pool = [], [], [], {}
    for model, run_id, expected in RUNS:
        run = OUT / model / run_id
        source = run / 'profile/result.json'
        value = json.loads(source.read_text())
        cleanup = json.loads((run / 'profile/task/cleanup.json').read_text())
        assert value['passed'] is True and len(value['rows']) == expected
        assert cleanup['passed'] is True and cleanup['gpu_idle'] is True
        posts=list((run/'dashboard_guard/checks').glob('*-post.json'))
        assert posts and all(json.loads(p.read_text()).get('scientific_success') is True for p in posts)
        assert value['actual_world_size'] == 1 and value['timed_communication'] is False
        assert value['numerical'] and all(n['passed'] for n in value['numerical'])
        spec = json.loads((run / 'dashboard_spec.json').read_text())
        assert all(digest(i['path']) == i['sha256'] for i in spec['frozen_inputs'])
        evidence.append(dict(model=model, run=run_id, source=str(source),
                             sha256=digest(source), cleanup_sha256=digest(run/'profile/task/cleanup.json'),
                             postchecks=[dict(path=str(p),sha256=digest(p)) for p in posts],
                             rows=expected, numerical_checks=len(value['numerical']),
                             node=value['authorization']['node'], gpu=value['gpu'], backend=value['backend']))
        for r in value['rows']:
            total = r['global_unique_tokens']
            cohort, mode = r['cohort'], r['mode']
            raw = r.get('cuda_event_ms', r.get('raw_ms'))
            assert len(raw) == 30 and all(t > 0 for t in raw)
            expected_assignments = total * (2 if cohort == 'equal_call' else 1)
            assert sum(r['counts']) == r['local_assignments'] == expected_assignments
            assert math.isclose(statistics.median(raw),r['timing']['median_ms'],rel_tol=1e-12)
            row = dict(model=model, run=run_id, timing_mode=mode, whole_batch_unique_tokens=total,
                       unique_tokens_in_this_call=r['microbatch_logical_tokens'],
                       comparison_case=cohort, emulated_ep_size=r['ep_size'],
                       resident_experts=r['local_experts'], active_experts=r['active_experts'],
                       actual_token_expert_pairs_on_this_gpu=r['local_assignments'],
                       median_ms=r['timing']['median_ms'], min_ms=min(raw), max_ms=max(raw),
                       samples=30, route_sha256=r['route_sha256'], backend=value['backend'],
                       source=str(source), source_sha256=digest(source))
            flat.append(row)
            pool[model, run_id, total, cohort, mode] = row
            for i, ms in enumerate(raw):
                samples.append(dict(model=model, run=run_id, whole_batch_unique_tokens=total,
                                    comparison_case=cohort, timing_mode=mode, sample=i, cuda_event_ms=ms))

    # Primary GLM curve: first run through2048; second run only adds4096.
    # All overlapping measurements remain visible as independent drift controls.
    primary, drift = [], []
    for model in ['glm', 'hy3']:
        for total in TOKENS:
            run = 'run_b' if model == 'glm' and total == 4096 else 'run_a'
            rows = {c: pool[model, run, total, c, 'eager'] for c in ['whole','half0','half1','equal_call']}
            ep8 = rows['whole']['median_ms']
            half0, half1 = [rows[c]['median_ms'] for c in ['half0','half1']]
            half = (half0 + half1) / 2
            equal = rows['equal_call']['median_ms']
            primary.append(dict(model=model, run=run, whole_batch_unique_tokens=total,
                                ep8_call_unique_tokens=total, ep4_half_call_unique_tokens=total//2,
                                actual_pairs_per_gpu_equal_batch=total,
                                ep8_active_experts=rows['whole']['active_experts'],
                                ep4_half0_active_experts=rows['half0']['active_experts'],
                                ep4_half1_active_experts=rows['half1']['active_experts'],
                                ep8_whole_ms=ep8, ep4_half0_ms=half0, ep4_half1_ms=half1,
                                ep4_mean_half_ms=half, half_vs_ep8_ratio=half/ep8,
                                extra_half_compute_ms=half-ep8,
                                ep4_equal_call_unique_tokens=total,
                                ep4_equal_call_pairs_per_gpu=2*total,
                                ep4_equal_call_ms=equal, equal_call_vs_ep8_ratio=equal/ep8))
    for total in TOKENS[1:-1]:
        for c in ['whole','half0','half1','equal_call']:
            a,b = [pool['glm',run,total,c,'eager']['median_ms'] for run in ['run_a','run_b']]
            drift.append(dict(whole_batch_unique_tokens=total,comparison_case=c,
                              first_run_ms=a,second_run_ms=b,second_minus_first_percent=100*(b/a-1)))
    csv_write(OUT/'expert_all_rows.csv', flat)
    csv_write(OUT/'expert_samples.csv', samples)
    csv_write(OUT/'expert_comparison.csv', primary)
    csv_write(OUT/'glm_repeat_controls.csv', drift)
    validation = dict(passed=True, primary_points=len(primary), raw_rows=len(flat),
                      raw_samples=len(samples), evidence=evidence,
                      numerical_checks=sum(e['numerical_checks'] for e in evidence),
                      scope='one actual GPU; representative destination owner0; no timed communication',
                      primary_mode='eager', source_copies=1,
                      input='same BF16 deterministic activation formula and routing as ce_rdma_r1 iteration0')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')

    text = ['# E06：A100 单层 Expert，EP8 整批与 EP4 半批／同次输入对照', '',
            '只加载真实的一层专家权重，不加载完整模型。两个模型均在065节点GPU0顺序执行。', '',
            '## 测量边界', '',
            '- GLM-4.7-AWQ：第3层，Humming W4A16 GROUPED_MASKED；EP8每卡20个常驻专家，EP4每卡40个。',
            '- Hy3-BF16：第1层，Triton BF16专家核心；EP8每卡24个常驻专家，EP4每卡48个。',
            '- top-8均衡确定性路由。输入激活是合成BF16，权重来自真实检查点；不是语义请求。只扫描token数，不额外扫描活跃专家数。',
            '- 相同整批：EP8一次处理整批；EP4分别处理前后两个半批。表中EP4取两半耗时中位数的平均值，**没有把单层实验称为完整PP测量**。',
            '- 相同单次输入：EP4和EP8每次处理相同数量的唯一token，EP4每卡实际token—expert配对行数为EP8的两倍。',
            '- 路由、排序、复制输入、权重加载、通信、Attention、KV缓存和流水调度均不计时；计时包含专家FFN的实际核心调用。',
            '- 主表使用预热后的eager CUDA事件跨度，每行3个交替区块、每区块10次。事件跨度不是硬件计数器测得的纯GPU忙碌时间。Hy3另有Graph控制，不能替代eager主表。',
            '- GLM分两次运行覆盖末端4096点；重叠点保留为复测控制。表中8—2048来自第一轮，4096来自第二轮，不合并原始样本伪造同进程统计。',
            '- 每种布局只在一张GPU模拟一个代表性目的rank的计算形状，不含真实EP8/EP4跨卡执行或最慢rank。权重反复调用保持预热状态，不代表冷HBM流量实测。', '',
            'GLM和Hy3的形状、精度、内核均不同；它们的差异不能只归因于权重大小。活跃专家数由固定路由自然决定，小token点不保证所有常驻专家都被命中。', '',
            '## 全部主结果', '',
            '延迟单位：毫秒，每次单层专家调用。最后一列不等工作量，不能作为PP吞吐直接对比。', '',
            '| 模型 | 整批唯一token | EP8整批 | EP4前半批 | EP4后半批 | EP4半批均值／EP8 | EP4相同单次输入 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for r in primary:
        text.append(f"| {r['model']} | {r['whole_batch_unique_tokens']} | {r['ep8_whole_ms']:.4f} | {r['ep4_half0_ms']:.4f} | {r['ep4_half1_ms']:.4f} | {r['half_vs_ep8_ratio']:.3f}× | {r['ep4_equal_call_ms']:.4f} |")
    glm128=next(r for r in primary if r['model']=='glm' and r['whole_batch_unique_tokens']==128)
    hy128=next(r for r in primary if r['model']=='hy3' and r['whole_batch_unique_tokens']==128)
    text += ['', '## 可以直接读出的现象', '',
             f"1. 同样每卡128配对行，EP4半批比EP8整批多用：GLM {glm128['extra_half_compute_ms']:.4f} 毫秒（{100*(glm128['half_vs_ep8_ratio']-1):.1f}%），Hy3 {hy128['extra_half_compute_ms']:.4f} 毫秒（{100*(hy128['half_vs_ep8_ratio']-1):.1f}%）。每卡配对数相同，并不保证延迟相同。", '',
             '2. Hy3在整批64—1024 token时，EP4半批约为EP8的1.82—1.86倍；4096 token时缩小至1.35倍。较大输入开始改善每专家行数、权重复用和有效计算效率，但未测HBM/SM计数器，不能定量拆出各原因。', '',
             '3. 平台确实存在：Hy3的EP4半批从整批64到1024 token，配对行数增加16倍，耗时只从1.4075增至1.5465毫秒；GLM的EP4半批从64到256 token只从0.3428增至0.3538毫秒。平台不是覆盖所有token范围的统一线性函数，分块/内核选择会改变台阶。', '',
             '4. 很小的整批8/16 token时，两种布局每卡实际命中的专家数都只有8/16；不能用这些点断言“放更多常驻专家一定更慢”。GLM小于约4%的差异与跨轮变化同量级，不作稳定胜负结论。', '',
             f"5. GLM跨轮重叠控制的延迟变化中位数为{statistics.median(r['second_minus_first_percent'] for r in drift):.2f}%，范围{min(r['second_minus_first_percent'] for r in drift):.2f}%到{max(r['second_minus_first_percent'] for r in drift):.2f}%；没有把两轮样本混为一次统计。", '',
             '6. Hy3日志提示缺少24/48专家的A100专用Triton调优表，本轮走当前默认配置。因此这些数字能说明当前部署路径的代价，不能证明BF16专家的硬件性能上限。', '']
    text += ['', f"正确性通过：{validation['numerical_checks']}项独立数值比较；{len(flat)}行计时、{len(samples)}个原始样本。冻结输入及各轮GPU清理核验通过。", '',
             '[完整比较表](expert_comparison.csv) · [全部计时行](expert_all_rows.csv) · [原始样本](expert_samples.csv) · [GLM跨轮控制](glm_repeat_controls.csv) · [验收证据](validation.json)', '',
             '![两种比较方式](expert_curves.png)', '',
             '## 如何与通信表一起使用', '',
             '与T01使用相同token编号、均衡路由和BF16输入公式，可以对齐每卡实际配对行数，比较“缩短通信所省时间”和“每卡增多专家的额外计算”。',
             '但两部分是独立调用的微基准，通信表按多rank对齐取最大值，本文为单GPU代表rank；没有测量通信与计算重叠、PP边界、完整层关键路径或请求TPOT。因此两列相加只能作为筛选候选的估算，不能称为完整PP性能。',
             'CE／RDMA是通信策略，不改变本次独立Expert核心；无需分别重跑同一个GEMM来制造两份计算数字。', '']
    (OUT/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(validation,indent=2))


if __name__ == '__main__':
    main()
