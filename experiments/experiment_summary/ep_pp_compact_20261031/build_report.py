"""Build a short, indexed reading guide from qualified saved measurements only."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT.parent
WORK = SUMMARY.parents[1]
AGENT = WORK / 'experiments/agentx_pd_dynamic_20261030'
MANAGER = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
REPORT = SUMMARY / 'EP_PP_EXPERIMENTS_COMPACT.md'
ID = 'moe_ep_pp_experiments_compact_20261031'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return list(csv.DictReader(path.open()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(values, **fields):
    selected = [r for r in values if all(r.get(k) == v for k, v in fields.items())]
    assert len(selected) == 1, (fields, len(selected))
    return selected[0]


def close(actual, expected):
    assert abs(float(actual) - float(expected)) < 1e-7, (actual, expected)


def main():
    now = datetime.now(ZoneInfo('Australia/Melbourne')).isoformat(timespec='seconds')
    fixed = SUMMARY / 'ep4_fixed_phase_comparison_20261030'
    fixed_manifest = read(fixed / 'manifest.json')
    source_paths = [Path(s['path']) for s in fixed_manifest['sources']]
    for source, path in zip(fixed_manifest['sources'], source_paths):
        assert sha(path) == source['sha256'], path
    decode, decode_rounds, mixed, mixed_rounds = map(rows, source_paths)
    fixed_rows = rows(fixed / 'metrics.csv')
    checked_rounds = 0
    for r in fixed_rows:
        if r['phase'] == 'fixed_decode':
            original = one(decode, run=r['run'], case=r['case'], layout=r['layout'])
            repeats = [x for x in decode_rounds if all(x[k] == original[k] for k in ['run', 'arm', 'case'])]
            fields = {'ttft_ms': 'mean_request_ttft_ms', 'tpot_ms': 'mean_request_tpot_ms',
                      'e2e_ms': 'mean_request_e2e_ms', 'makespan_ms': 'cohort_makespan_ms'}
        else:
            original = one(mixed, run=r['run'], case=r['case'], layout=r['layout'], route='least_outstanding')
            repeats = [x for x in mixed_rounds if all(x[k] == original[k] for k in ['run', 'layout', 'case', 'route'])]
            assert all(x['arrival_valid'] == 'True' for x in repeats)
            fields = {'ttft_ms': 'all_mean_ttft_ms', 'e2e_ms': 'all_mean_e2e_ms', 'makespan_ms': 'makespan_ms'}
        assert len(repeats) == int(r['rounds'])
        checked_rounds += len(repeats)
        for dest, source in fields.items():
            close(r[dest], statistics.mean(float(x[source]) for x in repeats))
    assert len(fixed_rows) == 143 and checked_rounds == 451

    baseline = AGENT / 'matrix_analysis'
    baseline_rows, baseline_trials = rows(baseline / 'means.csv'), rows(baseline / 'trials.csv')
    baseline_audit = read(baseline / 'audit.json')
    accepted = [r for r in baseline_audit['runs'] if r['run_accepted']]
    assert len(accepted) == 3 and sum(r['finished_episodes'] for r in accepted) == 18
    agent_rows = []
    for case in ['agentx_c8', 'agentx_c128']:
        for arm in ['col_ep4', 'col_pp2', 'col_single']:
            run = one(accepted, arm=arm)['run']
            r = one(baseline_rows, run=run, arm=arm, case=case, policy='sticky_hash')
            repeats = [x for x in baseline_trials if all(x[k] == r[k] for k in ['run', 'arm', 'case', 'policy'])]
            assert r['complete_point'] == 'True' and len(repeats) == int(r['repeats']) == 3
            for metric in ['ttft_ms', 'request_tpot_ms', 'e2e_ms', 'root_mean_s', 'wall_s']:
                close(r[metric], statistics.mean(float(x[metric]) for x in repeats))
            agent_rows.append(r)

    pressure = AGENT / 'e7_t300_b7200_analysis'
    pd = one(rows(pressure / 'means.csv'), arm='pd_ep4', case='agentx_c128')
    pd_trials = rows(pressure / 'trials.csv')
    assert pd['run_accepted'] == pd['complete_point'] == 'True' and len(pd_trials) == 3
    for metric in ['ttft_ms', 'request_tpot_ms', 'e2e_ms', 'root_mean_s', 'wall_s']:
        close(pd[metric], statistics.mean(float(x[metric]) for x in pd_trials))
    pd_run = AGENT / 'runs' / pd['run']
    progress, cleanup = read(pd_run / 'task/progress.json'), read(pd_run / 'task/cleanup.json')
    assert progress['stage'] == 'complete' and progress['valid_points'] == 3
    assert cleanup['passed'] and cleanup['gpu_idle']
    post = sorted((pd_run / 'dashboard_guard/checks').glob('*-post.json'))[-1]
    assert read(post)['passed']
    assert all(int(x['requests']) == 2736 for x in pd_trials)
    waits = rows(pd_run / 'analysis/pd_wait_episodes.csv')
    assert len(waits) == 3
    p_queue_s = statistics.mean(float(r['p_ready_queue_ms']) for r in waits) / 1000
    waiter = read(AGENT / 'wait_for_e7_v4_controls_r1.json')
    assert waiter['stage'] == 'cancelled_before_successor'
    v4_audit = read(AGENT / 'e7_v4_b7200_analysis/audit.json')
    assert not any(r['episodes'] for r in v4_audit['runs'])

    layouts = ['ep4', 'two_ep2', 'four_ep1', 'two_pp2']
    decode_table, prefill_table, exported = [], [], []
    for n in [8, 16, 32, 64]:
        group = [one(fixed_rows, phase='fixed_decode', node='spartan-gpgpu011',
                     case=f'p4096_n{n}_repeated', layout=layout) for layout in layouts]
        decode_table.append('| ' + str(n) + ' | ' + ' | '.join(f"{float(r['tpot_ms']):.3f}" for r in group) + ' |')
        exported.extend(dict(experiment='synthetic_decode', **r) for r in group)
    for case, label in [('cold_b32_q512', '32×512＝16,384'),
                        ('cold_b16_q8192', '16×8192＝131,072'),
                        ('cold_b8_q16384', '8×16384＝131,072')]:
        group = [one(fixed_rows, phase='cold_prefill', node='spartan-gpgpu011',
                     case=case, layout=layout) for layout in layouts]
        prefill_table.append('| ' + label + ' | ' + ' | '.join(f"{float(r['ttft_ms']):.2f}" for r in group) + ' |')
        exported.extend(dict(experiment='synthetic_prefill', **r) for r in group)
    names = {'col_ep4': '2×EP4', 'col_pp2': '4×PP2', 'col_single': '8×单卡'}
    agent_table = []
    for r in agent_rows:
        agent_table.append(f"| {r['case'].removeprefix('agentx_').upper()} | {names[r['arm']]} | "
                           f"{float(r['ttft_ms']):.2f} | {float(r['request_tpot_ms']):.3f} | "
                           f"{float(r['e2e_ms']):.2f} | {float(r['wall_s']):.2f} |")
        exported.append(dict(experiment='agentx_colocated_e5_v3', **r))
    exported.append(dict(experiment='agentx_pd_pressure_e7_v3', **pd))
    for r in exported:
        if r['experiment'].startswith('agentx_'):
            r['source'] = str(pressure / 'means.csv' if r['arm'] == 'pd_ep4' else baseline / 'means.csv')

    text = f'''# EP／PP 实验速览

更新：{now}（墨尔本）。[全部实验索引]({SUMMARY / 'README.md'})。本文只整理已有数据，没有启动新实验。

**目前最清楚的结论：固定 synthetic Decode 有“低并发单卡、高并发 EP4”的反转；AgentX 合并部署按平均请求 E2E 看，C8 偏单卡、C128 偏 PP2。P/D 分离和动态配置的净收益仍待同版本对照。**

| 需要关注的实验 | 用来回答什么 | 最短入口 |
|---|---|---|
| Synthetic 整模型 P／D | 同卡预算下，单卡、PP、EP 各适合什么工作区间？ | [固定阶段对照]({fixed / 'REPORT.md'}) |
| Expert／通信分项 | 计算粒度、专家数和通信域如何改变成本？ | [组件专题]({SUMMARY / 'ep_pp_tradeoff/REPORT.md'}) |
| AgentX 整模型 | 缓存、排队、分支和 P/D 干扰后，优势是否仍成立？ | [数据盘点与 benchmark 审计]({SUMMARY / 'agentx_data_audit_20261031/REPORT.md'}) · [详细结果]({SUMMARY / 'agentx_pd_dynamic_20261030/REPORT.md'}) |

口径：`P`＝每请求旧缓存，`Q`＝新增输入，`O`＝输出；`N`＝synthetic 全组请求数；`C`＝AgentX 根会话并发上限；`T`＝单卡收到的 token–expert 行数。**N、C、T 不能互换。** 下列整模型表均为三轮算术平均、绝对时间；组件源图保留单次事件及全部样本。

## 1. Synthetic：用来确定部署边界

主对照：完整 Qwen3.6-35B-A3B，BF16 权重／KV，FP32 GDN；每种布局固定单节点四张 H100。EP 的 Attention 为 TP1、对应 DP。下面只列 011 节点，不混合不同节点均值。

**Decode：固定并发、GPU 热缓存。** 每请求先建立 P4096，再以 Q1 生成 O128；重复文本带独立标识，忽略 EOS。旧 Prefill 不计时，HiCache 关闭，无 CPU 恢复，也无持续 Prefill 到达。N8 时四单卡各承担 2 请求，N64 时各承担 16 请求。

| 全组 N | EP4 TPOT ms | 2×EP2 | 4×单卡 | 2×优化 PP2 |
|---:|---:|---:|---:|---:|
{chr(10).join(decode_table)}

低并发单卡更快，N32 接近，N64 偏 EP4；N8／N64 在另一节点五轮反向复测保留方向。完整扫描涵盖 P256–131072、N8–128，PP2 在 16 个工况中均未取得最低平均 TPOT。128K／N32 时单卡容量不足，PP2 整请求 E2E 2181.57 ms、EP4 2284.38 ms，但 PP2 的 TPOT 略慢。[完整 Decode 与容量审计]({SUMMARY / 'decode_pool_necessity_20261030/REPORT.md'})

**Prefill：冷输入、O1。** P0，无缓存命中；下表为请求平均 TTFT，越低越好。PP2 使用可变批次原生 policy0，与上表固定 Decode 的优化 policy9 分开。

| N×每请求 Q＝总新输入 token | EP4 TTFT ms | 2×EP2 | 4×单卡 | 2×PP2 |
|---|---:|---:|---:|---:|
{chr(10).join(prefill_table)}

三档均为 PP2 平均首 token 最快、单卡副本整批完成最快。例如 N16/Q8192，整批完成为单卡 706.53 ms、PP2 721.86 ms、EP4 1030.50 ms。首 token 与整批时间是两个目标。

另外保留两组背景数据：

- **Synthetic P/D 混合到达：** 64 请求，Q8192/O16 与 Q256/O256 按三种比例、16／32 请求每秒混合；各请求在同一实例完成 P 和 D。两节点各六档中四单卡平均 E2E 均最低，尚无布局混合的独有收益。[混合到达报告]({SUMMARY / 'pd_mix_pool_20261030/REPORT.md'})
- **HiCache 自然／全 CPU 重放：** 35B 四卡五布局、P8256／130304，缓存起点会改变排序；两卡组拆分显示长前缀下单卡副本可能明显落后于 EP2／PP2。它与上面的纯 GPU 热 Decode 分开比较。[缓存实验及两卡拆分]({MANAGER / 'qwen35b_h100_five_layouts_hicache_20261028/two_gpu_three_layouts_analysis/REPORT.md'})

GLM／HY3／122B、A100／H100 的早期 EP8↔PP2×EP4、EP2↔PP2 结果继续保存在[总索引历史表]({SUMMARY / 'README.md'})，按原模型、精度、缓存和通信路径查看；本表的 35B 边界不直接外推。

## 2. 分项分析：用来解释，暂不用于确定切换阈值

| 数据 | 已有覆盖与用途 | 关键边界／入口 |
|---|---|---|
| 单层 Expert 二维图 | GLM-4.7、HY3、35B、122B；单卡 T×活跃专家数；两份独立测量，1584 格，每格 30 次事件 | 权重已驻留，含 GEMM、激活与 padding；不含 Attention、通信、CPU 权重加载。[数据]({MANAGER / 'h100_four_model_expert_surface_20261029/REPORT.md'}) |
| EP 通信二维图 | 四模型激活尺寸；EP2／4／8×全组原始 token 8–32768；624 条件，每点 30 事件 | BF16、DeepEP LL；EP2／4 节点内，EP8 跨两节点；超过 2048 token 分块，无计算 overlap。[数据]({MANAGER / 'h100_ep_token_comm_surface_20261029/REPORT.md'}) |
| 计算＋通信推导 | 估算“少通信时间－多算 Expert 时间”，定位待验证区域 | 旧图微批归一化曾让两侧复用同一 Expert 格；已核对，不能把预算正负视作实测胜负。[口径修正]({SUMMARY / 'ep_pp_tradeoff/REPORT.md'}) |

最重要的限制：35B 的 EP4 每卡应驻留 64 个专家，PP 负责层驻留全部 256 个；原 Expert 扫描始终驻留 256 个，尚缺真实分片对照。还缺实际路由、PP 交接／流水等待和暴露通信的联合测量。GLM／HY3 部署候选按最小 EP4 处理。

## 3. AgentX：用来检验实际回放中的端到端收益

[已有数据盘点与 benchmark 审计]({SUMMARY / 'agentx_data_audit_20261031/REPORT.md'}) 已重新核对主表21轮共34,371调用；另有7个历史点、21轮不重复测量（包括早期EP4混部／PD C2／C8的12轮）。早期480调用片段和仅准备的V4也独立索引，完整列出输入构造、回放方法、输出差异和协议限制。

完整 35B BF16，两个四卡 H100 节点、共 8 GPU；**各 rank 开 HiCache，CPU 32 decimal GB／rank，GPU KV 按 0.95 原生自动分配，会话内保留缓存。** 选取 8 个 AgentX 家族共 171 次调用；C128 用 16 份缓存隔离副本，共 2736 次调用。输入按 hash 重建前缀、保留长度，属于派生回放。

**已验收的合并部署基线：18 轮，E5 源码＋V3 回放。** P、D 在同一实例完成。以下 TPOT 为请求平均，延迟越低越好。

| 根会话并发 | 8 GPU 部署 | TTFT ms | TPOT ms/token | 请求 E2E ms | 压缩整轮 s |
|---|---|---:|---:|---:|---:|
{chr(10).join(agent_table)}

按平均 E2E，C8 单卡最好、C128 PP2 最好。C128 单卡虽 TPOT 最低，但 TTFT 很高；不能只按 Decode 速度选择布局。严格 SLO 下 EP4 仍有优势点，见[完整 SLO 与均值]({baseline / 'REPORT.md'})。

这些部署的缓存容量不同：GPU KV 总容量约单卡 271 万、PP2 1508 万、EP4 1781 万 token；单卡 GDN 预算 40、EP／PP 为完整模型等效 80。C128 的本地纯 Decode batch 均值仅约 EP4 5.62、PP2 5.04、单卡 2.21，不能把 C128 当成单步 N128。

**新完成的独立 P/D 点：E7 源码＋V3 回放，4P4D、P=EP4→D=EP4、C128。** 三轮共 8208 调用，完整验收及清理通过；GDN80、receive8，交接等待上限 300 s、整轮上限 7200 s。

| 配置 | TTFT ms | TPOT ms/token | 请求 E2E ms | 压缩整轮 s |
|---|---:|---:|---:|---:|
| EP4 P→EP4 D | {float(pd['ttft_ms']):.2f} | {float(pd['request_tpot_ms']):.3f} | {float(pd['e2e_ms']):.2f} | {float(pd['wall_s']):.2f} |

该点请求平均 P 可运行队列等待 {p_queue_s:.2f} s，首 token 等待很长；P/D 等待有重叠，不相加，也不当纯网络时间。它验证了修复后能完整跑通，但缺同源码对照，不能与上表直接作受控性能归因。[三轮结果]({pressure / 'REPORT.md'}) · [等待分解]({pd_run / 'analysis/PD_WAITS.md'})

**协议边界：** V3 有空闲压缩和分支时序限制；后续 V4 修正分支时序，须独立比较。当前尚无 V4 GPU 结果，原 24 轮交错队列已取消、准备文件保留。当前数据不是官方 AgentX 分数；不同布局的计时输出也未做到全部逐 token 相同。[协议与输出范围]({SUMMARY / 'agentx_pd_dynamic_20261030/REPORT.md'})

## 4. 接下来只关注这三个问题

1. **合并部署的最好基线是什么？** 在同一 V4 回放、同一 8 GPU 预算下，先补全单卡、PP2、EP4 等固定配置；保留缓存容量、实际 batch、TTFT、TPOT、E2E 和任务完成时间。
2. **P/D 分离本身是否有净收益？** 再比较 P:D=6:2、4:4、2:6，每个配比内 P 侧保留单卡／PP2；D 侧先比较单卡副本，具备四卡时加入 EP4，长缓存保留 PP2／EP2 容量对照。均为待验证候选，单卡 D 尚无正式 AgentX 结果。
3. **是否需要动态配比？** 只有固定配置在同一目标下互有胜负，再测动态路由／配比相对最佳固定合并、固定 P/D、仅调整 P:D 的额外收益，计入缓存、排队、batch 分流与迁移成本。

后续新增结果优先更新本文对应表格，细节留在专题报告。[本文数据及来源清单]({ROOT / 'manifest.json'}) · [重建脚本]({ROOT / 'build_report.py'})
'''
    REPORT.write_text(text)
    keys = list(dict.fromkeys(k for r in exported for k in r))
    with (ROOT / 'full_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(exported)
    paths = source_paths + [fixed / 'metrics.csv', fixed / 'manifest.json',
        baseline / 'means.csv', baseline / 'trials.csv', baseline / 'audit.json',
        pressure / 'means.csv', pressure / 'trials.csv', pressure / 'audit.json',
        pd_run / 'task/progress.json', pd_run / 'task/cleanup.json', post,
        pd_run / 'analysis/pd_wait_episodes.csv',
        AGENT / 'wait_for_e7_v4_controls_r1.json', AGENT / 'e7_v4_b7200_analysis/audit.json',
        SUMMARY / 'agentx_pd_dynamic_20261030/REPORT.md',
        MANAGER / 'h100_four_model_expert_surface_20261029/REPORT.md',
        MANAGER / 'h100_ep_token_comm_surface_20261029/REPORT.md',
        SUMMARY / 'ep_pp_tradeoff/REPORT.md']
    provenance = [AGENT / 'runs' / r['run'] / 'launch_manifest.json' for r in accepted]
    provenance += [pd_run / 'launch_manifest.json',
        MANAGER / 'h100_four_model_expert_surface_20261029/manifest.json',
        MANAGER / 'h100_ep_token_comm_surface_20261029/manifest.json']
    paths += provenance
    previous = read(ROOT / 'manifest.json') if (ROOT / 'manifest.json').exists() else {}
    manifest = dict(id=ID, title='EP/PP compact experiment guide', created_at=now,
        updated_at=now,
        status='complete_snapshot', research_status='PD benefit and dynamic allocation remain unproven',
        source_report=str(REPORT), report_sha256=sha(REPORT), new_gpu_measurements=False,
        implementation='CPU-only aggregation; exact serving revisions remain in upstream manifests',
        deployment='No server launch, booking, gateway change or new endpoint',
        workloads=['Fixed synthetic GPU-hot Decode', 'Cold synthetic Prefill',
                   'Expert and DeepEP component scans', 'AgentX-derived E5/V3 and E7/V3, separate cohorts'],
        models=['Qwen3.6-35B-A3B BF16 full model',
                'GLM AWQ, HY3 BF16, 35B BF16, 122B FP8 Expert components; BF16 communication'],
        testbed='H100 80GB; synthetic whole-model candidates use four GPUs; AgentX uses two four-GPU nodes',
        source_mean_records_rechecked=143, source_round_records_rechecked=checked_rounds,
        accepted_agentx_colocated_episodes=18, accepted_agentx_e7_pd_episodes=3,
        v4_gpu_episodes=0, exported_records=len(exported),
        comparison_boundaries=['Do not pool nodes, source versions, replay versions or cache protocols',
            'N, root concurrency C, local Decode batch and routed rows T differ',
            'Component original-event plots do not establish whole-model break-even',
            'Current P/D point lacks a matching-source colocated control'],
        sources=[dict(path=str(p), sha256=sha(p)) for p in paths],
        upstream_manifest_paths=[str(p) for p in provenance],
        full_metrics_sha256=sha(ROOT / 'full_metrics.csv'))
    manifest['created_at'] = previous.get('created_at', now)
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for link in re.findall(r'\]\(([^)]+)\)', text):
        assert Path(link).exists(), link
    snapshot = MANAGER / ID
    snapshot.mkdir(exist_ok=True)
    for name in ['manifest.json', 'full_metrics.csv', 'build_report.py']:
        shutil.copyfile(ROOT / name, snapshot / name)
    shutil.copyfile(REPORT, snapshot / 'summary.md')
    registry = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    start, end = f'<!-- begin {ID} -->', f'<!-- end {ID} -->'
    entry = f'''{start}
### {ID}

Compact Markdown entry: {REPORT}

Snapshot: {snapshot}

- CPU-only reading guide for synthetic whole-model Prefill/Decode, four-model Expert and EP communication components, and AgentX-derived E2E. No new GPU measurement or serving launch.
- Rechecked143 saved mean records against451 formal rounds; selected tables retain node/cache/PP-policy boundaries. Decode P4096/N8 favors single replicas;N64 favorsEP4. Cold Prefill PP2 TTFT versus single-replica makespan remains a metric tradeoff.
- AgentX contains18 accepted E5/V3 colocated episodes and3 accepted E7/V3 EP4 4P4D C128 episodes, kept as separate source cohorts; no matched-source PD benefit or dynamic-allocation claim. V4 has0 GPU episodes; old automatic queue cancelled before successor admission, candidate order being reorganized.
- Canonical summary, selected absolute metrics, source hashes and rebuild script. Raw logs and upstream immutable launch manifests remain at their original paths.

{end}'''
    content = registry.read_text()
    if start in content:
        assert content.count(start) == content.count(end) == 1
        left, rest = content.split(start, 1)
        _, right = rest.split(end, 1)
        content = left + entry + right
    else:
        content += '\n\n' + entry + '\n'
    registry.write_text(content)
    print(json.dumps(dict(report=str(REPORT), lines=len(text.splitlines()),
                         checked_mean_records=143, checked_round_records=checked_rounds,
                         accepted_agentx_episodes=21, snapshot=str(snapshot))))


if __name__ == '__main__':
    main()
