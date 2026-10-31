"""Export auditable lightweight data and a report; never copy private configs."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'deliverables'
ID = 'qwen35b_a100_h100_pp_ep_steady_decode_20261026'
ARCHIVE = ROOT.parents[4] / 'experiment_manager' / 'experiments' / ID


def read_csv(path):
    return list(csv.DictReader(path.open()))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    data_dir = OUT / 'data'
    source_dir = OUT / 'reproduce'
    evidence_dir = OUT / 'evidence'
    for directory in [data_dir, source_dir, evidence_dir]:
        directory.mkdir(exist_ok=True)
    metrics, components, ranks, requests, rounds, evidence = [], [], [], [], [], []
    source_files = []
    for name in ['a100_r2', 'h100_r2']:
        run = ROOT / 'runs' / name
        manifest = json.loads((run / 'launch_manifest.json').read_text())
        for adapter in manifest['private_adapters']:
            adapter_path = Path(adapter['path'])
            assert hashlib.sha256(adapter_path.read_bytes()).hexdigest() == adapter['sha256']
            source_files.append(adapter_path)
        source_files.append(run / 'config.json')  # hash only; contents never enter the package
        hw = manifest['hardware']
        validation = json.loads((run / 'analysis/validation.json').read_text())
        assert validation['passed'] and not validation['partial']
        metrics.extend(read_csv(run / 'analysis/native_metrics.csv'))
        components.extend(read_csv(run / 'analysis/components.csv'))
        ranks.extend(read_csv(run / 'analysis/components_per_rank.csv'))
        post_path = max((run / 'dashboard_guard/checks').glob('*-post.json'))
        post = json.loads(post_path.read_text())
        assert post['passed'] and post['cleanup_verified'] and post['model_cache']['passed']
        # Snapshot scientific evidence without lease tokens, admin keys or private configs.
        item = dict(run=name, hardware=hw, postcheck_path=str(post_path),
            postcheck_sha256=hashlib.sha256(post_path.read_bytes()).hexdigest(),
            passed=post['passed'], exit_code=post['exit_code'],
            cleanup_verified=post['cleanup_verified'], model_cache_passed=post['model_cache']['passed'],
            exact_ranks_cache_checked=True, launch_manifest=str(run/'launch_manifest.json'))
        # The run-local acquisition receipt is intentionally immutable. Query the
        # shared registry for the final state, exporting only non-secret fields.
        lease = json.loads((run/'dashboard_guard/gpu_registry_lease.json').read_text())
        registry_path = Path(lease['registry_dir'])/'registry.json'
        registry = json.loads(registry_path.read_text())
        def find_lease(value):
            if isinstance(value, dict):
                if value.get('lease_id') == lease['lease_id']:
                    return value
                for child in value.values():
                    found = find_lease(child)
                    if found is not None:
                        return found
            elif isinstance(value, list):
                for child in value:
                    found = find_lease(child)
                    if found is not None:
                        return found
            return None
        final_lease = find_lease(registry)
        assert final_lease is not None and final_lease['state'] == 'released'
        item['final_gpu_lease'] = dict(lease_id=lease['lease_id'], state='released',
            registry=str(registry_path), checked_at=datetime.now(timezone.utc).isoformat())
        for arm in ['ep2', 'pp2']:
            cleanup = json.loads((run / arm / 'cleanup.json').read_text())
            assert cleanup['cleanup_verified'] and cleanup['process_groups_exited']
            item[arm+'_cleanup'] = cleanup
            raw_path = run / arm / 'metrics.decode.json'
            raw = json.loads(raw_path.read_text())
            assert raw['valid'] and not raw['profiler_enabled'] and len(raw['rounds']) == 3
            for number, group in enumerate(raw['rounds'], 1):
                rounds.append(dict(hardware=hw, arm=arm, round=number,
                    output_tokens_per_s=group['output_tokens_per_s'], makespan_s=group['makespan_s'],
                    requests=len(group['requests']), total_output_tokens=1024))
                for index, request in enumerate(group['requests'], 1):
                    requests.append(dict(hardware=hw, arm=arm, round=number, request=index,
                        **{k:request[k] for k in ['cached_tokens', 'prompt_tokens', 'output_tokens',
                            'ttft_s', 'tpot_ms', 'e2e_s']}))
            source_files.append(raw_path)
        evidence.append(item)
        destination = evidence_dir / name
        destination.mkdir(exist_ok=True)
        for arm in ['ep2', 'pp2']:
            qualification_dir = destination / 'qualification' / arm
            qualification_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted((run/arm/'qualification').glob('*.json')):
                qualification = json.loads(path.read_text())
                assert qualification['passed']
                shutil.copy2(path, qualification_dir/path.name)
                source_files.append(path)
        for rel in ['launch_manifest.json', 'topology.txt', 'analysis/validation.json']:
            shutil.copy2(run / rel, destination / Path(rel).name)
            source_files.append(run / rel)
        save_json(destination / 'guard_and_cleanup.json', item)
        for rel in ['analysis/native_metrics.csv', 'analysis/components.csv', 'analysis/components_per_rank.csv']:
            source_files.append(run / rel)
    assert len(metrics) == 4 and len(components) == 6 and len(ranks) == 36
    assert len(requests) == 96 and len(rounds) == 12
    assert sum(int(row['selected_forwards']) for row in ranks) == 864
    assert sum(int(row['actual_layer_invocations']) for row in ranks) == 25920
    save_csv(data_dir / 'native_metrics.csv', metrics)
    save_csv(data_dir / 'components.csv', components)
    save_csv(data_dir / 'components_per_rank.csv', ranks)
    save_csv(data_dir / 'requests.csv', requests)
    save_csv(data_dir / 'rounds.csv', rounds)
    deck_data = json.loads((OUT / 'deck_data.json').read_text())
    for key in ['history', 'prefill', 'workload', 'long_context', 'attention_interference']:
        save_csv(data_dir / (key+'.csv'), deck_data[key])
    source_files.extend(Path(row['path']) for row in json.loads((OUT / 'sources.json').read_text()))
    source_files = sorted(set(source_files))
    save_json(data_dir / 'source_hashes.json', [dict(path=str(p), bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_files])
    protocol = dict(model='Qwen3.6-35B-A3B', revision='995ad96eacd98c81ed38be0c5b274b04031597b0',
        precision='BF16', layers=40, routed_experts=256, top_k=8,
        source_commit='870269702285d94bdd926a50cc29ad1a20bcf60c',
        source_tag='experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated',
        frozen_engine_worktree=str(ROOT.parents[1]/'.engine-worktrees/glm47-ep16-canonical-profile-20261004'),
        actual_gpu_pair=[0,1], lease_gpus_per_node=4, arms_sequential_on_each_node=True,
        A100=dict(node='spartan-gpgpu069', job='33463295', pair_topology='NV12',
            device='NVIDIA A100 80GB PCIe',
            expert_backend='Triton', attention_backend='Triton', graph=False, overlap=False,
            deep_ep='normal BF16 bridge, actual-weight numerical validation against independent all-gather'),
        H100=dict(node='spartan-gpgpu178', job='33646188', pair_topology='NV6',
            device='NVIDIA H100 80GB HBM3',
            expert_backend='BF16 DeepGEMM', attention_backend='FA3 full / Triton linear', graph=True,
            ep_overlap=True, pp_overlap=False, pp_ready_send_policy=False),
        EP2=dict(attention_tp=1, attention_dp=2, expert_ep=2, layers_per_rank=40, experts_per_layer_per_rank=128),
        PP2=dict(attention_tp=1, expert_ep=1, pipeline_stages=2, layers_per_rank=20,
            experts_per_layer_per_rank=256, pipeline_slots=2, max_requests_per_microbatch=4),
        workload=dict(requests=8, concurrent=True, cached_context_each=4096, new_input_each=1,
            prompt_tokens_each=4097, output_tokens_each=128, old_cached_total=32768,
            new_input_total=8, output_total_per_round=1024,
            cache='real model-produced native GPU prefix plus GDN state; no CPU checkpoint restore',
            prompts='common semantic prompt fingerprints across platforms and arms'),
        timing=dict(rounds=3, profiler=False, excludes=['weight load','JIT','prefix preparation','shape warmup','profiles'],
            throughput='mean of 3 cohort rates; error = sample standard deviation of 3 rates',
            latency='per-request TTFT / TPOT / e2e; each cell 24 requests'),
        profiling=dict(independent_profiles=3, ranks_per_profile=2,
            scheduler_forwards_requested=64, actual_local_batch=4, initial_decode_forwards_discarded=4,
            selected_actual_forwards_per_rank=24, normalized_by='actual 40 or 20 layer invocations',
            error='sample standard deviation across 3 profile means, after averaging two ranks',
            H100_eager_diagnostic='3 extra profiles per arm; verified no GraphLaunch; never used for native performance',
            scopes='A100 native/H100 eager include Attention projections; H100 native Attention core and routed GEMM only',
            pp_boundary='GPU NCCL residency in selected CPU window divided by selected forwards; includes waits and overlap'),
        limitations=['different native stacks, not hardware-only causality','35B BF16 versus 122B FP8 is not size-only causality',
            'GPU components overlap and are not additive exclusive TPOT attribution',
            'fixed EP then PP order; no randomized order or long-duration confidence interval',
            'no inter-node EP/PP deployment threshold measured',
            'H100 shape warmup expanded in r2; prior r1 retained, not cherry-picked rounds'])
    save_json(data_dir / 'protocol.json', protocol)
    mm = {(r['hardware'],r['arm']):r for r in metrics}
    cc = {(r['hardware'],r['arm'],r['execution']):r for r in components}
    def num(row, key): return float(row.get(key) or 0)
    def change(hw, field, execution='native'):
        return 100*(num(cc[(hw,'pp2',execution)],field)/num(cc[(hw,'ep2',execution)],field)-1)
    a_reduction = 100*(1-num(mm[('A100','pp2')],'tpot_ms')/num(mm[('A100','ep2')],'tpot_ms'))
    h_reduction = 100*(1-num(mm[('H100','ep2')],'tpot_ms')/num(mm[('H100','pp2')],'tpot_ms'))
    table = '\n'.join(f"| {r['hardware']} | {r['arm'].upper()} | {num(r,'output_tokens_per_s'):.2f} ± {num(r,'throughput_stdev'):.2f} | {num(r,'throughput_min'):.2f}–{num(r,'throughput_max'):.2f} | {num(r,'tpot_ms'):.3f} | {num(r,'ttft_s')*1000:.2f} |" for r in metrics)
    component_table = []
    for hw, execution, att, exp in [('A100','native','attention.total','expert.routed'),
            ('H100','native','attention.core','expert.routed_gemm'),
            ('H100','eager_diagnostic','attention.total','expert.routed')]:
        for arm in ['ep2','pp2']:
            r=cc[(hw,arm,execution)]
            component_table.append(f"| {hw} {execution} | {arm.upper()} | {num(r,att):.6f} | {num(r,exp):.6f} | {num(r,'communication.dispatch'):.6f} | {num(r,'communication.combine'):.6f} |")
    history_table = '\n'.join(f"| {r['model']} | {r['gpu']} | {r['phase']} | {r['pp_gain_pct']:+.2f}% |" for r in deck_data['history'])
    report = f'''# PP / EP 部署决策：本轮补测与汇报材料

完成了 Qwen3.6-35B-A3B BF16 在同一对 A100、同一对 H100 上的 EP2 / 两槽 PP2 稳态 Decode 补测，并补齐较长的 Attention / Expert / 通信分项。结合已有模型和 workload 数据，制作了 12 页中文 PPT。结论支持同时比较专家执行和通信成本；尚未建立跨节点阈值或自动预测算法。

- [可编辑 PPT]({OUT}/PP_EP_deployment_decision_20261026.pptx)
- [PDF 预览]({OUT}/PP_EP_deployment_decision_20261026.pdf)
- [全部图表与数据]({OUT})
- [逐请求数据]({data_dir}/requests.csv)、[三轮数据]({data_dir}/rounds.csv)、[逐卡分项]({data_dir}/components_per_rank.csv)
- [协议]({data_dir}/protocol.json)、[来源哈希]({data_dir}/source_hashes.json)、[每页讲稿与来源]({OUT}/speaker_notes.md)

## 本轮原生端到端结果

各配置 3 轮，每轮同时发出 8 请求。每条真实 GPU 已缓存 P=4096，本轮输入 Q=1，生成 128；总旧缓存 32768、本轮新输入 8、每轮输出 1024 token。相同语义提示词；无 CPU KV 换入、无 synthetic KV。前缀准备、权重加载、JIT、预热和 profile 均不计入正式计时。吞吐越高越好，TPOT / TTFT 越低越好。

| 平台 | 布局 | 输出 token/s，均值 ± 三轮标准差 | 三轮范围 | 平均 TPOT ms | 平均 TTFT ms |
|---|---|---:|---:|---:|---:|
{table}

A100 上 PP 的 TPOT 降低 **{a_reduction:.1f}%**；H100 上 EP 的 TPOT 降低 **{h_reduction:.1f}%**。H100 此负载下 PP 的 TTFT 较短、EP 的持续生成 TPOT 和吞吐更好，因此仍需按部署目标选择。三轮不能代表长期稳定性或生产尾延迟；本轮保留了固定 EP 后 PP 的顺序。

A100 保留 Triton Expert / Triton Attention、Graph 关、overlap 关及 DeepEP normal BF16；H100 保留 BF16 DeepGEMM、FA3 / Triton、原生 Decode Graph 和 EP overlap，PP overlap 不支持。这是各平台原生配置的对比，不能归因为只更换 GPU。

## Attention / Expert / 通信对照

下面单位均为 **ms / 卡 / 一次实际层调用**。EP 每卡 40 层，PP 每卡 20 层；均筛选实际本地 batch=4 的 Decode。每配置 3 个独立 profile，每个 profile 两个 rank，各丢弃最初 4 个 Decode forward 后取 24 个完整 forward。共 36 份 rank trace、864 个 rank forward、25920 次实际层调用。误差条按三份独立 profile 的卡均值计算；不是把同一 trace 的重复层当作独立样本。

| 平台与执行范围 | 布局 | Attention | Routed Expert / GEMM | Dispatch | Combine |
|---|---|---:|---:|---:|---:|
{chr(10).join(component_table)}

**统计范围必须区分：** A100 native 与 H100 eager diagnostic 的 Attention 含投影、Expert 为模块；H100 native 使用实际 Graph replay，只识别 Attention 核心和 Routed GEMM，其余投影 / Shared Expert / routing 等留在 other。H100 eager 诊断使用同一批加载权重、单独走已有 eager fallback，验证没有整模型 Graph replay；不得把它的绝对时间当成原生 Graph 模块耗时。完整表另外保留 Shared Expert、other 和标准差。

A100 的 Attention 变化 **{change('A100','attention.total'):+.1f}%**，Routed Expert **{change('A100','expert.routed'):+.1f}%**；后者 PP 三份 profile 标准差较大，应表述为接近、略增。EP 逐层 Dispatch + Combine 明显，PP 免去这部分专家交换而最终 TPOT 更低。H100 原生 Graph 的 Attention 核心变化 **{change('H100','attention.core'):+.1f}%**，Routed GEMM **{change('H100','expert.routed_gemm'):+.1f}%**；eager 模块诊断分别 **{change('H100','attention.total','eager_diagnostic'):+.1f}%** / **{change('H100','expert.routed','eager_diagnostic'):+.1f}%**，支持关注专家执行效率的变化，但并未隔离工作集、padding 和路由复用各自的因果贡献。

PP 仍有阶段边界通信。A100 native / H100 native 的 NCCL 驻留约 **{num(cc[('A100','pp2','native')],'pp_boundary_residency_ms_per_selected_forward'):.3f} / {num(cc[('H100','pp2','native')],'pp_boundary_residency_ms_per_selected_forward'):.3f} ms / 卡 / 所选 stage forward**。该指标来自选定 CPU 窗口内 NCCL GPU 活动的时间并集，包含等待并可能重叠计算，不能解释为纯 NVLink 载荷时延，不能与上表每层数值相加。GPU 分项本身也可能重叠，不是独占 TPOT 百分比分解。

## 已有三维证据如何配合本轮结果

历史统一 moderate 对照中，PP 相对 EP 的三轮均值吞吐变化如下；两卡、8 请求，Prefill P=0 / Q=4096 / O=1；Decode P=4096 / Q=1 / O=128。历史运行用各自原生配置；没有和本轮请求记录混成一个平均值。

| 模型 | 平台 | 阶段 | PP 相对 EP |
|---|---|---|---:|
{history_table}

H100 冷 Prefill 中，35B / 122B 的 PP 额外 Routed + Shared Expert 活动约 **66.1 / 117.8 ms/卡**，而 EP Dispatch + Combine 约 **132.7 / 129.0 ms/卡**；Attention 在两布局间接近。这是“本地专家代价会吃掉通信收益”的机制线索。35B BF16 与 122B FP8 的架构、精度、后端不同，加上小模型 Prefill 的三轮波动，不能推出越大的模型必然越偏 EP。已有 Q07 有更细的跨模型审查。

H100 35B 的 workload 例子：8 请求中等负载 PP Prefill / Decode 吞吐 +14.2% / −25.1%；16×179968 GPU 缓存、各新增 512 的长 Prefill为 +11.7%，对应 Decode −18.6%；32×89984 缓存、各新增 512 时为 −4.5% / −18.0%。后两点旧缓存总量同为 2879488，但请求构成和总新增输入不同，不是单变量 batch 扫描。长上下文 EP 实际本地 batch8 Graph 中，Attention 核心约 10.095 ms、Routed GEMM 2.076 ms，因此 Attention 的绝对成本不能忽略。

A100 122B Prefill 还存在 Attention 干扰反例：两槽 PP Attention 比 EP 约高 11.2%；同层同形状 Marlin 投影与 NCCL 高度重叠时 0.7718→1.2679 ms，而 Full Attention 核心 5.0535→5.0619 ms。支持把通信驻留、等待及调度干扰作为独立考虑项；这仍是相关性证据，没有移除通信的因果 A/B，也未测精确 SM 利用率。

## 给部署决策的简洁用法

1. 用权重 + KV + 运行缓冲筛掉不可行布局，并标明实际 EP 通信域和 PP 边界。
2. 对候选测 Expert 工作集 / 输入复用对应的执行变化、Dispatch / Combine、阶段边界和等待；Attention 保留为对照，并检查长上下文与通信干扰。
3. 在目标 workload 上短测原生 TTFT / TPOT / 吞吐，按 SLO 确认。通信节省大且 Expert 惩罚小倾向 PP；Expert 惩罚或流水依赖更大倾向 EP。判断针对未被重叠隐藏的关键路径成本，不能直接相加独立内核时间。

这份材料是有测量支持的候选筛选方法，尚不是已校准的自动部署预测器。本轮只涉及节点内 NVLink 双卡，不提供跨节点 EP / PP 的切换阈值，也没有补造一、二、四卡全容量网格。

## 核验、失败记录和复现

最终采用 a100_r2 / h100_r2；两者原始 Guard postcheck、canonical CPU 权重缓存审查、拥有进程清理均通过，GPU 资源已释放。两平台完整冻结引擎 commit为 870269702285d94bdd926a50cc29ad1a20bcf60c；额外运行适配器逐文件哈希位于各 launch_manifest。A100 NV12 与 H100 NV6 是各节点 nvidia-smi 观测标签，并非实测带宽。

原始失败不覆盖：a100_r1 在 CPU 准备阶段失败，没有 GPU 测量；h100_r1 首轮 EP TTFT 有明显异常且 eager 开关钩在未使用的 facade 上，保留原始记录。修正为实际 DecodeCudaGraphRunner 的资格检查后，H100 r2 扩展本地 1/2/3/4 形状预热，并完整重测两个布局各三轮；没有只剔除旧轮中的慢样本。最终 eager trace 均确认无 Graph replay。A100 先前短 profile 的 Expert 方向与本轮较长共同提示词采样不同，旧数据保留；本轮不把两者合并为稳定的普适趋势。

运行前查询的 CPU KV 注册表没有该 Qwen 混合模型的兼容项，使用真实 GPU 原生前缀；没有导出、搬运或伪造不兼容 GDN checkpoint。现有后台资源管理器正常，本轮没有执行被自动审批拒绝的独立 backfill_trigger 动作。

PPT 文字和形状可编辑，图表另存 PNG / 矢量 PDF，数据为 CSV / JSON。已核验 12 页、所有对象边界和文字高度，并逐页检查布局预览；当前环境无 PowerPoint / LibreOffice，PDF 由同一布局的高分辨率页面生成，未声称通过 Office 实机渲染。PPT 使用 Noto Sans CJK SC；接收端字体回退可能影响排版，PDF 可直接展示。原始大 trace、权重、KV、admin key、私有 config 和 lease token 均留在原运行位置，不进入分享包。

原始补测目录：[{ROOT}/runs]({ROOT}/runs)。轻量快照：[{ARCHIVE}]({ARCHIVE})。每页 notes 和 sources.json 指向图表的原始来源。
'''
    (OUT / 'REPORT.md').write_text(report)
    (ROOT / 'README.md').write_text(f'# PP / EP 决策补测与 PPT\n\n[结果报告]({OUT}/REPORT.md) · [PPT]({OUT}/PP_EP_deployment_decision_20261026.pptx) · [PDF]({OUT}/PP_EP_deployment_decision_20261026.pdf)\n\n最终运行 a100_r2 / h100_r2；失败尝试保留，不覆盖历史资料。\n')
    notes = json.loads((OUT / 'slide_notes.json').read_text())
    (OUT / 'speaker_notes.md').write_text('\n\n'.join(f"## {r['slide']:02d} {r['title']}\n\n{r['notes']}\n\n来源：\n\n"+'\n'.join(f'- [{Path(p).name}]({p})' for p in r['sources']) for r in notes)+'\n')
    for filename in ['analyze.py','build_deck.py','deck_engine.py','finalize.py','validate_deliverables.py','diagnostic_switch.py','test_diagnostic_switch.py','sampling.py','prepare.py','runtime_adapter.py']:
        shutil.copy2(ROOT/filename, source_dir/filename)
    shutil.copy2('/tmp/pp_ep_ppt_assets/OFL.txt', source_dir/'Noto_CJK_LICENSE.txt')
    (source_dir/'README.md').write_text('''# Reproduction

The frozen runner/adapters, private launch configs and large original traces remain in the original run directories. Use launch_manifest.json and data/source_hashes.json to verify provenance. Do not rerun a GPU experiment without Experiment Guard.

CPU analysis: python analyze.py --run ORIGINAL_RUN_DIRECTORY

Deck build originally used Python 3.11, python-pptx 1.0.2, matplotlib 3.11.0 and Pillow 13.0.0. Install python-pptx, matplotlib, numpy and Pillow into an isolated environment. Obtain NotoSansCJKsc-Regular.otf and NotoSansCJKsc-Bold.otf from the official notofonts/noto-cjk GitHub repository (Sans/OTF/SimplifiedChinese), license included. Set deck_engine.FONT_ROOT to that directory. Run build_deck.py from the original experiment directory; it reads verified source artifacts at the paths in sources.json. The PDF uses the same deterministic layout, not Office rendering.

Archive data/ contains portable CSV/JSON inputs for independently redrawing every quantitative comparison; PPT text/shapes are editable. The separate figure PDFs are vector charts.
''')
    manifest = dict(id=ID, title='35B A100/H100 steady Decode EP2/PP2 and deployment-decision PPT',
        created_at=datetime.now(timezone.utc).isoformat(), status='complete',
        implementation=dict(engine='SGLang', commit=protocol['source_commit'], source_tag=protocol['source_tag'],
            worktree=protocol['frozen_engine_worktree'], engine_dirty=False, old_implementation=False,
            frozen_adapters='original run/code files with SHA256 in launch manifests'),
        deployment=dict(run_ids=['a100_r2','h100_r2'], nodes=['spartan-gpgpu069','spartan-gpgpu178'],
            jobs=['33463295','33646188'], endpoints='ephemeral; exited; no gateway route',
            rollback_endpoint=None, validation='evidence/*/guard_and_cleanup.json', original_root=str(ROOT)),
        workload=protocol['workload'], model=protocol['model'], testbed=protocol,
        key_results=dict(a100_pp_tpot_reduction_percent=a_reduction, h100_ep_tpot_reduction_percent=h_reduction),
        artifacts=dict(pptx='PP_EP_deployment_decision_20261026.pptx', pdf='PP_EP_deployment_decision_20261026.pdf',
            report='REPORT.md', full_metrics='data/native_metrics.csv', stage_metrics='data/components_per_rank.csv',
            sources='data/source_hashes.json', raw_traces='referenced only, original run/arm/traces'),
        limitations=protocol['limitations'], no_weights_kv_credentials_or_raw_traces_copied=True)
    save_json(OUT / 'manifest.json', manifest)
    print(json.dumps(dict(out=str(OUT), metrics=len(metrics), ranks=len(ranks), requests=len(requests), archive=str(ARCHIVE))))


if __name__ == '__main__':
    main()
