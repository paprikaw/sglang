"""Build an auditable report and small archive from completed, immutable runs.

This script performs no GPU work and never copies private Guard authorization,
weight banks, KV payloads, or large traces into the archive.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st
import subprocess

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT = ROOT / 'deliverables'
ARCHIVE = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_expert_fragmentation_20261026')
ID = ARCHIVE.name


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def table(path, rows):
    with Path(path).open('w', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def percentile(xs, q):
    xs = sorted(xs)
    p = (len(xs) - 1) * q
    i = int(p)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (p - i)


def link(label, path):
    return f'[{label}]({Path(path).absolute()})'


def main():
    OUT.mkdir(exist_ok=True)
    status = read(ROOT / 'analysis/analysis_status.json')
    assert not status['errors'], status
    metrics = list(csv.DictReader((ROOT / 'analysis/metrics.csv').open()))
    components = list(csv.DictReader((ROOT / 'analysis/components.csv').open()))
    rounds, requests, summaries = [], [], []
    for row in metrics:
        p = Path(row['source'])
        assert digest(p)['sha256'] == row['sha256'], p
        data = read(p)
        cohort_values = []
        req = []
        for i, group in enumerate(data['rounds'], 1):
            throughput = (sum(r['output_tokens'] for r in group['requests']) if row['phase'] == 'decode'
                          else int(row['requests']) * int(row['new_tokens'])) / group['makespan_s']
            cohort_values.append(throughput)
            common = {k: row[k] for k in ('run', 'mode', 'arm', 'phase', 'guard_stage')}
            rounds.append(dict(**common, round=i, cohort_ms=group['makespan_s'] * 1000,
                               throughput_tok_s=throughput, source=str(p)))
            for j, r in enumerate(group['requests']):
                assert r['output_tokens'] == int(row['output_tokens'])
                assert (r.get('cached_tokens') or 0) == int(row['context'])
                value = dict(**common, round=i, request=j, cached_tokens=r.get('cached_tokens') or 0,
                             prompt_tokens=r.get('prompt_tokens'), output_tokens=r['output_tokens'],
                             e2e_ms=r['e2e_s'] * 1000, ttft_ms=r['ttft_s'] * 1000,
                             tpot_ms=r.get('tpot_ms'), source=str(p))
                requests.append(value)
                req.append(value)
        summary = dict(row, throughput_median=st.median(cohort_values))
        for name in ('e2e_ms', 'ttft_ms', 'tpot_ms'):
            xs = [r[name] for r in req if r[name] is not None]
            if xs:
                summary.update({name + '_median': st.median(xs), name + '_p99': percentile(xs, .99)})
        summaries.append(summary)
    table(OUT / 'rounds.csv', rounds)
    table(OUT / 'per_request.csv', requests)
    table(OUT / 'full_metrics.csv', summaries)

    cfg = read(ROOT / 'runs/r5_adaptive_confirm/config.json')
    launch = read(ROOT / 'runs/r5_adaptive_confirm/launch_manifest.json')
    spec = read(ROOT / 'runs/r5_adaptive_confirm/dashboard_spec.json')
    native = next(d for d in spec['frozen_inputs'] if d['path'] == cfg['deepep_native'])
    communication = dict(native=native,
        wrapper_files=[d for d in spec['frozen_inputs'] if Path(d['path']).is_relative_to(cfg['deepep_wrapper'])],
        environment={k:v for k,v in cfg['launch_env'].items()
                     if any(prefix in k for prefix in ('DEEPEP', 'NVSHMEM', 'DEEP_EP', 'SGLANG_EP', 'SGLANG_MOE'))
                     and not any(secret in k for secret in ('SECRET', 'AUTH', 'PASSWORD', 'LICENSE', 'KEY'))},
        graph_mode='native CUDA Graph Decode; eager Prefill',
        path='same-node NVLink pair; no cross-node transport comparison in this experiment')
    write(OUT / 'communication_provenance.json', communication)
    git = lambda *args: subprocess.check_output(['git', '-C', cfg['engine_root'], *args], text=True).strip()
    source = dict(engine='SGLang', worktree=cfg['engine_root'], branch=git('branch', '--show-current'),
                  commit=git('rev-parse', 'HEAD'), tag=launch['source_tag'], dirty_state=git('status', '--porcelain'),
                  old_implementation=True, private_adapter=True)
    assert source['commit'] == launch['source_commit']
    assert not source['dirty_state'], source
    audits = []
    for run in sorted((ROOT / 'runs').iterdir()):
        progress = read(run / 'task/progress.json')
        if progress['stage'] == 'queued':
            audits.append(dict(run=run.name, launched=False, stage='prepared_only'))
            continue
        final = None
        for line in (run / 'supervisor.log').read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and 'gpu_registry_state' in item:
                final = {k:item.get(k) for k in ('experiment_exit_code', 'postcheck', 'gpu_registry_state')}
        assert final and final['gpu_registry_state'] == 'released', run
        clean = read(run / 'task/cleanup.json')
        assert clean['passed'] and clean['model_processes_exited'], run
        for arm in read(run / 'config.json')['arm_order']:
            c = read(run / arm / 'cleanup.json')
            assert c['cleanup_verified'] and c['process_groups_exited'], (run, arm)
        private = read(run / 'launch_manifest.json')['private_adapters']
        for entry in private:
            assert digest(entry['path'])['sha256'] == entry['sha256'], entry['path']
        audits.append(dict(run=run.name, launched=True, stage=progress['stage'],
                           cleanup_passed=True, **final, private_adapter_hashes_verified=len(private)))
    assert sum(a['launched'] for a in audits) == 7
    write(OUT / 'run_audit.json', audits)

    def metric(run, arm, phase):
        return next(r for r in summaries if (r['run'], r['arm'], r['phase']) == (run, arm, phase))

    refs = [('EP2 native', 'r4_reference_fixed', 'ep2'),
            ('PP2 native', 'r4_reference_fixed', 'pp2'),
            ('PP2 adaptive', 'r5_adaptive_confirm', 'pp2')]
    ep = metric('r4_reference_fixed', 'ep2', 'decode')
    pp = metric('r4_reference_fixed', 'pp2', 'decode')
    opt = metric('r5_adaptive_confirm', 'pp2', 'decode')
    number = lambda r, k: float(r[k])
    gain = (number(opt, 'throughput_tok_s') / number(pp, 'throughput_tok_s') - 1) * 100
    tpot_drop = (1 - number(opt, 'tpot_ms') / number(pp, 'tpot_ms')) * 100
    e2e_drop = (1 - number(opt, 'request_e2e_ms') / number(pp, 'request_e2e_ms')) * 100
    deficit = (1 - number(opt, 'throughput_tok_s') / number(ep, 'throughput_tok_s')) * 100

    def component(run, arm, phase, key, multiplier=1):
        xs = [float(c[key]) for c in components if c['run'] == run and c['arm'] == arm and c['phase'] == phase and c.get(key)]
        assert len(xs) == 2, (run, arm, phase, key)
        return st.mean(xs) * multiplier

    component_rows = []
    for label, run, arm in refs:
        component_rows.append(dict(configuration=label,
            attention_core_us=component(run, arm, 'decode', 'attention.core', 1000),
            expert_gemm_us=component(run, arm, 'decode', 'expert.routed_gemm', 1000),
            dispatch_us=component(run, arm, 'decode', 'communication.dispatch', 1000) if arm == 'ep2' else 0,
            combine_us=component(run, arm, 'decode', 'communication.combine', 1000) if arm == 'ep2' else 0,
            scope='Mean of two ranks; microseconds per actual layer call; local Decode batch4; nonadditive kernel scopes'))
    table(OUT / 'decode_components.csv', component_rows)
    expert_drop = (1 - component_rows[2]['expert_gemm_us'] / component_rows[1]['expert_gemm_us']) * 100
    prefill_opt = metric('r5_adaptive_confirm', 'pp2', 'prefill')
    prefill_pp = metric('r4_reference_fixed', 'pp2', 'prefill')

    performance_table = '\n'.join(
        f"| {label} | {phase.title()} | {number(r, 'throughput_tok_s'):,.2f} ± {number(r, 'throughput_stdev'):,.2f} | {number(r, 'throughput_median'):,.2f} | {number(r, 'request_e2e_ms'):.2f} | {number(r, 'ttft_ms'):.2f} | {number(r, 'tpot_ms'):.3f} | {r['repeats']} |"
        if phase == 'decode' else
        f"| {label} | Prefill | {number(r, 'throughput_tok_s'):,.2f} ± {number(r, 'throughput_stdev'):,.2f} | {number(r, 'throughput_median'):,.2f} | {number(r, 'request_e2e_ms'):.2f} | {number(r, 'ttft_ms'):.2f} | — | {r['repeats']} |"
        for phase in ('prefill', 'decode') for label, run, arm in refs
        for r in [metric(run, arm, phase)])
    component_table = '\n'.join(f"| {r['configuration']} | {r['attention_core_us']:.2f} | {r['expert_gemm_us']:.2f} | {r['dispatch_us']:.2f} | {r['combine_us']:.2f} |" for r in component_rows)
    residency = []
    for p in sorted((ROOT / 'runs/r4_reference_fixed/pp2/routing').glob('microbench.*.t4.json')):
        d = read(p)
        residency.append(dict(stage=d['routing']['pp_rank'], **{k:v['graph_median_ms'] * 1000 for k,v in d['modes'].items()}, source=str(p)))
    write(OUT / 'resident_dimension_control.json', residency)
    resident_table = '\n'.join(f"| {r['stage']} | {r['native_resident32']:.2f} | {r['native_resident64']:.2f} | {r['native_resident128']:.2f} | {r['native']:.2f} |" for r in residency)
    probabilities = read(ROOT / 'analysis/probability_validation_batched.json')
    probability = probabilities['summary'][2]
    assert probability['common_history_positions'] == 55 and probability['identical_argmax_positions'] == 53
    key_results = dict(decode_throughput_gain_percent=gain, decode_tpot_reduction_percent=tpot_drop,
                       decode_e2e_reduction_percent=e2e_drop, expert_gemm_reduction_percent=expert_drop,
                       adaptive_throughput_deficit_vs_ep_percent=deficit,
                       prefill_stable_gain=False, generation_bitwise_equivalent=False,
                       quality_evaluation_completed=False, atomic_batch_probability_comparison=probability)
    write(OUT / 'key_results.json', key_results)
    report = f'''# H100 PP Expert execution：定位、优化与完整模型验证

实测确认：PP 小微批次的 Expert 成本有一部分来自执行布局，能够改善。保持同一对 H100、完整 BF16 模型和 PP2 调度，仅对本地 token 数不超过 4 的 Expert 调用选择小 tile Triton，Decode 吞吐 **{number(pp, 'throughput_tok_s'):.1f}→{number(opt, 'throughput_tok_s'):.1f} output token/s（+{gain:.2f}%）**，TPOT 降低 {tpot_drop:.2f}%，请求端到端延迟降低 {e2e_drop:.2f}%。原生 Graph 中 Routed Expert GEMM 平均时间降低 **{expert_drop:.2f}%**，Attention core 接近。

本轮优化没有稳定的 Prefill 收益；优化后 PP Decode 仍比 EP2 低 {deficit:.2f}%。所有策略保持 BF16，但生成 token 不完全一致，因此当前结果是有数值差异的性能实验，尚不是已完成质量评估的生产部署方案。

## 1. 相同资源与明确负载

| 项目 | 本轮协议 |
|---|---|
| 模型 | 完整 Qwen3.6-35B-A3B，BF16 权重与激活；FP32 GDN 状态 |
| GPU | spartan-gpgpu178，同一对 GPU0/1，H100 80GB NVLink；已有作业 33646188；各 arm 顺序运行 |
| EP2 | MoE EP2；Attention DP2 × TP1；全局参数 tp2/dp2/ep2/pp1 |
| PP2 | MoE EP1，Attention TP1；每 stage 20 层；两个流水槽位，每个微批次最多 4 请求 |
| Prefill | 8 请求；每条无旧缓存，新增 4096 input token，输出 1；总新增 32768 token |
| Decode | 8 请求；每条已有真实模型生成的 GPU 缓存 4096 token，再新增 1 token，输出 128；ignore EOS |
| 缓存 | 原生 GPU radix prefix；没有兼容的混合 GDN CPU checkpoint，未使用 CPU restore 或合成 KV |
| 调度与 chunk | PP ready-send 关闭；PP chunk4096；EP 全局8192经 DP2 分为每实例4096；本轮不改变这些配置 |
| 内核 | H100 原生 BF16 DeepGEMM，FA3；EP 原生 overlap；PP 原生 overlap 不支持；Decode CUDA Graph，Prefill eager |
| 计时 | 缓存准备、预热、编译、数值验证和 profile 均在正式计时之外；原生各 3 轮，最终 adaptive 各 5 轮 |
| 输入组成 | 固定 tokenized 文档提示，每请求不同起始 token、主体相同；这是受控工作负载，尚未覆盖多样化服务流量 |

Decode 预热覆盖总请求数 2/4/6/8，对应每个本地执行实例的 Graph 形状 1/2/3/4。分析重新检查了每条正式请求的缓存计数和输出长度。

## 2. 完整模型结果

Prefill 吞吐是新增 input token/s，Decode 吞吐是 output token/s，越高越好；延迟单位为 ms，越低越好。吞吐列是各 cohort 的均值 ± 样本标准差；端到端、TTFT、TPOT 是逐请求均值。TPOT = (端到端−TTFT)/(输出数−1)，不是把整个请求时间除以输出数。

| 配置 | 阶段 | 吞吐均值 ± SD | 吞吐中位数 | 请求端到端 | TTFT | TPOT | 轮数 |
|---|---|---:|---:|---:|---:|---:|---:|
{performance_table}

最终 adaptive Prefill 的五轮吞吐为 **78,511、78,522、78,389、78,411、59,564**。最后一轮 cohort 为 550.13 ms，另外四轮约 417–418 ms。全部保留，均值相对原生 PP 下降 {(1-number(prefill_opt, 'throughput_tok_s')/number(prefill_pp, 'throughput_tok_s'))*100:.2f}%；中位数 **{number(prefill_opt, 'throughput_median'):,.2f} vs {number(prefill_pp, 'throughput_median'):,.2f}**。慢轮原因未确定，不能据此宣称 Prefill 完全无退化。Prefill 的实际 T4096 Expert 调用仍走原生路径。

{link('完整指标及逐请求中位数/P99', OUT/'full_metrics.csv')} · {link('每轮结果', OUT/'rounds.csv')} · {link('逐请求计时', OUT/'per_request.csv')}

![End-to-end and components]({OUT/'end_to_end_and_components.png'})

## 3. Expert 问题真实存在，Attention 不是本轮提速来源

以下是实际 Decode CUDA Graph 中每卡、每次实际层调用的平均 GPU kernel 时间，单位微秒；两 rank 各选择 16 个稳定的本地 B4 forward，丢弃前 4 个。Attention 仅包含 core，Expert 仅包含 routed GEMM，不包含全部投影、共享专家与路由。

| 配置 | Attention core | Routed Expert GEMM | Dispatch | Combine |
|---|---:|---:|---:|---:|
{component_table}

PP Expert **94.19→57.62 µs**，接近 EP 的 59.56 µs；PP Attention **12.50→12.33 µs**。这支持“先优化 Expert 执行”的方向。上述是独立 kernel scopes，不能相加成请求延迟；PP 没有逐层 Dispatch/Combine，但仍有 stage 边界与反馈通信。stage0 的 NCCL 驻留从 3.475→2.323 ms/所选 forward，包含等待和 overlap，不能直接当作网络传输时间或与 Expert 收益相加。

Prefill 独立全模块 profile 中，原生 EP/PP 的 Expert 累计约 119.69/175.02 ms/卡，Attention 约 126.55/125.16 ms/卡。EP 的 Dispatch+Combine 累计约 165.59 ms/卡。PP 虽有较慢的 Expert，仍通过省去逐层跨卡交换在该 Prefill 负载获益。Prefill 模块累计计时与 Decode core 计时范围不同，不混成同一占比图。

## 4. 等有效计算量对照：不活跃 Expert 维度也会增加成本

真实 PP 的每层有 256 个 resident experts。路由记录显示，本地 Decode T4 只有平均 **11.1 个活跃 Expert**（8–14），总共 32 次 token–Expert assignment；T4096 Prefill 平均活跃 193.15 个（123–230）。这些活跃数与本轮提示组成有关。

原生 masked 预处理为每个本地 Expert 分配 `m_max=(T//256+1)*256` 的 M 容量。因此 T4 仍使用 M 容量256。它表明布局容量较大，不能据此声称所有 padding 对应的 GEMM FLOPs 都实际执行了。

更强的控制实验固定同一真实层输入、所选权重、路由、有效计算量，使用同一个128-Expert权重 bank 的相同地址前缀，仅改变传给原生 kernel 的本地 Expert 维度；全局 num_experts=256 保持不变，expected-M tuning hint 保持1。Expert-only CUDA Graph 时间：

| 真实层来自 | E32 | E64 | E128 | 原始 E256 参考 |
|---|---:|---:|---:|---:|
{resident_table}

E32/64/128 是严格共享权重地址控制，输出与 DeepGEMM 参考完全相同；原始 E256 使用原模型权重 bank，仅作参考。权重收集准备不计入此内核实验，正式优化不做这种权重重排或复制。

这说明 PP 的 Expert 劣势不全是不可避免的有效计算量差异，执行布局对不活跃 Expert 的处理本身就能产生额外成本。

![Fragmentation mechanism]({OUT/'fragmentation_mechanism.png'})

## 5. 实现和几轮优化

| 轮次 | 方法 | 观察与取舍 |
|---|---|---|
| r1 | 原生 EP2/PP2、真实路由与同输入 kernel 对照 | PP 小形状原生约100 µs，已有小 tile Triton约35–42 µs |
| r2 | PP 本地 T≤4 选择 Triton，较大输入走原生 DeepGEMM | Decode 795.65 vs r1 PP 674.90 tok/s；后续以重新测量的原生参考确认 |
| r3 | 调整 M/N/K tile；同时测 compact DeepGEMM 与新 route-indexed BF16 GEMV | M16N64K128 单层略快，但完整模型没有稳定进一步收益；该轮后续 logprob 传输失败，整轮 Guard 标为 failed，计时保留为探索数据，不进入最终主结果 |
| r4 | 修复仅验证阶段的 PP logprob 元数据传输，重测 EP/PP；增加等有效工作量的 resident 维度控制 | 获得最终原生参考与因果证据 |
| r5 | 回到默认 small-tile Triton，5 轮确认 | Decode 805.67 tok/s；Expert GEMM −38.83%；Prefill 慢轮全部保留 |
| r6/r7 | 原生/优化各3轮原子批量概率验证 | 去除客户端到达分组差异；数值差异重复出现，见下节；没有计入性能结果 |

同实际 T4 输入的候选 kernel 对照，stage0/stage1 微秒：masked DeepGEMM 98.30/102.52；compact DeepGEMM 86.27/93.15；新 route GEMV 43.07/47.07；默认 small-tile 34.66/42.35；调 tile 后33.06/41.48。route GEMV 避免 token-row padding 并融合 up+SwiGLU，但仍慢于小 tile Tensor Core 路径，因此没有选入最终方案。只减少 padding 并不足以保证最快。

最终策略在 {link('fragmentation.py', ROOT/'fragmentation.py')}：只作用于 PP2，按实际本地 T≤4 切换现有 BF16 Triton runner。没有改路由专家数、权重精度、Attention、流水槽位，也没有跨微批次等待合批。每个形状在 Graph capture 时确定路径。更大的输入继续原生 DeepGEMM。

## 6. 数值验证和结果适用范围

单层真实输入对照通过有限值和相对 L2 阈值1.5%的实验检查；Triton 相对 DeepGEMM 约 **0.27%–0.33% L2**，新 GEMV约0.23%–0.27%，compact DeepGEMM在这些输入上完全相同。这不是完整生成的质量评估。

最初并发单请求 API 的原生重复也受到到达分组影响，因此另测原子 batch API：一次提交8条输入，每条生成8个 token，连续3轮。原生 PP 自身重复 **64/64位置完全一致，top10 logprob 差为0**。优化版三个重复也一致，但相对原生仅有 **55/64位置仍处于完全相同的生成历史**；其中下一 token 相同 **53/55（96.36%）**，top10 平均重合9.65，交集 token 的平均绝对 logprob 差 **0.08916**，最大 **0.74750**。后续分歧轨迹的位置没有被当作同条件概率比较。

因此不能称为逐 token 等价或已验证无损。完整128-token性能运行也可能沿不同轨迹执行；相同实际输入的单层对照才是隔离 kernel 成因的直接证据。本轮证明了性能机会，生产采用仍需任务质量评估和数值容差标准。

{link('固定批次概率比较', ROOT/'analysis/probability_validation_batched.json')} · {link('位置级记录', ROOT/'analysis/probability_positions_batched.csv')} · {link('原并发验证保留', ROOT/'analysis/probability_validation.json')}

r3 的验证失败来自该归档引擎将嵌套 CUDA logprob Tensor 作为 Python pickle 传回前一 PP stage；接收时在对方已满 GPU 创建第二个 CUDA context，导致 OOM。私有 adapter 只在未计时的 V 验证阶段把嵌套元数据转到 CPU，顶层 Tensor 仍保留原生 typed transport；正式性能路径不做此 CPU 转换。r4/r5/r6/r7 后续验证与清理均正常结束。

## 7. 对研究问题的影响

最有价值的新发现是：**部署决策中的 Expert 成本需要显式包含 local token 数、resident/active Expert 数和 kernel/layout；仅以有效 FLOPs 或模型总参数预测，会把一部分可优化的实现成本误认为 PP 的固有代价。**

这个负载下优化后 PP 的逐层 Expert 成本已经接近 EP，但完整 Decode 仍较慢，说明下一步还需定位流水依赖、反馈、边界等待和其他算子在关键路径上的作用。当前结果还没有证明新的跨槽位 Expert 合批调度，也不能据此宣称此机制本身的研究新颖性。更大的 batch、更多样路由和跨通信域，属于后续泛化范围。

## 8. 复现与来源

源代码 commit：`{source['commit']}`；不可变 tag：`{source['tag']}`。冻结引擎源码当前 clean。模型 revision：`{cfg['model_revision']}`。DeepEP wrapper：`{cfg['deepep_wrapper']}`；native：`{cfg['deepep_native']}`，SHA256 `{native['sha256']}`。{link('通信库哈希及实际环境', OUT/'communication_provenance.json')}。全部启动参数、每个私有 adapter 哈希、具体环境和 GPU 映射保存在各 run 的 launch_manifest/config；本报告与快照不复制授权文件、权重或 KV payload。

{link('最终优化运行来源', ROOT/'runs/r5_adaptive_confirm/launch_manifest.json')} · {link('原生参考来源', ROOT/'runs/r4_reference_fixed/launch_manifest.json')} · {link('数值/计时汇总 CSV', ROOT/'analysis/metrics.csv')} · {link('GPU分项 CSV', ROOT/'analysis/components.csv')} · {link('同输入内核 CSV', ROOT/'analysis/microbench.csv')} · {link('路由记录 CSV', ROOT/'analysis/routing.csv')}

离线重算，不启动 GPU：

```bash
python3 experiments/h100_pp_expert_fragmentation_20261026/analyze.py
python3 experiments/h100_pp_expert_fragmentation_20261026/validate_probabilities.py --batched
/home/bxb1/conda/bin/python experiments/h100_pp_expert_fragmentation_20261026/plot.py
python3 experiments/h100_pp_expert_fragmentation_20261026/finalize.py
```

新 GPU 复现先重新核对 booking、Guard 节点及 CPU 权重 bank，使用新的 run 目录。`prepare.py --mode native --arms ep2 pp2` 为参考；`--mode triton_small --arms pp2 --repeats 5` 为最终策略，再交由 `launch.py --run <new-run>` 执行 Guard。现有 run 的冻结代码不修改。

共执行7次 Guard，另有1个只准备未启动的目录。6次实验完整结束，r3 后续验证失败但清理正常。全部自有模型进程退出、7次租约均释放，后台工作可恢复；没有修改共享引擎默认或 gateway。{link('逐轮清理、释放和冻结哈希审计', OUT/'run_audit.json')}。

本次小型归档：{link('experiment-manager snapshot', ARCHIVE/'manifest.json')}；原始大 trace 和日志仍在本实验 runs 目录，快照通过路径与哈希关联。
'''
    if (OUT/'HEADROOM_ESTIMATE.md').exists():
        report += '\n\n后续离线分析：' + link('NVIDIA GEMM原则下的Expert理想模型与PP2端到端估算', OUT/'HEADROOM_ESTIMATE.md') + '；其中目标数据为估算，不是新GPU测量。\n'
    (OUT / 'REPORT.md').write_text(report)
    readme = f'''# H100 PP Expert fragmentation — completed

Full BF16 Qwen3.6-35B-A3B, the same H100 NVLink GPU0/1 pair, matched cold Prefill and GPU-cached Decode. Shape-adaptive Expert execution improves PP2 Decode throughput by {gain:.2f}% and reduces routed GEMM time by {expert_drop:.2f}%. Prefill has no stable gain; generation is not bitwise equivalent, and task quality has not been evaluated.

- {link('Report with protocol and all limitations', OUT/'REPORT.md')}
- {link('End-to-end and component figure', OUT/'end_to_end_and_components.png')} / {link('PDF', OUT/'end_to_end_and_components.pdf')}
- {link('Same-work mechanism figure', OUT/'fragmentation_mechanism.png')} / {link('PDF', OUT/'fragmentation_mechanism.pdf')}
- {link('Full metrics', OUT/'full_metrics.csv')} / {link('Round audit', OUT/'run_audit.json')}
- {link('Canonical archive', ARCHIVE/'manifest.json')}

Runtime adapter: `fragmentation.py`. Alternative kernel: `route_gemv.py` (measured, not selected). `prepare.py` freezes a fresh private run; `launch.py --run PATH` uses Guard. New launches require fresh resource/cache checks; completed run inputs remain immutable.

Offline replay: `analyze.py`, `validate_probabilities.py --batched`, `plot.py` (Matplotlib Python), then `finalize.py`. Original large logs and GPU traces stay in `runs/`. Seven launches completed cleanup and release; one exploratory run failed its later probability validation and is excluded from the headline comparison.
'''
    if (OUT/'HEADROOM_ESTIMATE.md').exists():
        readme += '\nOffline headroom analysis: ' + link('Expert roofline and PP2 scenarios', OUT/'HEADROOM_ESTIMATE.md') + ' (modeled, not new measurements).\n'
    (ROOT / 'README.md').write_text(readme)

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for folder in ('analysis', 'deliverables', 'code', 'private_runtime'):
        (ARCHIVE / folder).mkdir(exist_ok=True)
    for p in (ROOT / 'analysis').iterdir():
        if p.suffix in ('.csv', '.json'):
            shutil.copyfile(p, ARCHIVE/'analysis'/p.name)
    for p in OUT.iterdir():
        if p.suffix in ('.csv', '.json', '.png', '.pdf', '.md'):
            shutil.copyfile(p, ARCHIVE/'deliverables'/p.name)
    for p in ROOT.glob('*.py'):
        shutil.copyfile(p, ARCHIVE/'code'/p.name)
    for p in (ROOT/'runs/r5_adaptive_confirm/code').glob('*.py'):
        shutil.copyfile(p, ARCHIVE/'private_runtime'/p.name)
    shutil.copyfile(OUT/'REPORT.md', ARCHIVE/'summary.md')
    shutil.copyfile(OUT/'full_metrics.csv', ARCHIVE/'full_metrics.csv')
    shutil.copyfile(ROOT/'analysis/components.csv', ARCHIVE/'stage_metrics.csv')
    originals = [digest(Path(r['source'])) for r in metrics]
    originals += [digest(p) for p in (ROOT/'runs').glob('*/launch_manifest.json')]
    originals += [digest(ROOT/'runs'/run/arm/'validation.probabilities.json')
                  for run,arm in [('r6_native_batch_validation','pp2'),('r7_adaptive_batch_validation','pp2')]]
    traces = {}
    for p in (ROOT/'runs').glob('*/[ep]p2/traces/*/manifest.json'):
        for d in read(p)['files']:
            traces[d['path']] = dict(path=d['path'], sha256=d['sha256'])
    engine_args = {}
    keep = {'tp_size','pp_size','dp_size','ep_size','enable_dp_attention','enable_dp_lm_head',
            'tensor_parallel_size','pipeline_parallel_size','data_parallel_size','expert_parallel_size',
            'context_length','max_total_tokens','mem_fraction_static','chunked_prefill_size','max_prefill_tokens',
            'max_running_requests','pp_max_micro_batch_size','pp_async_batch_depth','kv_cache_dtype',
            'mamba_ssm_dtype','disable_cuda_graph','cuda_graph_bs','enable_hierarchical_cache','hicache_ratio',
            'hicache_size','attention_backend','moe_runner_backend','moe_a2a_backend','deepep_mode',
            'disable_overlap_schedule','page_size','max_total_num_tokens','cuda_graph_bs_decode',
            'dtype','quantization','max_mamba_cache_size'}
    for run,arm in [('r4_reference_fixed','ep2'),('r4_reference_fixed','pp2'),('r5_adaptive_confirm','pp2')]:
        p=ROOT/'runs'/run/arm/'server_info.json'
        info=read(p)
        args=info.get('server_args',info)
        engine_args[run+'.'+arm]=dict(source=str(p),sha256=digest(p)['sha256'],
            server_args={k:v for k,v in args.items() if k in keep},
            internal_states=[{k:v for k,v in state.items() if k in keep}
                             for state in info.get('internal_states',[])])
        assert engine_args[run+'.'+arm]['server_args']['quantization'] is None
    manifest = dict(id=ID, title='H100 PP Expert execution fragmentation: diagnosis and shape-adaptive BF16 execution',
        created_at=datetime.now(timezone.utc).isoformat(), status='completed_experiments_with_numerical_caveat',
        implementation=source, deployment=dict(node=cfg['node'], job_id=cfg['job_id'], gpu_pair=[0,1],
            hardware='H100 80GB NVLink', whole_node_lease_gpus=4, sequential_arms=True,
            deployment_ids=[a['run'] for a in audits if a['launched']], endpoint='retired',
            gateway_route=None, rollback_endpoint=None, all_owned_processes_exited=True, all_leases_released=True),
        model=dict(path=cfg['model_path'], revision=cfg['model_revision'], precision=cfg['precision']),
        workload=dict(requests=8, input='fixed tokenized prompts; distinct initial token, shared document body',
            offered_arrivals='concurrent client requests; validation separately uses atomic batch API',
            prefill=dict(cached=0,new_input=4096,output=1), decode=dict(gpu_cached=4096,new_input=1,output=128),
            prefixes='real native model-produced GPU cache; no CPU KV restore', formal_reference_repeats=3,
            formal_adaptive_repeats=5, quality_evaluation_completed=False),
        engine_parameters={k:cfg[k] for k in ('attention_backend','engine_commit','graph','overlap_schedule','pp_async_batch_depth','pp_loop_slots','prefill_tokens','prefix_cache','protocol','deepep_wrapper','deepep_native')},
        observed_server_parameters=engine_args, communication_provenance=communication,
        layouts=dict(ep2='MoE EP2; Attention DP2 x TP1',pp2='PP2 x MoE EP1; Attention TP1;20 layers/stage;2 slots;local batch4'),
        key_results=key_results, run_audit=audits, original_root=str(ROOT), original_artifacts=originals,
        original_large_traces=list(traces.values()),
        archived_files=[digest(p) for p in sorted(ARCHIVE.rglob('*')) if p.is_file() and p.name!='manifest.json'])
    if (OUT/'headroom_estimate.json').exists():
        manifest['offline_headroom_estimate'] = dict(report=str(ARCHIVE/'deliverables/HEADROOM_ESTIMATE.md'),
            data=str(ARCHIVE/'deliverables/headroom_estimate.json'),
            status='modeled_not_new_GPU_measurement')
    write(ARCHIVE/'manifest.json',manifest)

    qpath = WORKSPACE/'experiments/experiment_summary/component_reports/Q12.md'
    qpath.write_text(f'''# Q12：H100 PP Expert 执行碎片化优化

同一对 H100 GPU0/1，完整 Qwen3.6-35B-A3B BF16；EP2 为 Attention DP2×TP1，PP2 为每 stage20层、两槽、本地 B4。8 请求：Prefill无旧缓存、4096新输入→1输出；Decode为GPU已缓存4096、新输入1→128输出。

PP本地 T≤4 使用小tile Triton，其他形状使用原生 DeepGEMM。Decode **666.79→805.67 output tok/s（+{gain:.2f}%）**，TPOT **11.176→9.091ms**，请求端到端 **1532.34→1268.33ms**；原生EP2为863.20tok/s。Expert GEMM **94.19→57.62µs/实际层调用（−{expert_drop:.2f}%）**，Attention core基本接近。相同输入/权重/路由下，仅提高不活跃Expert维度也使原生kernel变慢，支持执行布局成因。

Prefill无稳定改善：均值78608.53→74679.43 input tok/s；优化版5轮包含一个慢cohort，中位数78508.29→78411.35，全部数据保留。单层L2数值检查通过，但完整生成不逐token一致；固定batch短生成55个相同历史位置中53个下一token相同，尚未做任务质量评估。

{link('完整报告、实验轮次与数值边界', OUT/'REPORT.md')} · {link('端到端/分项图', OUT/'end_to_end_and_components.png')} · {link('等计算量成因图', OUT/'fragmentation_mechanism.png')} · {link('完整CSV', OUT/'full_metrics.csv')} · {link('归档及哈希', ARCHIVE/'manifest.json')}

本轮7次Guard均完成自有进程清理及租约释放。r3后续logprob验证失败，仅保留探索计时；最终比较使用r4原生与r5优化，数值验证另由r6/r7收紧。共享引擎默认未修改。
''')
    index = WORKSPACE/'experiments/experiment_summary/README.md'
    content = index.read_text()
    qrow = f"| {link('Q12：H100 PP Expert 执行碎片化优化', qpath)} | 同一对178节点H100 GPU0/1；35B BF16；8请求；冷Prefill4096→1，GPU缓存4096后Decode1→128；原生3轮、优化5轮。 | PP Decode **666.79→805.67 tok/s（+{gain:.2f}%）**，Expert GEMM **94.19→57.62µs（−{expert_drop:.2f}%）**；EP为863.20tok/s。Prefill无稳定收益，慢轮全部保留。 | 等输入/权重/路由的resident Expert维度控制支持布局成本；完整生成存在数值差异，尚未验证任务质量。7次Guard均清理和释放；完整图、逐请求CSV与来源见报告。 |"
    if '| [Q12：' in content:
        content='\n'.join(qrow if line.startswith('| [Q12：') else line for line in content.splitlines())+'\n'
    else:
        marker='\n<!-- END Q01 QWEN EP2 PP2 -->'
        assert marker in content
        content=content.replace(marker,qrow+'\n'+marker)
    index.write_text(content)

    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    marker=f'<!-- begin {ID} -->'
    entry=f'''{marker}
### {ID}

Snapshot: {ARCHIVE}

- Completed H100178 same NVLink GPU0/1 pair, full Qwen3.6-35B-A3B BF16, sequential EP2 (Attention DP2/TP1) and two-slot PP2 (20 layers/stage;local batch4). Eight requests: cold Prefill4096->1 and real GPU-cached4096+new1->128 Decode; preparation/JIT/profiles/validation excluded from timing.
- PP shape-adaptive T<=4 small-tile Triton, larger shapes native DeepGEMM: final Decode666.79->805.67 output tok/s (+{gain:.2f}%); TPOT11.176->9.091ms; E2E1532.34->1268.33ms. EP863.20 remains faster. Native Graph routed GEMM94.19->57.62us/layer (-{expert_drop:.2f}%), Attention core12.50->12.33us. Components are nonadditive; PP boundary residency includes waits.
- Same useful work, routing, selected weights and common bank addresses: native resident dimension32/64/128 costs47.16/56.48/70.15us on stage0 and60.54/67.16/80.65us on stage1. Compact DeepGEMM and new BF16 route GEMV are slower than small-tile Triton; tile tuning gives no stable further E2E gain.
- Prefill has no stable gain: mean78608.53->74679.43; median78508.29->78411.35; all five adaptive cohorts retained including one slow cohort. Same BF16 is not bitwise generation equivalence: layer L2~0.27-0.33%; atomic-batch native self64/64 identical; adaptive55 common-history positions,53 same argmax; no task-quality evaluation.
- Seven launches cleaned owned processes and released leases; one prepared-only directory excluded. R3 timing preserved as exploratory because later logprob feedback validation failed; final referenceR4/adaptiveR5 and numericR6/R7 complete. Private untimed metadata fix avoids nested-CUDA-pickle context OOM. Frozen engine clean; shared defaults/gateway untouched.

Artifacts: report, mean/median/P99 and round/request CSVs, components/routing/kernel CSVs, two PNG/PDF scientific figures, probability-position comparisons, private adapters, exact source/model revisions, run audit and hashes. Large traces/logs stay in original run paths; no authorization files, weights or KV bytes copied.

<!-- end {ID} -->
'''
    content=registry.read_text()
    if marker in content:
        start=content.index(marker);end=content.index(f'<!-- end {ID} -->',start)+len(f'<!-- end {ID} -->')
        content=content[:start]+entry.rstrip()+content[end:]
    else:
        content=content.replace('## Saved Experiments\n','## Saved Experiments\n\n'+entry,1)
    registry.write_text(content)
    print(json.dumps(dict(archive=str(ARCHIVE), report=str(OUT/'REPORT.md'), timed_cohorts=len(rounds),
                         timed_requests=len(requests), key_results=key_results),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
