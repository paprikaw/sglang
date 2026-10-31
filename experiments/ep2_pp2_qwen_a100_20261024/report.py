"""CPU-only audit and report; never modifies a frozen run or Guard evidence."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def audit_metrics(path):
    data = load(path)
    assert data['valid'] is True and data['profiler_enabled'] is False
    assert len(data['rounds']) == 3
    phase = path.stem.split('.')[-1]
    requests = []
    for row in data['rounds']:
        assert len(row['requests']) == 8
        duration = max(r['finished'] for r in row['requests']) - min(r['started'] for r in row['requests'])
        assert math.isclose(duration, row['makespan_s'], rel_tol=1e-10)
        assert math.isclose(sum(r['output_tokens'] for r in row['requests']) / duration,
                            row['output_tokens_per_s'], rel_tol=1e-10)
        for request in row['requests']:
            assert request['output_tokens'] == (1 if phase == 'prefill' else 128)
            assert request['ttft_s'] > 0 and request['e2e_s'] >= request['ttft_s']
            assert math.isclose(request['finished'] - request['started'], request['e2e_s'], rel_tol=1e-10)
            if phase == 'decode':
                assert request['cached_tokens'] >= 4032
                assert math.isclose((request['e2e_s'] - request['ttft_s']) * 1000 / 127,
                                    request['tpot_ms'], rel_tol=1e-10)
        requests.extend(row['requests'])
    for name in ('ttft_s', 'tpot_ms', 'e2e_s'):
        values = [r[name] for r in requests if r[name] is not None]
        if values:
            assert math.isclose(statistics.mean(values), data[name]['mean'], rel_tol=1e-10)
            assert math.isclose(statistics.median(values), data[name]['median'], rel_tol=1e-10)
            assert max(values) == data[name]['p99']  # 24 observations, nearest-rank P99
    return data, dict(path=str(path.resolve()), sha256=sha(path), requests=len(requests), passed=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def build(ep_run, pp_run, out):
    out.mkdir(parents=True, exist_ok=True)
    sources = [('EP2', 'ep2', ep_run), ('PP2 two slots', 'pp2', pp_run)]
    metric_rows, stage_rows, audit, profiles, postchecks = [], [], [], [], []
    for label, arm, run in sources:
        cfg = load(run/'config.json')
        cleanup = load(run/arm/'cleanup.json')
        assert cleanup['cleanup_verified'] and cleanup['node_gpu_idle_observed']
        quals = sorted((run/arm/'qualification').glob('*.json'))
        assert len(quals) == 2
        for path in quals:
            q = load(path)
            assert q['passed'] and q['layers'] == (48 if arm == 'ep2' else 24)
            assert q['experts_per_layer'] == (128 if arm == 'ep2' else 256)
        checks = [load(p) for p in (run/'dashboard_guard/checks').glob('*-post.json')]
        assert len(checks) == 1
        check = checks[0]
        assert check['cleanup_verified'] and check['model_cache']['passed']
        postchecks.append(dict(run=str(run.resolve()), guard_passed=check['passed'],
                               cache_passed=True, cleanup_passed=True,
                               exit_code=check['exit_code']))
        if arm == 'pp2':
            assert check['passed'], 'Final PP2 run must finish cleanly'
            assert cfg['pp_async_batch_depth'] == 0 and cfg['pp_loop_slots'] == 2
        elif not check['passed']:
            # Do not relabel the Guard run. Audit completed measurements while
            # preserving the exact wrapper failure and all original evidence.
            assert 'SystemExit: 0' in (run/'controller.log').read_text()
        for phase in ('prefill', 'decode'):
            data, evidence = audit_metrics(run/arm/f'metrics.{phase}.json')
            audit.append(evidence)
            row = dict(configuration=label, workload=phase, source_run=str(run.resolve()),
                       throughput_tokens_per_s=data['input_tokens_per_s' if phase == 'prefill' else 'output_tokens_per_s'],
                       throughput_unit='input tok/s' if phase == 'prefill' else 'output tok/s')
            for name in ('ttft_s', 'tpot_ms', 'e2e_s'):
                for stat in ('mean', 'median', 'p99'):
                    row[name+'.'+stat] = data[name][stat] if data[name] else None
            metric_rows.append(row)
        for trace in (run/'analysis').glob(f'{arm}-*-TP-*.json'):
            data = load(trace)
            phase = 'decode' if '-decode-' in trace.name else 'prefill'
            expected = set(range(48)) if arm == 'ep2' else (
                set(range(24,48)) if '-PP-1.' in trace.name else set(range(24)))
            assert {r['layer'] for r in data['invocations']} == expected
            profiles.append(dict(configuration=label, workload=phase, trace=str(trace.resolve()),
                                 raw_trace=str(Path(data['path']).resolve()), raw_sha256=data['sha256']))
            for row in data['layer_means']:
                stage_rows.append(dict(configuration=label, workload=phase, trace=trace.name, **row))
    assert len(profiles) == 8
    write_csv(out/'full_metrics.csv', metric_rows)
    write_csv(out/'stage_metrics.csv', stage_rows)
    (out/'data_validation.json').write_text(json.dumps(dict(measurement_data_passed=True,
        original_guard_postchecks=postchecks, metrics=audit, traces=profiles,
        original_records_unchanged=True,
        warning='EP2 r7 completed all measurements but wrapper caught SystemExit(0); original Guard failure is retained.'), indent=2)+'\n')
    lines = ['# Qwen3.5-122B：同一对 A100 上的 EP2 与 PP2', '',
        '完整 48 层文本模型；序列化 FP8 权重约 118.42 GiB。固定 spartan-gpgpu069 的 GPU0／1，NV12；只使用已有预约。', '',
        '| 配置 | 每卡模型层数 | 每层每卡 routed experts | Attention | GPU 间通信 |',
        '|---|---:|---:|---|---|',
        '| EP2 | 48 | 128 | TP1 × DP2 | 每层 DeepEP normal NVLink Dispatch／Combine |',
        '| PP2，两槽 | 24 | 256 | 每 stage TP1 | stage 边界激活、反馈；无跨卡专家交换 |', '',
        '两组同用 FP8 W8A16 Marlin、BF16 激活与 KV、Triton Attention；CUDA Graph 和 overlap schedule 均关闭。A100 不使用原生 FP8 Tensor Core。', '',
        '| 场景 | 请求数 | 每条已缓存上下文 | 每条本轮新输入 | 每条输出 |',
        '|---|---:|---:|---:|---:|',
        '| Prefill | 8 | 0 | 4096 | 1 |',
        '| Decode | 8 | 4096，真实模型生成、GPU 常驻 | 1 | 128 |', '',
        '完整权重从 canonical CPU bank 加载，加载、首次编译、上下文构建和预热不计时。不使用合成 KV 或 CPU HiCache 换入；本轮容量充足。Decode 的首 token 包含新增输入的初始 Extend，TPOT 只统计首 token 之后。', '',
        '## 未开启 profiler 的性能', '',
        '每项 3 轮、每轮 8 请求。延迟表均给出 mean / median / P99；24 个观测的 nearest-rank P99 等于样本最大值，不代表稳定服务尾延迟。吞吐按每轮 cohort makespan 计算后取均值。', '',
        '| 场景 | 配置 | 吞吐 token/s ↑ | TTFT 秒 ↓ | TPOT 毫秒 ↓ | 端到端秒 ↓ |',
        '|---|---|---:|---|---|---|']
    for row in metric_rows:
        def triple(name):
            return '—' if row[name+'.mean'] is None else ' / '.join(f'{row[name+"."+stat]:.4f}' for stat in ('mean','median','p99'))
        lines.append(f'| {row["workload"]} | {row["configuration"]} | {row["throughput_tokens_per_s"]:.2f} | {triple("ttft_s")} | {triple("tpot_ms")} | {triple("e2e_s")} |')
    by_key = {(r['configuration'], r['workload']):r for r in metric_rows}
    ep_decode = by_key[('EP2','decode')]
    pp_decode = by_key[('PP2 two slots','decode')]
    ep_prefill = by_key[('EP2','prefill')]
    pp_prefill = by_key[('PP2 two slots','prefill')]
    lines += ['', '## 本轮结论', '',
        f'- 固定最终两槽配置：Decode 的 PP2 吞吐比 EP2 高 {(pp_decode["throughput_tokens_per_s"]/ep_decode["throughput_tokens_per_s"]-1)*100:.1f}%，平均 TPOT 低 {(1-pp_decode["tpot_ms.mean"]/ep_decode["tpot_ms.mean"])*100:.1f}%。',
        f'- 同样的两槽配置下，Prefill 的 EP2 吞吐比 PP2 高 {(ep_prefill["throughput_tokens_per_s"]/pp_prefill["throughput_tokens_per_s"]-1)*100:.1f}%。这不是所有场景下 PP 都更快的结论。',
        '- PP 的 Decode 每层 routed Expert GPU 成本略增，Attention 接近，避免了 EP2 的逐层 Dispatch／Combine。客户端改善还包含流水和 CPU 控制路径的影响，不能把全部 TPOT 差额都归给链路时间。', '',
        '### 槽位设置的独立影响', '',
        '原生循环槽数为 PP stage 数＋额外异步深度。旧 `pp_async_batch_depth=2` 实际为四槽，最终设为 0 才是两槽。模型、GPU、请求及计算内核不变；下表保留旧测量，不将其混进最终两槽均值。', '',
        '| 场景 | 四槽 PP2 吞吐 token/s | 两槽 PP2 吞吐 token/s | 四槽 TPOT ms | 两槽 TPOT ms |',
        '|---|---:|---:|---:|---:|']
    sensitivity = []
    for phase in ('prefill','decode'):
        old, _ = audit_metrics(ep_run/'pp2'/f'metrics.{phase}.json')
        new = by_key[('PP2 two slots',phase)]
        old_rate = old['input_tokens_per_s' if phase == 'prefill' else 'output_tokens_per_s']
        old_tpot = old['tpot_ms']['mean'] if old['tpot_ms'] else None
        new_tpot = new['tpot_ms.mean']
        sensitivity.append(dict(workload=phase, old_slots=4, new_slots=2,
            old_throughput=old_rate,new_throughput=new['throughput_tokens_per_s'],
            old_tpot_ms=old_tpot,new_tpot_ms=new_tpot,
            old_source=str((ep_run/'pp2'/f'metrics.{phase}.json').resolve())))
        t0='—' if old_tpot is None else f'{old_tpot:.2f}'
        t1='—' if new_tpot is None else f'{new_tpot:.2f}'
        lines.append(f'| {phase} | {old_rate:.2f} | {new["throughput_tokens_per_s"]:.2f} | {t0} | {t1} |')
    write_csv(out/'slot_comparison.csv',sensitivity)
    lines += ['', '这也说明 Prefill 与 Decode 的槽位选择可能不同；本轮只补一个两槽对照，没有进行完整调参扫描。', '',
        '## 逐层 GPU profile', '',
        '独立于上述计时请求采集。原始八份 trace 的每层数据见 `stage_metrics.csv`；Attention 包括投影，routed Expert 与 shared Expert 分开。CPU 注解与 GPU 镜像注解去重，GPU kernel 由实际 launch correlation 归属。不同 forward mode、microbatch 形状不混合。', '',
        '**以下为每卡、一次实际层调用的 GPU 活动均值，单位毫秒；不是每输出 token 的端到端时间。** Full Attention 只平均 Full Attention 层，linear 只平均线性 Attention 层；Expert、Dispatch、Combine 平均对应形状的全部层。', '',
        '| 场景与实际本地形状 | 卡／stage | Full Attention | Linear Attention | Routed Expert | Shared Expert | Dispatch | Combine |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for item in profiles:
        data = load(Path(item['trace']))
        phase = item['workload']
        selected = [r for r in data['invocations'] if (
            (phase == 'decode' and r['step'].startswith('step[DECODE ') and re.search(r'\bbs=4(?:\s|\])', r['step']))
            or (phase == 'prefill' and r['step'].startswith('step[EXTEND bs=1 toks=4096 ')))]
        assert selected, f'No comparable actual shape in {item["trace"]}'
        def mean(label, only_present=False):
            values = [r['gpu_component_union_ms'].get(label,0) for r in selected
                      if not only_present or label in r['gpu_component_union_ms']]
            return statistics.mean(values) if values else None
        values = [mean('attention.full',True), mean('attention.linear',True),
                  mean('expert.routed'),mean('expert.shared'),mean('communication.dispatch'),mean('communication.combine')]
        rank = Path(item['trace']).stem.replace('-decode','').replace('-prefill','')
        shape = 'Decode，每卡 4 请求' if phase == 'decode' else 'Prefill，每卡 1 请求 × 4096 新 token'
        lines.append('| '+shape+' | '+rank+' | '+' | '.join('—' if v is None else f'{v:.4f}' for v in values)+' |')
    lines += ['', '**PP 的表内 Dispatch／Combine 为零，只表示没有跨卡专家交换，不能理解为没有通信。** PP 的 NCCL stage 传输在层范围外；NCCL kernel 驻留时间可包含等待，不能直接当作链路传输延迟。GPU 分项也可能重叠，不能用相加值冒充客户端关键路径。', '',
        '## 完整性与限制', '',
        '- EP2 与 PP2 都检查完整层数；EP2 新通信接口使用真实一层、不同输入与稀疏／空目标 rank 进行数值对照。PP2 也经过真实 EP1 层 forward 检查。',
        '- EP2 来源 r7：四个性能点及八份 trace 已完整写出，canonical cache 审计和 GPU 清理通过；退出包装器误捕获 `SystemExit(0)`，因此原始 Guard 状态仍为失败，未补写成通过。数据单独校验见 `data_validation.json`。',
        '- 最终 PP2 来源 r8：仅补测 PP2，将额外异步槽数从 2 改成 0，使原生循环槽数从 4 改为 2；保留原 EP2 基线，不混用旧 PP 四槽结果。',
        '- 原始 r7 的 PP 曾有 2／4／2 动态 microbatch。最终两槽的实际形状可在原始 `step[...]` 注解与逐层表中检查；不把启动参数当作持续满流水的证据。',
        '- 此轮没有多组 GPU 型号、通信域或容量边界扫描；不能据此断言所有 EP 或 PP 部署的优劣。', '',
        '## 来源', '']
    for label, arm, run in sources:
        lines.append(f'- {label}：[冻结配置]({run.resolve()}/config.json)，[原始测量与 traces]({run.resolve()}/{arm})。')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    manifest = dict(id='qwen122_a100_nvlink_ep2_pp2_20261025',
        title='Qwen3.5-122B A100 EP2 versus PP2, ample-capacity Prefill and cached Decode',
        created_at=datetime.now(timezone.utc).isoformat(), status='complete-measurements-with-retained-wrapper-failure',
        engine='sglang', worktree=load(ep_run/'config.json')['engine_root'],
        source_commit=load(ep_run/'config.json')['engine_commit'],
        source_tag='experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated',
        model=load(ep_run/'config.json')['model_path'],
        model_revision=load(ep_run/'config.json')['model_revision'],
        node='spartan-gpgpu069', gpu_ids=[0,1], registered_gpus=4, used_gpus=2,
        topology='NV12 pair', repetitions=3, requests_per_round=8,
        sources=profiles, guard_postchecks=postchecks,
        artifacts={p.name:dict(path=str(p.resolve()),sha256=sha(p)) for p in out.iterdir()
                   if p.is_file() and p.name != 'manifest.json'},
        caveat='No model/cache-load, JIT, prewarm, CPU KV restore or profiler time in unprofiled request metrics')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(out=str(out.resolve()),metrics=len(metric_rows),layer_rows=len(stage_rows),traces=len(profiles))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep-run', type=Path, required=True)
    ap.add_argument('--pp-run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    build(args.ep_run, args.pp_run, args.out)
