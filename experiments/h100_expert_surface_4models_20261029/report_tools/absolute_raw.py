"""Present existing H100 CUDA-event measurements without aggregation."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_four_model_expert_surface_20261029')
LABELS = {
    'glm47': 'GLM-4.7 AWQ W4A16 (BZP TMA off)',
    'hy3': 'HY3 BF16',
    'qwen35b': 'Qwen3.6-35B-A3B BF16',
    'qwen122b': 'Qwen3.5-122B-A10B FP8',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join(['---'] * len(headers)) + '|',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
    ])


def replace_index(path, body):
    marker = 'h100_four_model_expert_surface_20261029'
    begin, end = f'<!-- begin {marker} -->', f'<!-- end {marker} -->'
    original = path.read_text()
    start = original.index(begin)
    stop = original.index(end, start) + len(end)
    content = original[:start] + begin + '\n' + body.strip() + '\n' + end + original[stop:]
    assert path.read_text() == original, 'Concurrent index edit; retry against current contents'
    temporary = path.with_name(path.name + f'.absolute-raw-{os.getpid()}.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def main(run, event):
    assert 1 <= event <= 30
    report = run.parents[1] / 'report'
    result_path = run / 'profile/result.json'
    result = json.loads(result_path.read_text())
    config = json.loads((run / 'config.json').read_text())
    sources = json.loads((report / 'sources.json').read_text())
    manifest = json.loads((SNAPSHOT / 'manifest.json').read_text())
    assert result['passed'] and len(result['rows']) == 1584
    assert all(item['passed'] for item in result['cleanup'])
    assert sha(result_path) == sources['result']['sha256']
    assert manifest['gpu_release']['state'] == 'released'
    assert manifest['gpu_release']['cleanup_verified'] is True

    workers = {worker['worker']: worker for worker in result['workers']}
    lookup, selected, raw = {}, [], []
    for row in result['rows']:
        worker = workers[row['worker']]
        values = row['raw_ms']
        assert len(values) == 30 and all(math.isfinite(x) and x > 0 for x in values)
        key = (worker['replica'], row['model'], row['mode'], row['T'], row['E'])
        assert key not in lookup
        lookup[key] = values[event - 1]
        identity = dict(model=row['model'], worker=row['worker'], node=row['node'],
                        physical_gpu=row['physical_gpu'], replica=worker['replica'],
                        mode=row['mode'], T_assignment_rows=row['T'], E_active=row['E'])
        selected.append(dict(identity, event_1based=event, latency_ms=values[event - 1]))
        raw.append(dict(identity, **{f'event_{index + 1:02d}_ms': value
                                    for index, value in enumerate(values)}))
    assert len(lookup) == 1584
    write_csv(report / 'absolute_single_event.csv', selected)
    write_csv(report / 'absolute_all_events.csv', raw)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    for replica in (1, 2):
        for mode in ('cuda_graph', 'eager'):
            figure, axes = plt.subplots(2, 2, figsize=(16, 10), layout='constrained')
            for axis, (model, model_config) in zip(axes.flat, config['models'].items()):
                worker = next(w for w in workers.values()
                              if w['model'] == model and w['replica'] == replica)
                tokens, experts = config['plan']['tokens'], model_config['experts']
                matrix = np.full((len(experts), len(tokens)), np.nan)
                for y, e in enumerate(experts):
                    for x, t in enumerate(tokens):
                        if e <= t:
                            matrix[y, x] = lookup[replica, model, mode, t, e]
                finite = matrix[np.isfinite(matrix)]
                scale = LogNorm(vmin=finite.min(), vmax=finite.max())
                picture = axis.imshow(matrix, origin='lower', aspect='auto',
                                      cmap='viridis', norm=scale)
                axis.set_xticks(range(len(tokens)), [str(t) for t in tokens], rotation=35)
                axis.set_yticks(range(len(experts)), experts)
                axis.set_xlabel('Received token-expert assignment rows T')
                axis.set_ylabel('Active experts E')
                axis.set_title(f"{LABELS[model]}\n{worker['node']} / GPU {worker['physical_gpu']}")
                for y in range(len(experts)):
                    for x in range(len(tokens)):
                        value = matrix[y, x]
                        if np.isfinite(value):
                            axis.text(x, y, f'{value:.4f}', ha='center', va='center',
                                      fontsize=6.1, color='white' if scale(value) < .65 else 'black')
                figure.colorbar(picture, ax=axis, shrink=.8, label='Absolute latency (ms; log color scale)')
            mode_label = 'CUDA Graph' if mode == 'cuda_graph' else 'eager'
            figure.suptitle(f'H100 absolute latency (ms) | {mode_label} | recorded event {event}/30\n'
                           f'Independent run {replica}; each cell is one recorded CUDA event', fontsize=14)
            stem = f'absolute_raw_{mode}_rep{replica}'
            for extension in ('png', 'pdf'):
                figure.savefig(report / f'{stem}.{extension}', dpi=180)
            plt.close(figure)

    now = datetime.datetime.now(ZoneInfo('Australia/Melbourne')).isoformat()
    view = dict(updated_at=now, unit='ms', aggregation='none',
                selected_event_1based=event, timing_block_1based=(event - 1) // 10 + 1,
                sample_in_block_1based=(event - 1) % 10 + 1,
                across_gpu_aggregation=False, across_sample_aggregation=False,
                accepted_result=dict(path=str(result_path), sha256=sha(result_path)),
                selected_rows=len(selected), raw_cuda_events=len(raw) * 30,
                workers=[{key: w[key] for key in ('worker', 'model', 'node', 'physical_gpu', 'replica')}
                         for w in workers.values()])
    write_json(report / 'absolute_raw_view.json', view)

    previous = report / 'REPORT_prior_aggregate_view.md'
    if not previous.exists():
        shutil.copyfile(report / 'REPORT.md', previous)
    sections = [
        '# H100 四模型：逐 GPU 的绝对原始耗时', '',
        f'更新时间：{now}。所有图表单位均为 **毫秒（ms）**。', '',
        f'每个格子直接使用该 GPU 已保存的第 **{event} / 30** 次 CUDA event 计时；'
        f'对应第 {(event - 1) // 10 + 1} 个计时块中的第 {(event - 1) % 10 + 1} 次。'
        '不做归一化，不计算均值或中位数，不合并两张 GPU。展示保留四位小数，CSV 保留原始精度。', '',
        'T 是单卡实际接收的 token–expert 配对行数，E 是有输入的活跃专家数；采用均衡路由。'
        '测真实单层 Expert 的两次 GEMM、SiLU/multiply 及 FP8 内部激活量化；'
        '输入准备、初始量化、Attention、router、shared expert、通信及模型加载不计时。', '',
        '## CUDA Graph：两次独立测量分别展示', '',
    ]
    anchor_rows = {}
    for replica in (1, 2):
        node = next(w['node'] for w in workers.values() if w['replica'] == replica)
        values = []
        for model in LABELS:
            worker = next(w for w in workers.values() if w['model'] == model and w['replica'] == replica)
            values.append([LABELS[model], str(worker['physical_gpu'])] +
                          [f'{lookup[replica, model, "cuda_graph", t, e]:.4f}'
                           for t in (128, 8192, 32768) for e in (2, 32)])
        anchor_rows[str(replica)] = values
        sections += [f'### 独立测量 {replica}：{node}', '',
                     f'![绝对原始耗时](absolute_raw_cuda_graph_rep{replica}.png)', '',
                     table(['模型', 'GPU', 'T128 E2', 'T128 E32', 'T8192 E2', 'T8192 E32',
                            'T32768 E2', 'T32768 E32'], values), '']
    sections += [
        '## 完整数值与其他执行模式', '',
        '- [当前展示的每格单次计时 CSV](absolute_single_event.csv)。',
        '- [每格全部 30 次原始计时 CSV](absolute_all_events.csv)。',
        '- [eager：独立测量 1](absolute_raw_eager_rep1.png) · [独立测量 2](absolute_raw_eager_rep2.png)。',
        '- [CUDA Graph PDF：测量 1](absolute_raw_cuda_graph_rep1.pdf) · [测量 2](absolute_raw_cuda_graph_rep2.pdf)。',
        '- [展示口径与节点/GPU 映射](absolute_raw_view.json) · [来源校验](sources.json)。', '',
        '## 配置与验证', '',
        'GLM-4.7 为 AWQ/Humming grouped masked，驻留 40 个真实专家；'
        '使用 `use_tma_bzp=False` 兼容配置绕过 grouped 零点参数的 TMA 描述符错误。'
        'HY3 为 BF16 DeepGEMM，驻留 48 个专家；35B 为 BF16 DeepGEMM，122B 为 FP8 DeepGEMM，均驻留 256 个专家。'
        'DeepGEMM 每专家按 128 行对齐，补齐开销包含在测量中。不同模型使用不同精度和内核。', '',
        '沿用已经通过数值检查和 Guard 清理检查的原实验；这里只改变已有数据的展示方式。'
        '原始结果、冻结配置、计时顺序、模型权重摘要和源版本均保持原样。', '',
        f'[原始结果]({result_path}) · [冻结配置]({run / "dashboard_spec.json"})。', '',
    ]
    (report / 'REPORT.md').write_text('\n'.join(sections))
    (report / 'REPRODUCE.md').write_text(
        '# Reproduce the absolute raw view on CPU\n\n'
        f'Run `/home/bxb1/conda/bin/python {Path(__file__).resolve()} --run {run} --event {event}`.\n\n'
        'Each displayed value is read directly from raw_ms[event - 1] in the accepted result JSON. '
        'No reduction across samples or GPUs is performed. The source experiment is not rerun.\n')
    shutil.copyfile(Path(__file__).resolve(), report / 'absolute_raw.py')
    sources['absolute_raw_generator'] = dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__)))
    sources['default_presentation'] = view
    sources['artifacts'] = {p.name: sha(p) for p in report.iterdir()
                            if p.is_file() and p.name != 'sources.json'}
    write_json(report / 'sources.json', sources)
    for path in report.iterdir():
        if path.is_file():
            shutil.copyfile(path, SNAPSHOT / path.name)
    manifest['updated_at'] = now
    manifest['presentation'] = view
    manifest['artifacts'] = {p.name: dict(path=str(SNAPSHOT / p.name), sha256=sha(p))
                             for p in report.iterdir() if p.is_file()}
    write_json(SNAPSHOT / 'manifest.json', manifest)

    body = f'''### H100 四模型：逐 GPU 的绝对原始耗时

GLM-4.7 AWQ、HY3 BF16、Qwen3.6-35B BF16、Qwen3.5-122B FP8 的真实单层 Expert 二维图。T=8..32768 个本地 token–expert 配对行，E 独立控制。两个 H100 节点，每模型两份独立单 GPU 测量。

默认展示各 GPU 的第 {event}/30 次原始 CUDA event，单位 ms；不归一化，不计算均值或中位数，不合并 GPU。Graph 与 eager 分开，所有 47520 次原始计时均已保存。GLM 使用 use_tma_bzp=False 兼容配置。原实验的数值和清理检查已通过，当前更新只重画已有记录。

[绝对耗时报告]({SNAPSHOT / 'REPORT.md'}) · [每格原始计时]({SNAPSHOT / 'absolute_single_event.csv'}) · [全部 30 次原始值]({SNAPSHOT / 'absolute_all_events.csv'})。Source run: {run}。'''
    replace_index(Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md'), body)
    replace_index(BASE.parent / 'experiment_summary/README.md', body)
    for path in report.iterdir():
        if path.is_file():
            assert sha(path) == sha(SNAPSHOT / path.name)
    print(json.dumps(dict(report=str(SNAPSHOT / 'REPORT.md'), selected_event=event,
                          selected_rows=len(selected), raw_events=len(raw) * 30,
                          anchor_rows_ms=anchor_rows), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=BASE / 'v4/runs/r6')
    parser.add_argument('--event', type=int, default=1)
    arguments = parser.parse_args()
    main(arguments.run.resolve(), arguments.event)
