"""Recompute raw cohorts, audit receipts, and compare existing moderate cells."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def csv_write(path, rows):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def analyze(run):
    run = run.resolve(); out = run / 'analysis'; out.mkdir(exist_ok=True)
    cfg = json.loads((run / 'config.json').read_text())
    spec = json.loads((run / 'dashboard_spec.json').read_text())
    posts = list((run / 'dashboard_guard/checks').glob('*-post.json'))
    assert len(posts) == 1, 'Need exactly one final Guard postcheck'
    post = json.loads(posts[0].read_text())
    assert post['passed'] and post['model_cache']['passed'], 'Guard or cache audit did not pass'
    for frozen in spec['frozen_inputs']:
        assert digest(frozen['path']) == frozen['sha256'], frozen['path']
    cleanup = json.loads((run / 'task/cleanup.json').read_text())
    assert cleanup['passed'] and cleanup['gpu_idle'] and cleanup['model_processes_exited']
    assert json.loads((run / 'task/progress.json').read_text())['valid_points'] == 4
    metrics, requests, source_files = [], [], []
    for arm in ('ep2', 'pp2'):
        assert json.loads((run / arm / 'cleanup.json').read_text())['cleanup_verified']
        quals = list((run / arm / 'qualification').glob('*.json'))
        assert len(quals) == 2
        for p in quals:
            d = json.loads(p.read_text())
            assert d['passed'] and d['layers'] == (40 if arm == 'ep2' else 20)
            assert d['experts_per_layer'] == (128 if arm == 'ep2' else 256)
        for phase in ('prefill', 'decode'):
            path = run / arm / f'metrics.{phase}.json'
            d = json.loads(path.read_text()); assert d['valid'] and not d['profiler_enabled']
            assert len(d['rounds']) == 3
            all_requests = []
            rates = []
            for index, group in enumerate(d['rounds']):
                rows = group['requests']; assert len(rows) == 8
                assert all(r['output_tokens'] == (1 if phase == 'prefill' else 128) for r in rows)
                assert all(r['prompt_tokens'] == (4096 if phase == 'prefill' else 4097) for r in rows)
                assert all(r['cached_tokens'] == (0 if phase == 'prefill' else 4096) for r in rows)
                duration = max(r['finished'] for r in rows) - min(r['started'] for r in rows)
                numerator = 8 * (4096 if phase == 'prefill' else 128)
                rate = numerator / duration
                field = 'input_tokens_per_s' if phase == 'prefill' else 'output_tokens_per_s'
                assert abs(rate - group[field]) < 1e-7
                rates.append(rate)
                for rid, r in enumerate(rows):
                    assert abs(r['e2e_s'] - (r['finished'] - r['started'])) < 1e-8
                    if phase == 'decode':
                        assert abs(r['tpot_ms'] - (r['e2e_s'] - r['ttft_s']) * 1000 / 127) < 1e-7
                    requests.append(dict(arm=arm, phase=phase, round=index + 1, request_index=rid,
                        **{k:r[k] for k in ('ttft_s','tpot_ms','e2e_s','output_tokens','cached_tokens','prompt_tokens')}))
                all_requests += rows
            row = dict(arm=arm, phase=phase, requests=8, rounds=3, cached_context=0 if phase=='prefill' else 4096,
                new_input=4096 if phase=='prefill' else 1, output=1 if phase=='prefill' else 128,
                tokens_per_s=statistics.mean(rates), round_min=min(rates), round_max=max(rates),
                round_std=statistics.stdev(rates), source=str(path), sha256=digest(path))
            for key in ('ttft_s', 'tpot_ms', 'e2e_s'):
                values = sorted(r[key] for r in all_requests if r[key] is not None)
                if not values:
                    continue
                for stat, value in [('mean', statistics.mean(values)), ('median', statistics.median(values)), ('p99', values[-1])]:
                    assert abs(d[key][stat] - value) < 1e-8
                    row[key+'_'+stat] = value
            metrics.append(row); source_files.append(dict(path=str(path), sha256=digest(path)))
    csv_write(out / 'full_metrics.csv', metrics); csv_write(out / 'per_request.csv', requests)
    # Historical cohorts retain their actual numerical and scheduling backends.
    history = []
    for model, gpu, reference, filter_case in [
        ('122B FP8', 'H100', WORKSPACE/'experiments/ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7/analysis/full_metrics.csv', True),
        ('35B BF16', 'H100', WORKSPACE/'experiments/ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3/analysis/full_metrics.csv', True)]:
        with reference.open() as f:
            for r in csv.DictReader(f):
                if r['case'] != 'moderate': continue
                history.append(dict(model=model,gpu=gpu,arm=r['arm'],phase=r['phase'],tokens_per_s=float(r['tokens_per_s']),
                    source=str(reference),source_sha256=digest(reference)))
    reference = WORKSPACE/'experiments/experiment_summary/qwen_ep2_pp2_metrics.csv'
    with reference.open() as f:
        for r in csv.DictReader(f):
            if r['configuration'] not in ('EP2','PP2，两槽'):continue
            history.append(dict(model='122B FP8',gpu='A100',arm='ep2' if r['configuration']=='EP2' else 'pp2',
                phase=r['workload'],tokens_per_s=float(r['throughput']),source=r['source_path'],source_sha256=r['source_sha256']))
    history += [dict(model='35B BF16',gpu='A100',arm=r['arm'],phase=r['phase'],tokens_per_s=r['tokens_per_s'],
                     source=r['source'],source_sha256=r['sha256']) for r in metrics]
    for row in history:
        assert digest(row['source']) == row['source_sha256']
    csv_write(out/'moderate_comparison.csv', history)
    comparisons = []
    for model in ('35B BF16','122B FP8'):
        for gpu in ('A100','H100'):
            for phase in ('prefill','decode'):
                cells = {r['arm']:r for r in history if (r['model'],r['gpu'],r['phase'])==(model,gpu,phase)}
                assert len(cells)==2
                comparisons.append(dict(model=model,gpu=gpu,phase=phase,
                    ep2=cells['ep2']['tokens_per_s'],pp2=cells['pp2']['tokens_per_s'],
                    pp_change_percent=(cells['pp2']['tokens_per_s']/cells['ep2']['tokens_per_s']-1)*100))
    csv_write(out/'relative_comparison.csv', comparisons)
    validation = dict(passed=True,formal_cells=4,rounds=12,request_timings=len(requests),
        exact_cache_hits=True,frozen_inputs_verified=len(spec['frozen_inputs']),guard_postcheck=str(posts[0]),
        guard_postcheck_sha256=digest(posts[0]),owned_cleanup=True,
        method='Independent rates and latency statistics recomputed from all raw request timings',sources=source_files)
    save(out/'validation.json',validation)
    plot_comparison(comparisons,out)
    print(json.dumps(dict(passed=True,metrics=metrics,comparisons=comparisons,output=str(out)),indent=2))


def plot_comparison(comparisons,out):
    # Standard scientific artifact; historical ratios are configuration cohorts, not a hardware-only control.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    order=[('35B BF16','A100'),('122B FP8','A100'),('35B BF16','H100'),('122B FP8','H100')]
    for ax,phase in zip(axes,('prefill','decode')):
        values=[next(r['pp_change_percent'] for r in comparisons if (r['model'],r['gpu'],r['phase'])==(m,g,phase)) for m,g in order]
        bars=ax.bar(range(4),values,color=['#4477aa' if v>=0 else '#cc6677' for v in values],width=.62)
        ax.axhline(0,color='black',lw=.7);ax.set_title(phase.capitalize());ax.set_ylim(-34,44)
        ax.set_xticks(range(4),[g+'\n'+m for m,g in order],fontsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for rect,v in zip(bars,values):ax.annotate(f'{v:+.1f}%',(rect.get_x()+rect.get_width()/2,v),
            ha='center',va='bottom' if v>=0 else 'top',xytext=(0,3 if v>=0 else -3),textcoords='offset points',fontsize=9)
    axes[0].set_ylabel('PP2 throughput change vs EP2 (%)')
    fig.suptitle('Fixed 2 GPUs, 8 requests; PP2 has two slots',fontsize=12)
    fig.text(.5,.01,'Three-round means. Native kernels / Graph / overlap settings differ; see report.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run)
