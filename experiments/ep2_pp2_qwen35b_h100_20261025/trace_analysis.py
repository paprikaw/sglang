"""Correlate actual Decode Graph GPU kernels; do not add stage means into TPOT."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics


def summarize(run, followup=None):
    replacements = {}
    if followup:
        original_cfg=json.loads((run/'config.json').read_text())
        follow_cfg=json.loads((followup/'config.json').read_text())
        assert follow_cfg['decode_profile_only'] and follow_cfg['reference_run']==str(run)
        assert all(follow_cfg[k]==original_cfg[k] for k in
                   ('node','model_path','model_revision','engine_root','engine_commit','launch_env'))
        keys=('id','tp','ep','dp','pp','gpu_groups','replicas')
        assert [{k:x[k] for k in keys} for x in follow_cfg['layouts']]==[
            {k:x[k] for k in keys} for x in original_cfg['layouts'] if x['id']=='pp2']
        assert json.loads((followup/'completed.json').read_text())['passed']
        assert json.loads((followup/'worker_cleanup.json').read_text())['cleanup_verified']
        posts=[json.loads(p.read_text()) for p in (followup/'dashboard_guard/checks').glob('*-post.json')]
        assert len(posts)==1 and posts[0]['passed'] and posts[0]['model_cache']['passed']
        spec=followup/'dashboard_spec.json'
        assert hashlib.sha256(spec.read_bytes()).hexdigest()==posts[0]['dashboard']['spec_sha256']
        for item in json.loads(spec.read_text())['frozen_inputs']:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
        assert follow_cfg['reference_cases']==[r for r in json.loads((run/'completed.json').read_text())['cases']
                                              if r['id'].startswith('near_')]
        for case in follow_cfg['reference_cases']:
            directory=followup/'pp2/traces'/case['id']
            assert json.loads((directory/'sampling_validation.json').read_text())['passed']
            for path in directory.glob('*.trace.json.gz'):
                replacements[('pp2',case['id'],path.name)]=path
        assert len(replacements)==4
    rows, inventory = [], []
    for arm in ('ep2', 'pp2'):
        for path in sorted((run / arm / 'traces').glob('*/*.trace.json.gz')):
            if path.parent.name.endswith('_prefill'):
                continue
            original_path=path
            replacement=replacements.get((arm,path.parent.name,path.name))
            if replacement:
                with gzip.open(path,'rt') as handle:
                    previous=json.load(handle)['traceEvents']
                assert not any(e.get('name')=='cudaGraphLaunch' for e in previous)
                path=replacement
            with gzip.open(path, 'rt') as handle:
                trace = json.load(handle)
            events = trace['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel']
            launches = [e for e in events if e.get('name') == 'cudaGraphLaunch']
            decode_steps=[e for e in events if e.get('cat')=='user_annotation'
                          and e.get('name','').startswith('step[DECODE')]
            samples = []
            for launch in launches:
                correlation = launch['args']['correlation']
                selected = [e for e in kernels if e.get('args', {}).get('correlation') == correlation]
                if not selected:
                    continue
                # The wrapper disables Prefill Graph; matched Graph kernels
                # belong to actual Decode replays, not the initial Extend.
                assert all(e.get('args', {}).get('graph id') for e in selected)
                def total(predicate):
                    return sum(e['dur'] for e in selected if predicate(e['name'])) / 1000
                enclosing=[e for e in decode_steps
                           if (e.get('pid'),e.get('tid'))==(launch.get('pid'),launch.get('tid'))
                           and e['ts']<=launch['ts']<=e['ts']+e.get('dur',0)]
                step=min(enclosing,key=lambda e:e['dur'])['name'] if enclosing else ''
                batch=re.search(r'\bbs=(\d+)',step)
                samples.append(dict(
                    decode_step=step,actual_batch_size=int(batch[1]) if batch else None,
                    graph_span_ms=(max(e['ts']+e['dur'] for e in selected)-min(e['ts'] for e in selected))/1000,
                    kernel_sum_ms=total(lambda n:True),
                    deepep_kernel_sum_ms=total(lambda n:'deep_ep::' in n),
                    routed_gemm_kernel_sum_ms=total(lambda n:'deep_gemm::' in n and '(deep_gemm::GemmType)2' in n)))
            if not samples:
                raise AssertionError('No correlated actual Decode Graph GPU kernels: ' + str(path))
            case=next(x for x in json.loads((run/'completed.json').read_text())['cases']
                      if x['id']==path.parent.name)
            assert all(s['actual_batch_size']==case['requests']//2 for s in samples), (path,samples)
            row = dict(arm=arm, case=path.parent.name, rank_trace=path.name,
                       graph_replays=len(samples), unit='ms per actual rank Graph replay')
            row['actual_batch_sizes']=','.join(map(str,sorted({s['actual_batch_size'] for s in samples
                                                              if s['actual_batch_size'] is not None})))
            for key in ('graph_span_ms','kernel_sum_ms','deepep_kernel_sum_ms','routed_gemm_kernel_sum_ms'):
                row[key] = statistics.mean(s[key] for s in samples)
            rows.append(row)
            counts, durations = Counter(), Counter()
            for launch in launches:
                for e in kernels:
                    if e.get('args', {}).get('correlation') == launch['args']['correlation']:
                        counts[e['name']] += 1
                        durations[e['name']] += e['dur'] / 1000
            inventory.append(dict(trace=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  original_truncated_trace=str(original_path) if replacement else None,
                                  replacement_reason='Four-step window ended before Decode; isolated PP-only 16-step diagnostic' if replacement else None,
                                  graph_kernel_counts=dict(counts),
                                  graph_kernel_duration_ms=dict(durations), samples=samples))
    assert len(rows) == 12 + 2 * len(json.loads((run/'completed.json').read_text()).get('extra_cases',[])), len(rows)
    out = run / 'analysis'
    with (out / 'decode_graph_summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / 'decode_graph_inventory.json').write_text(json.dumps(dict(rows=inventory,
        limitations=['Profile is separate from formal metrics; profiler overhead is included here',
            'Kernel sums can overlap; not exclusive attention/expert/communication shares',
            'Routed GEMM classification excludes activation, permutation, shared expert and projections',
            'Pipeline boundary transfer and feedback occur outside model Graph',
            'Per-stage Graph times cannot be summed and called request TPOT',
            'Missing PP long-cache windows supplemented in a separate PP-only diagnostic; no formal results replaced']), indent=2) + '\n')
    text = ['## Decode Graph 资格检查', '',
        '通过 CUDA Graph launch 的 correlation ID 对齐实际 GPU kernel，排除首 token 的 Extend。'
        '以下是独立 profiler 样本，**不是正式 TPOT，也不是可相加的通信／计算独占占比**。'
        'Routed GEMM 只统计 DeepGEMM grouped expert GEMM，未包含激活、重排、共享专家；'
        '流水边界传输和反馈发生在模型 Graph 外。', '',
        '| 负载 | 配置 | rank trace | Graph 次数 | Graph GPU 跨度，毫秒 | DeepEP kernel 合计，毫秒 | Routed GEMM kernel 合计，毫秒 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['case']} | {r['arm'].upper()} | {r['rank_trace']} | {r['graph_replays']} | "
                    f"{r['graph_span_ms']:.3f} | {r['deepep_kernel_sum_ms']:.3f} | {r['routed_gemm_kernel_sum_ms']:.3f} |")
    text += ['', '[Graph 对照 CSV](decode_graph_summary.csv) · [逐 kernel 数据与分类边界](decode_graph_inventory.json)。', '']
    (out / 'GRAPH_PROFILE.md').write_text('\n'.join(text))
    print(json.dumps(dict(profile_rows=len(rows), output=str(out / 'GRAPH_PROFILE.md'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    summarize(parser.parse_args().run)
