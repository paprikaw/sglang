"""Summarize existing Python-visible local forward shapes; no new GPU sampling."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

from completed_episode_files import completed_formals, gate_sha256


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if rows:
        with path.open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    elif path.exists():
        path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = read(run/'config.json')
    files, pending = completed_formals(run)
    rank_rows, histograms, trials, sources = [], [], [], []
    for formal in files:
        arm, case, policy = formal.relative_to(run).parts[1:4]
        repeat = formal.stem.split('.')[-1]
        path = formal.with_name('forwards.'+repeat+'.json')
        data = read(path)
        assert len(data['delta']) == 8
        role_groups = defaultdict(list)
        for rank, value in data['delta'].items():
            node, parsed_arm, instance, pp, tp = rank.split('.')
            assert parsed_arm == arm
            shape = value['shape']
            assert shape['pp_rank'] == int(pp[2:]) and shape['tp_rank'] == int(tp[2:])
            # PP stages hold the same sequence batch; count the head once.
            if shape['pp_rank'] != 0:
                continue
            info = read(run/'nodes'/node/arm/'ready.json')['server_info'][int(instance[1:])]
            assert info['pp_size'] == shape['pp_size'] and info['tp_size'] == shape['tp_size']
            if shape['tp_size'] > 1:
                assert info['enable_dp_attention'] and info['dp_size'] == shape['tp_size']
            roles = {s['disaggregation_mode'] for s in info['internal_states']}
            assert len(roles) == 1
            native_role = roles.pop()
            role = 'colocated' if native_role in (None, 'null') else native_role
            identity = dict(arm=arm, case=case, policy=policy, repeat=repeat, role=role, rank=rank)
            calls = token_rows = 0
            for key, count in value['counts'].items():
                match = re.fullmatch(r'DECODE:b(\d+)', key)
                if not match:
                    continue
                batch = int(match.group(1))
                assert count >= 0 and batch > 0
                calls += count
                token_rows += count * batch
                histograms.append(dict(**identity, local_batch=batch, forward_calls=count))
            row = dict(**identity, decode_forward_calls=calls, local_batch_rows=token_rows,
                mean_local_decode_batch=token_rows/calls if calls else None,
                idle_forward_calls=value['counts'].get('IDLE:b0', 0))
            rank_rows.append(row)
            role_groups[role].append(row)
        for role, rows in role_groups.items():
            calls = sum(r['decode_forward_calls'] for r in rows)
            tokens = sum(r['local_batch_rows'] for r in rows)
            trials.append(dict(arm=arm, case=case, policy=policy, repeat=repeat,
                role=role, cache_owners=len(rows), decode_forward_calls=calls,
                local_batch_rows=tokens, mean_local_decode_batch=tokens/calls if calls else None))
        sources.append(dict(path=str(path), sha256=sha(path), formal=str(formal), formal_sha256=sha(formal)))
    groups = defaultdict(list)
    for row in trials:
        groups[(row['arm'], row['case'], row['policy'], row['role'])].append(row)
    means = []
    for key, rows in groups.items():
        assert len({r['cache_owners'] for r in rows}) == 1
        values = [r['mean_local_decode_batch'] for r in rows if r['mean_local_decode_batch'] is not None]
        means.append(dict(zip(('arm','case','policy','role'), key)))
        means[-1].update(repeats=len(rows), planned_repeats=cfg['repeats'],
            cache_owners=rows[0]['cache_owners'],
            mean_local_decode_batch=statistics.mean(values) if values else None,
            mean_decode_forward_calls=statistics.mean(r['decode_forward_calls'] for r in rows))
    out = run/'analysis'
    out.mkdir(exist_ok=True)
    for name, rows in [('forward_batch_ranks.csv',rank_rows), ('forward_batch_histograms.csv',histograms),
                       ('forward_batch_trials.csv',trials), ('forward_batch_means.csv',means)]:
        write_csv(out/name, rows)
    lines = ['# 实际本地Decode forward的batch形状', '',
        '使用已保存的计时前后计数差，没有新增GPU采样。只纳入有完成标记的episode；整组验收仍另行核对。', '',
        '|配置|负载|角色|轮数|独立缓存位置|平均本地Decode batch|每轮Decode forward调用数|',
        '|---|---|---|---:|---:|---:|---:|']
    for row in means:
        value = row['mean_local_decode_batch']
        shown = '无纯Decode调用' if value is None else f'{value:.3f}'
        lines.append(f"|{row['arm']}|{row['case']}|{row['role']}|{row['repeats']}/{row['planned_repeats']}|"
            f"{row['cache_owners']}|{shown}|{row['mean_decode_forward_calls']:.1f}|")
    lines += ['', '每轮按纯DECODE模式的本地ModelRunner.forward调用加权batch，然后对完整轮次取算术平均。'
        'PP只取stage0，EP在已核对Attention-DP=TP-size的配置中取每个DP rank。'
        'PP的单次stage0 forward batch不表示整个实例的全部在途请求。'
        '不包含EXTEND/MIXED模式里的请求，也不是按时间加权的并发、全EP组token数或Expert收到的token-expert行数。'
        'IDLE调用不进入Decode均值；空rank仍可能执行其他rank路由过来的Expert工作。', '',
        'dispatch_output计数只记录Python可见的DeepEPDispatcher.dispatch返回值；CUDA Graph重放不逐次经过该hook，'
        '因此不能用Normal/LL计数推导实际通信后端比例。graph_eligible也只表示条件检查。', '',
        '[逐rank](forward_batch_ranks.csv) · [原始batch计数](forward_batch_histograms.csv) · '
        '[每轮](forward_batch_trials.csv) · [均值](forward_batch_means.csv) · [审计](forward_batch_audit.json)']
    (out/'FORWARD_BATCHES.md').write_text('\n'.join(lines)+'\n')
    (out/'forward_batch_audit.json').write_text(json.dumps(dict(sources=sources,
        excluded_unmarked_formals=pending, completion_gate_sha256=gate_sha256(),
        analyzer_sha256=sha(Path(__file__)), pp_stage_deduplicated=True,
        attention_dp_layout_checked=True, scope='Pure DECODE local forward shape; not time-weighted concurrency or Expert rows'),indent=2)+'\n')
    print(json.dumps(dict(run=run.name, episodes=len(files), points=len(means))))


if __name__ == '__main__':
    main()
