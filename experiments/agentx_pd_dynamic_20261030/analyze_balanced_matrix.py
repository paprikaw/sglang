"""Compare only the matched, qualified E5 AgentX matrix; preserve partial points."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics as st

ROOT = Path(__file__).resolve().parent
RUNS = (
    'eight_gpu_col_ep4_durable_r3',
    'eight_gpu_pd_ep4_durable_slots8_r4',
    'eight_gpu_col_pp2_durable_m16_r3',
    'eight_gpu_col_single_durable_kv40_r3',
)
EXPECTED_COMMIT = '37e0828c642ce4f3795c4c4394f4ce913bcd1a35'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def csv_write(p, rows):
    if rows:
        with p.open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'matrix_analysis')
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    trials, sources, run_status, rank_rows, slo_rows, capacities, root_rows = [], [], [], [], [], [], []
    reference = None
    for name in RUNS:
        run = ROOT/'runs'/name
        cfg = read(run/'config.json'); manifest = read(run/'launch_manifest.json')
        assert manifest['engine_commit'] == EXPECTED_COMMIT
        comparable = dict(
            graphs_sha256=digest(Path(cfg['graphs_file'])), cases=cfg['cases'],
            repeats=cfg['repeats'], nodes=cfg['nodes'],
            model_revision=manifest['model_revision'],
            idle_gap_cap_s=cfg['idle_gap_cap_s'],
            warmup_cases=cfg['warmup_cases'], clone_lane_policy=cfg['clone_lane_policy'],
            replayer_sha256=digest(run/'code/agentx_replay.py'),
        )
        if reference is None:
            reference = comparable
        assert comparable == reference, (name, 'Unmatched input/protocol/source')
        assert cfg['clone_lane_policy'] == 'pool_counterbalanced_copy_offset_v2'
        assert len(cfg['nodes']) == 2 and cfg['repeats'] == 3
        progress = read(run/'task/progress.json') if (run/'task/progress.json').exists() else {}
        cleanup = read(run/'task/cleanup.json') if (run/'task/cleanup.json').exists() else {}
        posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
        post_pass = bool(posts and read(posts[-1])['passed'])
        files = sorted(run.glob('results/*/*/*/formal.r*.json'))
        finished = len(files) == cfg['expected_points']
        accepted_run = finished and progress.get('stage') == 'complete' and cleanup.get('passed') and cleanup.get('gpu_idle') and post_pass
        arm = cfg['arms'][0]
        qpath = run/'evidence'/arm/'serial_qualification.json'
        wpath = run/'evidence'/arm/'untimed_warmup.json'
        if files:
            assert read(qpath)['passed'] and read(qpath)['exact_match_reference']
            assert read(wpath)['passed']
            for rp in run.glob('nodes/*/*/ready.json'):
                infos = read(rp)['server_info']
                for info in infos if isinstance(infos,list) else [infos]:
                    for s in info['internal_states']:
                        assert s['enable_hierarchical_cache'] and s['hicache_size'] == 32
                        assert s['mem_fraction_static'] == .95
                        if s.get('disaggregation_mode') == 'decode':
                            assert s['disaggregation_decode_extra_slots'] == cfg['decode_extra_slots'] == 8
        run_status.append(dict(run=name, arm=arm, stage=progress.get('stage','prepared'),
            finished_episodes=len(files), planned_episodes=cfg['expected_points'],
            run_accepted=bool(accepted_run), profile_override=cfg['profile_override'], imported_episodes=len(cfg.get('imported_episodes', {}))))
        for p in sorted((run/'cache').glob('*/*.log')):
            for match in re.finditer(r'\[([^\]\n]+)\] KV Cache is allocated\. dtype: ([^,]+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB', p.read_text(errors='replace')):
                label, dtype, tokens, k_gib, v_gib = match.groups()
                capacities.append(dict(run=name,arm=arm,node=p.parent.name,instance=p.stem,
                    rank_label=label[20:],dtype=dtype,tokens=int(tokens),
                    k_gib=float(k_gib),v_gib=float(v_gib),source=str(p)))
        for p in files:
            a, case, policy = p.relative_to(run).parts[1:4]
            d = read(p); rows = d['requests']; repeat = p.stem.split('.')[-1]
            imported = cfg.get('imported_episodes', {}).get(str(p.relative_to(run)))
            if imported:
                assert a == 'col_ep4' and d['clone_lane_policy'] == 'cyclic_copy_offset_v1'
                for artifact in imported['artifacts']:
                    assert digest(run/artifact['relative_path']) == artifact['sha256']
            else:
                assert d['clone_lane_policy'] == cfg['clone_lane_policy']
            assert read(p.with_name('completed.'+repeat+'.json'))['passed']
            intervals = sum(r['output_tokens']-1 for r in rows)
            after_first = sum(r['e2e_ms']-r['ttft_ms'] for r in rows)
            identity = dict(run=name, arm=a, case=case, policy=policy, repeat=repeat, measurement_origin=imported['origin_run'] if imported else str(run))
            roots = {}
            for row in rows:
                roots.setdefault(row['root_id'], []).append(row)
            durations = []
            for rid, calls in roots.items():
                ready = min(x['started']-x['client_dispatch_lag_ms']/1000 for x in calls)
                finished = max(x['finished'] for x in calls)
                assert finished >= ready
                durations.append(finished-ready)
                root_rows.append(dict(**identity,root_id=rid,calls=len(calls),
                    ready_to_last_llm_s=finished-ready,first_ready_client_clock_s=ready,
                    final_response_client_clock_s=finished))
            m = dict(**identity, requests=len(rows),
                ttft_ms=st.mean(r['ttft_ms'] for r in rows),
                request_tpot_ms=st.mean(r['tpot_ms'] for r in rows if r['tpot_ms'] is not None),
                interval_tpot_ms=after_first/intervals,
                e2e_ms=st.mean(r['e2e_ms'] for r in rows),
                dispatch_lag_ms=st.mean(r['client_dispatch_lag_ms'] for r in rows),
                ready_ttft_ms=st.mean(r['ttft_ms']+r['client_dispatch_lag_ms'] for r in rows),
                ready_e2e_ms=st.mean(r['e2e_ms']+r['client_dispatch_lag_ms'] for r in rows),
                root_mean_s=st.mean(durations),
                wall_s=d['wall_s'], idle_compressed_s=d['idle_compressed_s'],
                output_tokens_per_s=d['output_tokens_per_s'],
                cached_tokens=st.mean(r['meta_info'].get('cached_tokens',0) or 0 for r in rows),
                host_hit_requests=sum((r['meta_info'].get('cached_tokens_details') or {}).get('host',0)>0 for r in rows),
                retractions=sum(r['meta_info'].get('num_retractions',0) or 0 for r in rows),
                switched_requests=d['switched_requests'])
            assert abs(m['e2e_ms']-m['ttft_ms']-after_first/len(rows))<1e-5
            trials.append(m)
            sources.append(dict(path=str(p), sha256=digest(p), imported_from=imported['origin_run'] if imported else None))
            for s in d['slo']:
                passed = sum(r['ttft_ms'] + r['client_dispatch_lag_ms'] <= s['ttft_ms'] and
                    (r['tpot_ms'] is None or r['tpot_ms'] <= s['tpot_ms']) for r in rows)
                slo_rows.append(dict(**identity,ttft_limit_ms=s['ttft_ms'],tpot_limit_ms=s['tpot_ms'],
                    http_goodput_rps=s['goodput_rps'],ready_goodput_rps=passed/d['wall_s'],
                    ready_fraction=passed/len(rows)))
            fp = p.with_name('forwards.'+repeat+'.json')
            forward = read(fp)
            assert len(forward['delta']) == 8
            for rank, v in forward['delta'].items():
                counts = v['counts']; hist = {int(k.split(':b')[1]):n for k,n in counts.items() if k.startswith('DECODE:b') and n}
                assert all(n>=0 for n in counts.values())
                n = sum(hist.values()); token_rows = sum(b*v for b,v in hist.items())
                rank_rows.append(dict(**identity,rank=rank,pp_rank=v['shape']['pp_rank'],
                    decode_forwards=n,decode_rows=token_rows,mean_decode_batch=token_rows/n if n else 0,
                    decode_batch1_fraction=hist.get(1,0)/n if n else 0,
                    idle_forwards=counts.get('IDLE:b0',0),decode_histogram=json.dumps(hist,sort_keys=True)))
    def means(rows, group_keys, value_keys):
        groups = {}
        for r in rows:
            groups.setdefault(tuple(r[k] for k in group_keys),[]).append(r)
        return [dict(**dict(zip(group_keys,k)),repeats=len(g),complete_point=len(g)==3,
                **{v:st.mean(r[v] for r in g) for v in value_keys}) for k,g in groups.items()]
    values = ('ttft_ms','request_tpot_ms','interval_tpot_ms','e2e_ms','dispatch_lag_ms',
              'ready_ttft_ms','ready_e2e_ms','root_mean_s','wall_s','idle_compressed_s','output_tokens_per_s','cached_tokens',
              'host_hit_requests','retractions','switched_requests')
    summary = means(trials,('run','arm','case','policy'),values)
    slo = means(slo_rows,('run','arm','case','policy','ttft_limit_ms','tpot_limit_ms'),
                ('http_goodput_rps','ready_goodput_rps','ready_fraction'))
    for fname, rows in [('trials.csv',trials),('means.csv',summary),('slo_means.csv',slo),('rank_batches.csv',rank_rows),('cache_capacities.csv',capacities),('root_sessions.csv',root_rows)]:
        csv_write(args.out/fname,rows)
    audit = dict(expected_commit=EXPECTED_COMMIT, comparable=reference, runs=run_status,
        all_runs_accepted=all(s['run_accepted'] for s in run_status),
        sources=sources,analyzer_sha256=digest(Path(__file__)),
        excluded_runs=[dict(run='eight_gpu_pd_ep4_durable_r3', formal_episodes=0,
            reason='Decode default transfer slots expanded GDN80 to144 outside KV budget; startup OOM, cleanup passed.')],
        caveats=['Three repeat arithmetic means; partial points explicitly labeled.',
        'Single GPU uses 40 GDN slots, other layouts 80; native KV capacities differ.',
        'PD Decode uses 8 native transfer preallocation slots per DP worker; active-request cap and GDN80 preserved.',
        'HTTP and request-ready boundaries both retained; finite replay goodput includes tool waits.',
        'Episode wall/root completion and goodput refer to compressed replay; skipped source-idle time is reported separately. Custom branch timing differs from the official harness runtime.',
        'Per-rank local scheduled batch counts are not synchronized global batches. PP stages duplicate token work; never sum into delivered throughput.',
        'Reported cache counts are not transfer bytes. Captured CUDA graph replays need not execute Python dispatch hooks.'])
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    text = ['# AgentX：同预算 E5 正式对照','',
        '两个节点共8张H100；同一E5源码、输入家族、合成token命名空间、预热、HiCache与副本分配协议。单卡40个GDN槽位，EP4／PP2为80；均保留实际可用GPU KV容量。','',
        '**未完成全部三轮的行仅作进度，不能当作最终均值；整组尚需Guard后检查与清理通过。**','',
        'EP4的C8三轮及C128第一轮来自原E5运行，逐文件哈希验证后导入，只补缺失两轮。修复涉及计时外计数通知与结果保存；2/4/8缓存位置的请求路由保持原样。各行原始运行见trials.csv的measurement_origin。','',
        'P/D的Decode端显式使用每DP worker 8个接收预留槽位，保持活跃请求上限和GDN80池。默认32槽位曾导致池增长到144、启动OOM，该失败运行无正式样本。[容量修正与验证](../DECODE_SLOT_BUDGET.md)。','',
        '| 布局 | 负载 | 完成轮数 | TTFT ms | 请求平均TPOT ms/token | 间隔加权TPOT ms/token | E2E ms | 客户端排队 ms | 含排队E2E ms | 整轮 s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        text.append(f"| {s['arm']} | {s['case']} | {s['repeats']}/3 | {s['ttft_ms']:.2f} | {s['request_tpot_ms']:.3f} | {s['interval_tpot_ms']:.3f} | {s['e2e_ms']:.2f} | {s['dispatch_lag_ms']:.2f} | {s['ready_e2e_ms']:.2f} | {s['wall_s']:.3f} |")
    text += ['', '| 布局 | 当前状态 | 完成轮数 | 完整后检查及清理通过 |','|---|---|---:|---|']
    for s in run_status:
        text.append(f"| {s['arm']} | {s['stage']} | {s['finished_episodes']}/{s['planned_episodes']} | {s['run_accepted']} |")
    text += ['', '## 根会话平均完成时间', '',
        '从该根会话首个请求就绪，到它及其全部子代理的最后一个已捕获LLM响应结束；包含工具等待（保留原有全局空闲压缩规则）与客户端排队。每个根会话权重相同。这是补充的派生指标，不是官方AgentX分数，也不包含最后一个捕获响应之后未记录的工具工作。', '',
        '| 布局 | 负载 | 轮数 | 根会话平均完成 s |','|---|---|---:|---:|']
    for s in summary:
        text.append(f"| {s['arm']} | {s['case']} | {s['repeats']}/3 | {s['root_mean_s']:.3f} |")
    text += ['', '## 原始空闲压缩量', '',
        '整轮／根会话完成时间及达标请求/s均基于压缩回放。每轮跳过的原始空闲如下；'
        '当前分支触发与官方运行时也有差别，见[时间协议核查](../TIMING_AUDIT.md)。'
        '把墙钟与跳过量相加只是虚拟时钟推进量，不能作为未压缩重跑的性能。', '',
        '| 布局 | 负载 | 轮数 | 回放整轮均值 s | 跳过的原始空闲均值 s |',
        '|---|---|---:|---:|---:|']
    for s in summary:
        text.append(f"| {s['arm']} | {s['case']} | {s['repeats']}/3 | {s['wall_s']:.3f} | {s['idle_compressed_s']:.3f} |")
    text += ['', '[全部均值](means.csv) · [每轮原值](trials.csv) · [各根会话](root_sessions.csv) · [根会话时间分解](ROOT_EXPOSURE.md) · [历史前缀位置](PREFIX_LOCALITY.md) · [全部固定延迟目标](slo_means.csv) · [每rank实际Decode批次](rank_batches.csv) · [每GPU实际KV容量](cache_capacities.csv) · [来源与可比性核查](audit.json)', '',
        'TPOT为客户端观察，不是GPU核函数时间。客户端排队从DAG请求就绪算起，含客户端准备／线程池等待；在请求就绪前的工具等待仍保留在整轮耗时中。通过率针对请求平均TPOT，并不要求每个token间隔均达标。HiCache命中不等于实测RDMA传输字节。', '',
        '实际批次按每个GPU rank分别保存，PP不同阶段不能相加成有效输出吞吐；Graph重放不会逐次执行Python dispatch hook，所以此类hook计数不能用作LL/Normal使用比例。']
    (args.out/'REPORT.md').write_text('\n'.join(text)+'\n')
    from analyze_root_exposure import main as analyze_root_exposure
    analyze_root_exposure(args.out)
    from analyze_prefix_locality import main as analyze_prefix_locality
    analyze_prefix_locality(args.out)
    print(json.dumps(dict(complete_points=sum(s['complete_point'] for s in summary),episodes=len(trials),all_runs_accepted=audit['all_runs_accepted'])))


if __name__=='__main__':
    main()
