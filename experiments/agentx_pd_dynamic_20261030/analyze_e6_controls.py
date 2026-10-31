"""Summarize one source cohort; defaults preserve the original E6 table."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import time

ROOT = Path(__file__).resolve().parent
COMMIT = 'fc014e3553c4d234f38f3088a3158c0333e64830'
RUNS = ('eight_gpu_pd_ep4_checkpoint_e6_r2',
        'eight_gpu_pp2_pd_controls_checkpoint_e6_r1',
        'eight_gpu_ep2_split26_checkpoint_e6_r1',
        'eight_gpu_ep4_native_queue32_checkpoint_e6_r1')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    if rows:
        with path.open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    elif path.exists():
        path.unlink()


def metrics(data):
    rows = data['requests']
    roots = defaultdict(list)
    for r in rows:
        roots[r['root_id']].append(r)
        if r['tpot_ms'] is not None:
            assert abs(r['e2e_ms']-r['ttft_ms']-(r['output_tokens']-1)*r['tpot_ms']) < 1e-5
    root_time = [max(x['finished'] for x in calls) -
                 min(x['started']-x['client_dispatch_lag_ms']/1000 for x in calls)
                 for calls in roots.values()]
    intervals = sum(r['output_tokens']-1 for r in rows)
    value = dict(requests=len(rows), roots=len(roots),
        ttft_ms=st.mean(r['ttft_ms'] for r in rows),
        request_tpot_ms=st.mean(r['tpot_ms'] for r in rows if r['tpot_ms'] is not None),
        interval_tpot_ms=sum(r['e2e_ms']-r['ttft_ms'] for r in rows)/intervals,
        e2e_ms=st.mean(r['e2e_ms'] for r in rows),
        dispatch_lag_ms=st.mean(r['client_dispatch_lag_ms'] for r in rows),
        ready_ttft_ms=st.mean(r['ttft_ms']+r['client_dispatch_lag_ms'] for r in rows),
        ready_e2e_ms=st.mean(r['e2e_ms']+r['client_dispatch_lag_ms'] for r in rows),
        root_mean_s=st.mean(root_time), wall_s=data['wall_s'],
        idle_compressed_s=data['idle_compressed_s'],
        reported_client_cached_tokens=st.mean(r['meta_info'].get('cached_tokens') or 0 for r in rows),
        reported_prefill_cached_tokens=st.mean(r.get('prefill_meta', {}).get('cached_tokens') or 0 for r in rows),
        host_hit_requests=sum((r['meta_info'].get('cached_tokens_details') or {}).get('host', 0)>0 for r in rows),
        retractions=sum(r['meta_info'].get('num_retractions') or 0 for r in rows),
        switched_requests=data['switched_requests'])
    for external, internal in [('ttft_ms','ttft_ms'), ('tpot_ms','request_tpot_ms'), ('e2e_ms','e2e_ms')]:
        assert abs(data['means'][external]-value[internal]) < 1e-5
    return value


def means(rows, keys, values):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [dict(**dict(zip(keys, key)), repeats=len(group), complete_point=len(group)==3,
                 **{v: st.mean(r[v] for r in group) for v in values})
            for key, group in groups.items()]


def cache_capacity_rows(run, measured_arms):
    """Count independent sequence capacity once per PP replica / Attention DP."""
    rows = []
    for arm in measured_arms:
        arm_rows, attached_ranks = [], 0
        for p in sorted(run.glob(f'nodes/*/{arm}/ready.json')):
            ready = read(p)
            assert ready['passed']
            infos = ready['server_info']
            for index, info in enumerate(infos if isinstance(infos, list) else [infos]):
                states = info['internal_states']
                roles = {s['disaggregation_mode'] for s in states}
                assert len(roles) == 1
                for s in states:
                    assert s['enable_hierarchical_cache'] and s['hicache_size'] == 32
                    assert s['mem_fraction_static'] == .95
                role = roles.pop()
                owners = info['dp_size'] if info['enable_dp_attention'] else 1
                gpus = info['tp_size'] * info['pp_size']
                arm_rows.append(dict(run=run.name, arm=arm, node=p.parents[1].name,
                    instance=index, role='colocated' if role == 'null' else role,
                    gpus=gpus, independent_cache_owners=owners,
                    kv_tokens_per_owner=info['max_total_num_tokens'],
                    independent_kv_tokens=owners * info['max_total_num_tokens'],
                    hicache_budget_decimal_gb=32 * gpus, source=str(p)))
        for log in sorted((run/'cache').glob(f'*/{arm}.r*.log')):
            # Server-info only exposes PP rank zero. Native startup records
            # confirm that HiCache also attached on the downstream stages.
            attached_ranks += len(re.findall(
                r'^\[[^\]\n]+\] Tree cache initialized: [^\n]*hicache_attached=True',
                log.read_text(errors='replace'), re.M))
        assert sum(r['gpus'] for r in arm_rows) == 8, (arm, 'missing ready instances')
        assert attached_ranks == 8, (arm, 'missing native HiCache rank attachment', attached_ranks)
        rows.extend(arm_rows)
    return rows


def main(*, expected_commit=COMMIT, run_names=RUNS, cohort='E6'):
    out = ROOT/(cohort.lower()+'_analysis'); out.mkdir(exist_ok=True)
    common = None
    known_cases = {}
    trials, slo_trials, sources, status, capacities = [], [], [], [], []
    for name in run_names:
        run = ROOT/'runs'/name
        cfg, manifest = read(run/'config.json'), read(run/'launch_manifest.json')
        assert manifest['engine_commit'] == expected_commit and cfg['repeats'] == 3
        assert all(sum(counts.values()) == 8 for counts in cfg['role_gpu_counts'].values())
        waiting_timeouts = {int(c['launch_env']['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'])
                            for c in cfg['node_configs'].values()}
        assert len(waiting_timeouts) == 1
        episode_timeout = cfg.get('episode_timeout_s', 2400)
        replay_source = (run/'code/agentx_replay.py').read_text()
        if 'episode_timeout_s' in cfg:
            assert 'deadline = start + episode_timeout_s' in replay_source
        else:
            assert 'deadline = start + 2400' in replay_source
        invariant = dict(model_revision=manifest['model_revision'], nodes=cfg['nodes'],
            graphs_sha256=sha(Path(cfg['graphs_file'])), replayer_sha256=sha(run/'code/agentx_replay.py'),
            warmup_cases=cfg['warmup_cases'], idle_gap_cap_s=cfg['idle_gap_cap_s'],
            clone_lane_policy=cfg['clone_lane_policy'], waiting_timeout_s=waiting_timeouts.pop(),
            episode_timeout_s=episode_timeout,
            branch_timing_policy=cfg.get('branch_timing_policy', 'legacy_post_branch_v3'),
            branch_helper_sha256=sha(run/'code/agentx_branch_timing.py')
                if 'branch_timing_policy' in cfg else None)
        if 'branch_timing_policy' in cfg:
            assert manifest['branch_timing_policy'] == cfg['branch_timing_policy']
            assert manifest['branch_timing']['helper']['sha256'] == invariant['branch_helper_sha256']
        if common is None:
            common = invariant
        assert common == invariant
        assert len(cfg['nodes']) == 2
        for case in cfg['cases']:
            if case['id'] in known_cases:
                assert case == known_cases[case['id']]
            known_cases[case['id']] = case
        progress = read(run/'task/progress.json')
        clean = read(run/'task/cleanup.json') if (run/'task/cleanup.json').exists() else {}
        posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
        durable_files = sorted(run.glob('results/*/*/*/formal.r*.json'))
        files = [p for p in durable_files
                 if p.with_name('completed.'+p.stem.split('.')[-1]+'.json').exists()]
        pending_formals = [str(p) for p in durable_files if p not in files]
        accepted = bool(len(files) == cfg['expected_points'] and progress['stage'] == 'complete'
            and clean.get('passed') and clean.get('gpu_idle') and posts and read(posts[-1])['passed'])
        if accepted:
            for frozen in read(run/'dashboard_spec.json')['frozen_inputs']:
                assert sha(Path(frozen['path'])) == frozen['sha256'], frozen['path']
        if files:
            for p in run.glob('nodes/*/*/ready.json'):
                infos = read(p)['server_info']
                for info in infos if isinstance(infos, list) else [infos]:
                    for s in info['internal_states']:
                        assert s['enable_hierarchical_cache'] and s['hicache_size'] == 32
                        assert s['mem_fraction_static'] == .95
            capacities.extend(cache_capacity_rows(run, sorted({p.relative_to(run).parts[1] for p in files})))
        markers, restore_markers = [], []
        for log in sorted((run/'cache').glob('*/*.log')):
            notices = [line for line in log.read_text(errors='replace').splitlines()
                       if 'MAMBA_CACHE_CHECKPOINT_DEFERRED:' in line]
            if notices:
                markers.append(dict(path=str(log), notices=notices))
            restore_notices = [line for line in log.read_text(errors='replace').splitlines()
                              if 'HICACHE_LOAD_BACK_' in line or 'HiCache promised prefix lost' in line]
            if restore_notices:
                restore_markers.append(dict(path=str(log), notices=restore_notices))
        interruption = read(run/'interruption.json') if (run/'interruption.json').exists() else None
        status.append(dict(run=name, arms=cfg['arms'],
            stage=interruption['stage'] if interruption else progress['stage'],
            interruption=interruption,
            episodes=len(files), planned=cfg['expected_points'], accepted=accepted,
            durable_formals_awaiting_completion_marker=pending_formals,
            partial_diagnostics_excluded=[str(p) for p in sorted(run.glob('results/*/*/*/partial.r*.json'))],
            profile_override=cfg['profile_override'], pp_microbatch=cfg.get('pp_microbatch_cap'),
            decode_extra_slots=cfg['decode_extra_slots'],
            checkpoint_pressure_notices=markers, host_restore_notices=restore_markers))
        graph_counts = {g['id']:sum(len(c['turns']) for c in g['conversations'])
                        for g in read(Path(cfg['graphs_file']))['graphs']}
        for p in files:
            arm, case, policy = p.relative_to(run).parts[1:4]
            repeat = p.stem.split('.')[-1]
            assert read(run/'evidence'/arm/'serial_qualification.json')['exact_match_reference'] is True
            assert read(run/'evidence'/arm/'untimed_warmup.json')['passed']
            assert read(p.with_name('completed.'+repeat+'.json'))['passed']
            forwards = read(p.with_name('forwards.'+repeat+'.json'))
            assert len(forwards['delta']) == 8
            data = read(p); rows = data['requests']; c = known_cases[case]
            assert data.get('episode_timeout_s', 2400) == episode_timeout
            if 'branch_timing_policy' in cfg:
                assert data['branch_timing_policy'] == cfg['branch_timing_policy']
                assert all('credit_issued_virtual_s' in r for r in rows)
            expected = sum(graph_counts[g] for g in c['graph_ids'])*c['copies']
            assert data['completed_requests'] == len(rows) == expected
            assert len({(r['root_id'],r['conversation_id'],r['turn']) for r in rows}) == expected
            assert data['clone_lane_policy'] == cfg['clone_lane_policy']
            ident = dict(run=name, arm=arm, case=case, policy=policy, repeat=repeat)
            trials.append(dict(**ident, **metrics(data)))
            for target in data['slo']:
                passed = sum(r['ttft_ms']+r['client_dispatch_lag_ms'] <= target['ttft_ms'] and
                    (r['tpot_ms'] is None or r['tpot_ms'] <= target['tpot_ms']) for r in rows)
                slo_trials.append(dict(**ident, ttft_limit_ms=target['ttft_ms'], tpot_limit_ms=target['tpot_ms'],
                    ready_goodput_rps=passed/data['wall_s'], ready_fraction=passed/len(rows)))
            sources.append(dict(path=str(p), sha256=sha(p)))
    keys = ('run','arm','case','policy')
    values = [k for k in trials[0] if k not in keys+('repeat',)] if trials else []
    summary = means(trials, keys, values)
    slo = means(slo_trials, keys+('ttft_limit_ms','tpot_limit_ms'), ('ready_goodput_rps','ready_fraction'))
    accepted_names = {s['run'] for s in status if s['accepted']}
    for row in summary+slo:
        row['run_accepted'] = row['run'] in accepted_names
    for filename, data in [('trials.csv',trials),('means.csv',summary),('slo_trials.csv',slo_trials),('slo_means.csv',slo),('cache_capacities.csv',capacities)]:
        csv_write(out/filename, data)
    (out/'audit.json').write_text(json.dumps(dict(updated_at=time.time(), engine_commit=expected_commit,
        cohort=cohort,
        protocol=common, cases=known_cases, runs=status, sources=sources,
        analyzer_sha256=sha(Path(__file__)), all_runs_accepted=all(s['accepted'] for s in status)),indent=2)+'\n')
    replay_label = ('V4分支规则修正的派生回放' if common['branch_timing_policy'] == 'official_branch_issue_and_return_v1'
                    else 'V3派生回放')
    lines = [f'# AgentX：{cohort} 修复版与同版本控制', '',
        f'两节点共8张H100，HiCache开启、GPU KV原生自动容量；{replay_label}，完整SLO网格。'
        f'本表只纳入{cohort}，交接等待上限{common["waiting_timeout_s"]}秒、整轮上限{common["episode_timeout_s"]}秒；其他源码或超时协议保留在原表。'
        '完整点取三轮算术平均；少于三轮时只汇总已有完整轮，整组未验收的行仅作进度。已保存但缺完成标记的原始回放列在audit中，不计入均值。', '',
        '| 配置 | 负载 | 轮数 | 整组验收 | TTFT ms | 请求TPOT ms/token | 加权TPOT ms/token | 请求E2E ms | 含客户端排队E2E ms | 根会话 s | 压缩整轮 s |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in summary:
        config = next(s for s in status if s['run'] == m['run'])
        label = m['arm'] + f" (GDN{config['profile_override']['mamba_equiv_per_full_gpu']})"
        if m['arm'].startswith('pd_'):
            label += f" / receive{config['decode_extra_slots']}"
        lines.append(f"| {label} | {m['case']} | {m['repeats']}/3 | {m['run_accepted']} | {m['ttft_ms']:.2f} | {m['request_tpot_ms']:.3f} | {m['interval_tpot_ms']:.3f} | {m['e2e_ms']:.2f} | {m['ready_e2e_ms']:.2f} | {m['root_mean_s']:.3f} | {m['wall_s']:.3f} |")
    lines += ['', '| 运行 | 阶段 | 完成/计划 | 整组验收 | 有压力延期日志的进程日志数 |', '|---|---|---:|---|---:|']
    for s in status:
        lines.append(f"| {s['run']} | {s['stage']} | {s['episodes']}/{s['planned']} | {s['accepted']} | {len(s['checkpoint_pressure_notices'])} |")
    if capacities:
        capacity_groups = defaultdict(list)
        for row in capacities:
            capacity_groups[(row['run'], row['arm'], row['role'])].append(row)
        lines += ['', '| 运行 / 配置 | 角色 | GPU | 独立缓存位置 | 完整层KV容量 token | HiCache配置预算 GB |',
                  '|---|---|---:|---:|---:|---:|']
        for (run, arm, role), rows in capacity_groups.items():
            lines.append(f"| {run} / {arm} | {role} | {sum(r['gpus'] for r in rows)} | "
                f"{sum(r['independent_cache_owners'] for r in rows)} | {sum(r['independent_kv_tokens'] for r in rows):,} | "
                f"{sum(r['hicache_budget_decimal_gb'] for r in rows)} |")
        lines += ['', 'KV容量按每个实例报告的可用token上限乘Attention DP数；PP各阶段只计一次。'
                  'P和D分别列出，不能相加当成Decode可用容量。HiCache一栏是CPU配置预算（十进制GB），不是实际驻留字节。'
                  '所有已有正式样本的配置均核对了8个rank的原生HiCache挂载日志。']
    lines += ['', 'GDN表示每GPU的完整模型等效状态预算；PP2每阶段的实际槽位数翻倍。receive表示每Attention-DP worker的Decode接收预留。'
        '160/receive32是独立容量配置控制，不与80/receive8合并平均。', '',
        '表中的进程日志数只表示该日志含压力延期记录，不是延期频次或延迟。'
        'P/D客户端缓存字段可能合并P侧计数，不等于D侧独立复用量或传输字节。'
        '压缩回放的root、wall及达标请求/s不能当作原始生产时间或官方AgentX分数。', '']
    if trials:
        lines += ['[每轮值](trials.csv) · [全部均值](means.csv) · [完整SLO网格](slo_means.csv) · [逐实例缓存容量](cache_capacities.csv)']
    else:
        lines += [f'尚无完整{cohort} GPU样本，CSV在第一份正式样本完成后生成。']
    lines += ['', '[验收、压力日志和来源](audit.json) · [E5原始对照](../matrix_analysis/REPORT.md)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cohort=cohort, episodes=len(trials), accepted_runs=len(accepted_names), planned_runs=len(run_names))))


if __name__ == '__main__':
    main()
