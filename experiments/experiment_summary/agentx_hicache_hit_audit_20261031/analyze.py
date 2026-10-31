"""CPU-only audit of the frozen E5/V3 colocated AgentX cache evidence.

Raw request metadata is authoritative. C8 is an empirical low-pressure
reference, not a guarantee that every prefix must be resident at C128.
"""
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RUNS = WORK / 'experiments/agentx_pd_dynamic_20261030/runs'
ENGINE = WORK / '.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031'
ARMS = ('col_ep4', 'col_pp2', 'col_single')
SLOTS = dict(col_ep4=80, col_pp2=160, col_single=40)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readcsv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def writecsv(name, rows):
    assert rows, name
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family(q):
    return q['root_id'].split(':')[-1]


def conversation(q):
    return q['conversation_id'].split(':', 2)[-1]


def sourcekey(q):
    return family(q), conversation(q), q['turn']


def cache(q):
    m = q['meta_info']
    d = m.get('cached_tokens_details') or {}
    total, gpu, host = m['cached_tokens'], d.get('device', 0), d.get('host', 0)
    assert total == gpu + host
    assert 0 <= total <= q['input_tokens']
    return total, gpu, host


def summarize(rows):
    ins = sum(r['input_tokens'] for r in rows)
    return dict(requests=len(rows), input_tokens=ins,
        gpu_hit_pct=100 * sum(r['gpu'] for r in rows) / ins,
        host_hit_pct=100 * sum(r['host'] for r in rows) / ins,
        total_hit_pct=100 * sum(r['cached'] for r in rows) / ins,
        reference_c8_hit_pct=100 * sum(r['c8_cached'] for r in rows) / ins,
        mean_new_tokens=statistics.mean(r['input_tokens'] - r['cached'] for r in rows),
        mean_reference_c8_new_tokens=statistics.mean(r['input_tokens'] - r['c8_cached'] for r in rows),
        lost_vs_c8_tokens=sum(max(0, r['c8_cached'] - r['cached']) for r in rows),
        gained_vs_c8_tokens=sum(max(0, r['cached'] - r['c8_cached']) for r in rows),
        additional_zero_hits=sum(r['c8_cached'] > 0 and r['cached'] == 0 for r in rows),
        worse_than_c8_requests=sum(r['c8_cached'] > r['cached'] for r in rows),
        zero_hits=sum(r['cached'] == 0 for r in rows),
        any_host_hit_requests=sum(r['host'] > 0 for r in rows),
        retractions=sum(r['retractions'] for r in rows))


def main():
    accepted = WORK / 'experiments/experiment_summary/agentx_data_audit_20261031/accepted_episodes.csv'
    episodes = [r for r in readcsv(accepted) if r['cohort'] == 'E5/V3']
    assert len(episodes) == 18
    verified, raw = [], {}
    for e in episodes:
        path = Path(e['source'])
        assert sha(path) == e['sha256'], path
        x = json.loads(path.read_text())
        assert len(x['requests']) == int(e['completed_requests'])
        raw[e['arm'], e['case'], e['repeat']] = x['requests']
        verified.append(dict(path=str(path), sha256=e['sha256']))
    baseline = {}
    for (arm, case, rep), qs in raw.items():
        if case != 'agentx_c8':
            continue
        for q in qs:
            key = sourcekey(q)
            val = q['input_tokens'], cache(q)[0]
            assert key not in baseline or baseline[key] == val, key
            baseline[key] = val
    assert len(baseline) == 171

    rows, paired, trials = [], defaultdict(dict), []
    for (arm, case, rep), qs in sorted(raw.items()):
        ep_rows = []
        owners = {}
        for q in qs:
            sk = sourcekey(q)
            inp, ref = baseline[sk]
            assert q['input_tokens'] == inp
            total, gpu, host = cache(q)
            clone = int(re.search(r':copy(\d+):', q['root_id']).group(1))
            owner = q['prefill_instance'] + ':' + str(q.get('routed_dp_rank', 0))
            old = owners.setdefault(q['conversation_id'], owner)
            assert owner == old
            r = dict(arm=arm, case=case, repeat=rep, clone=clone,
                source_family=sk[0], source_conversation=sk[1], turn=sk[2],
                source_kind=q.get('source_kind'), is_root=q['conversation_id'] == q['root_id'],
                owner=owner, input_tokens=inp, c8_cached=ref, cached=total, gpu=gpu, host=host,
                retractions=q['meta_info'].get('num_retractions', 0),
                started=q['started'], finished=q['finished'],
                forward_entry_ts=q['meta_info'].get('forward_entry_time'))
            rows.append(r)
            ep_rows.append(r)
            if case == 'agentx_c128':
                key = rep, clone, *sk
                assert arm not in paired[key]
                paired[key][arm] = r
        trials.append(dict(arm=arm, case=case, repeat=rep, **summarize(ep_rows)))
    assert len(paired) == 8208
    assert all(set(v) == set(ARMS) for v in paired.values())
    writecsv('requests.csv', rows)
    writecsv('trials.csv', trials)
    summary = [dict(arm=arm, case=case,
        **summarize([r for r in rows if r['arm'] == arm and r['case'] == case]))
        for arm in ARMS for case in ('agentx_c8', 'agentx_c128')]
    writecsv('full_metrics.csv', summary)
    parts = []
    for arm in ARMS:
        for kind in sorted({r['source_kind'] for r in rows}):
            selected = [r for r in rows if r['arm'] == arm and r['case'] == 'agentx_c128' and r['source_kind'] == kind]
            if selected:
                parts.append(dict(arm=arm, source_kind=kind, **summarize(selected)))
    writecsv('source_kind_metrics.csv', parts)
    owner_rows = []
    for arm, case, rep in sorted(raw):
        subset = [r for r in rows if (r['arm'], r['case'], r['repeat']) == (arm, case, rep)]
        for owner in sorted({r['owner'] for r in subset}):
            selected = [r for r in subset if r['owner'] == owner]
            root = [r for r in selected if r['is_root']]
            child = [r for r in selected if not r['is_root']]
            owner_rows.append(dict(arm=arm, case=case, repeat=rep, owner=owner,
                root_requests=len(root), child_requests=len(child),
                root_input=sum(r['input_tokens'] for r in root),
                child_input=sum(r['input_tokens'] for r in child),
                root_new=sum(r['input_tokens'] - r['cached'] for r in root),
                child_new=sum(r['input_tokens'] - r['cached'] for r in child),
                signed_cache_loss_vs_c8=sum(r['c8_cached'] - r['cached'] for r in selected),
                **summarize(selected)))
    writecsv('owner_metrics.csv', owner_rows)
    pair_rows = []
    for key, val in sorted(paired.items()):
        r = val['col_ep4']
        p = {k:r[k] for k in ('repeat','clone','source_family','source_conversation','turn','source_kind','is_root','input_tokens','c8_cached')}
        for arm in ARMS:
            for k in ('cached','gpu','host','owner','started','finished'):
                p[arm + '_' + k] = val[arm][k]
        p['pp2_minus_ep4_cached'] = val['col_pp2']['cached'] - val['col_ep4']['cached']
        pair_rows.append(p)
    writecsv('paired_requests.csv', pair_rows)
    pair_parts = []
    for label, subset in [('all',pair_rows), ('root',[r for r in pair_rows if r['is_root']]), ('child',[r for r in pair_rows if not r['is_root']])]:
        pair_parts.append(dict(scope=label, requests=len(subset),
            pp2_more_cached_requests=sum(r['pp2_minus_ep4_cached'] > 0 for r in subset),
            ep4_more_cached_requests=sum(r['pp2_minus_ep4_cached'] < 0 for r in subset),
            same_cached_requests=sum(r['pp2_minus_ep4_cached'] == 0 for r in subset),
            pp2_more_cached_tokens=sum(max(0, r['pp2_minus_ep4_cached']) for r in subset),
            ep4_more_cached_tokens=sum(max(0, -r['pp2_minus_ep4_cached']) for r in subset),
            signed_pp2_more_cached_tokens=sum(r['pp2_minus_ep4_cached'] for r in subset)))
    writecsv('pair_metrics.csv', pair_parts)
    examples = sorted([r for r in pair_rows if r['col_ep4_cached'] == 0 and r['c8_cached'] == r['col_pp2_cached'] and r['c8_cached'] > 32000], key=lambda r:-r['pp2_minus_ep4_cached'])[:12]
    if examples:
        writecsv('examples.csv', examples)

    # Host pool bytes are split in proportion to GPU pool bytes. Logs expose
    # exact host KV tokens, and rounded decimal-GB GDN bytes. Infer the only
    # integer slot count compatible with those rounded bytes and model shapes.
    model_path = Path('/data/gpfs/projects/punim2715/huggingface/Qwen3.6-35B-A3B/config.json')
    model = json.loads(model_path.read_text())['text_config']
    full_layers = model['layer_types'].count('full_attention')
    linear_layers = model['layer_types'].count('linear_attention')
    conv_dim = (2 * model['linear_num_key_heads'] * model['linear_key_head_dim'] + model['linear_num_value_heads'] * model['linear_value_head_dim'])
    state_per_layer = conv_dim * (model['linear_conv_kernel_dim'] - 1) * 2 + model['linear_num_value_heads'] * model['linear_key_head_dim'] * model['linear_value_head_dim'] * 4
    resources, log_sources = [], []
    for arm in ARMS:
        run_name = next(e['run'] for e in episodes if e['arm'] == arm)
        logs = sorted((RUNS / run_name / 'cache').glob('*/*.log'))
        cpu_tokens, cpu_states, kv_gbs, state_gbs, policies = set(), set(), set(), set(), set()
        for path in logs:
            text = path.read_text()
            kv = re.findall(r'Allocating kv hierarchical KV host pool: (\d+) tokens, ([\d.]+) GB',text)
            st = re.findall(r'Allocating ([\d.]+) GB host memory for hierarchical Mamba cache',text)
            assert kv and st, path
            cpu_tokens.update(int(t) for t,_ in kv)
            kv_gbs.update(float(n) for _,n in kv)
            state_gbs.update(float(n) for n in st)
            state_bytes = state_per_layer * (linear_layers // 2 if arm == 'col_pp2' else linear_layers)
            for n in st:
                candidates = [s for s in range(1,1000) if abs(s * state_bytes / 1e9 - float(n)) < .005000001]
                assert len(candidates) == 1, (arm,n,candidates)
                cpu_states.add(candidates[0])
            args = next(l for l in text.splitlines() if 'server_args' in l)
            for k in ('hicache_size','hicache_ratio','hicache_write_policy','hicache_storage_backend','mamba_max_states_per_path','enable_session_radix_cache'):
                m = re.search(r'(?:^|[ ,(])'+k+r'=([^,)]*)',args)
                assert m, k
                policies.add((k,m.group(1)))
            log_sources.append(dict(path=str(path),sha256=sha(path)))
        assert len(cpu_states) == 1, (arm, cpu_states)
        assert set(v for k,v in policies if k == 'hicache_size') == {'32'}
        assert set(v for k,v in policies if k == 'hicache_write_policy') == {"'write_through'"}
        gpu = [r for r in readcsv(RUNS.parent / 'matrix_analysis/cache_capacities.csv') if r['arm'] == arm]
        pools = defaultdict(list)
        for r in gpu:
            key = r['node'],r['instance']
            if arm == 'col_ep4':
                key += (r['rank_label'].split()[0],)
            pools[key].append(int(r['tokens']))
        independent = len(pools)
        resources.append(dict(arm=arm, gpus=8, independent_cache_owners=independent,
            gpu_kv_tokens_per_owner_min=min(min(v) for v in pools.values()),
            gpu_kv_tokens_per_owner_max=max(min(v) for v in pools.values()),
            cpu_kv_tokens_per_owner_min=min(cpu_tokens),
            cpu_kv_tokens_per_owner_max=max(cpu_tokens),
            gpu_state_slots_per_owner=SLOTS[arm],
            cpu_state_slots_per_owner=next(iter(cpu_states)),
            cpu_kv_gb_per_rank_min=min(kv_gbs),
            cpu_kv_gb_per_rank_max=max(kv_gbs),
            cpu_state_gb_per_rank=next(iter(state_gbs)),
            state_bytes_per_slot=state_bytes,
            c128_roots_per_owner=128//independent,
            gpu_kv_tokens_total_logical=sum(min(v) for v in pools.values()),
            cpu_kv_tokens_total_logical_min=min(cpu_tokens)*independent,
            cpu_kv_tokens_total_logical_max=max(cpu_tokens)*independent,
            scope='one state slot stores stage-local layers; stages counted once' if arm == 'col_pp2' else 'one state slot stores full-model linear layers'))
    writecsv('cache_resources.csv', resources)

    source_files = [model_path,
        ENGINE / 'python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py',
        ENGINE / 'python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py',
        ENGINE / 'python/sglang/srt/mem_cache/unified_radix_cache.py',
        ENGINE / 'python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
        ENGINE / 'python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py',
        ENGINE / 'python/sglang/srt/mem_cache/pool_host/mamba.py',
        ENGINE / 'python/sglang/srt/mem_cache/pool_host/base.py',
        ENGINE / 'python/sglang/srt/managers/scheduler_components/pool_stats_observer.py']
    # Describe concentration without claiming routing changes causally fix it.
    concentration = []
    for arm in ARMS:
        for rep in ('r1','r2','r3'):
            selected = [r for r in owner_rows if (r['arm'],r['case'],r['repeat']) == (arm,'agentx_c128',rep)]
            hot = sorted(selected,key=lambda r:-r['child_requests'])[:2]
            total_loss = sum(r['signed_cache_loss_vs_c8'] for r in selected)
            concentration.append(dict(arm=arm,repeat=rep,selected_by='two owners with most child requests',
                owners=';'.join(r['owner'] for r in hot),
                owner_fraction=2/len(selected),
                child_request_fraction=sum(r['child_requests'] for r in hot)/sum(r['child_requests'] for r in selected),
                input_fraction=sum(r['input_tokens'] for r in hot)/sum(r['input_tokens'] for r in selected),
                signed_loss_fraction=sum(r['signed_cache_loss_vs_c8'] for r in hot)/total_loss))
    writecsv('concentration.csv', concentration)
    # Retain a concrete same-conversation miss, plus the actual preceding
    # completion gap; these are client timestamps, not cache-commit timestamps.
    example_history = []
    for arm in ARMS:
        selected = sorted([r for r in rows if r['arm'] == arm and r['case'] == 'agentx_c128'
            and r['repeat'] == 'r2' and r['clone'] == 6
            and r['source_conversation'] == 'a3344328ef1674cda1018e0791fb85a839a8'],key=lambda r:r['turn'])
        prev = None
        for r in selected:
            if 6 <= r['turn'] <= 10:
                example_history.append(dict(**r,
                    since_previous_client_completion_s=r['started']-prev['finished'] if prev else None))
            prev = r
    writecsv('example_history.csv',example_history)

    # Hash exact launch/adapters, so a copied but unused older hook is not
    # mistaken for the installed runtime. No private configuration is copied.
    for run_name in sorted({e['run'] for e in episodes}):
        r = RUNS / run_name
        for relative in ('launch_manifest.json','agentx_graphs.json','code/runtime_adapter.py',
                         'code/h100_runtime_adapter.py','code/cache_probe.py','code/agentx_replay.py'):
            source_files.append(r / relative)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),cpu_only=True,
        engine_commit='37e0828c642ce4f3795c4c4394f4ce913bcd1a35',
        accepted_episodes_source=dict(path=str(accepted),sha256=sha(accepted)),
        verified_episodes=verified, request_count=len(rows), paired_c128_count=len(paired),
        c8_source_requests=len(baseline), metadata_gpu_plus_host_invariant=True,
        same_conversation_kept_same_cache_owner=True, resources=resources, summary=summary,
        log_sources=log_sources, source_files=[dict(path=str(p),sha256=sha(p)) for p in source_files],
        pair_metrics=pair_parts, concentration=concentration,
        cache_policy=dict(hicache_size_gb=32,hicache_ratio_argument=2.0,
            actual_sizing='Explicit size > 0 overrides ratio-based sizing; budget split by GPU KV/GDN byte proportions',
            write_policy='write_through',storage_backend=None,
            mamba_max_device_states_per_path=1,session_radix_cache=False),
        limits=['C8 is measured reuse at low pressure; a C128 loss relative to C8 is not an instrumented eviction event.',
        'Raw cached_tokens lacks separate full-KV match, Mamba match and restore-failure reasons.',
        'CPU pool warnings show limited capacity, not per-request causal attribution.',
        'Native usage logs subtract evictable cache; low logged usage does not prove free cache capacity.',
        'PP ranks store different layers of the same requests; count logical token/state capacity once per instance.',
        'All three layouts are P/D colocated, same E5 engine and V3 replay; no new GPU experiments.'])
    (HERE/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(summary=summary,resources=resources,pair_metrics=pair_parts,concentration=concentration),indent=2))


if __name__ == '__main__':
    main()
