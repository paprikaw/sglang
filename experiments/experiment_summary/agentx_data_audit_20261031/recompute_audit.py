"""Offline audit of saved AgentX measurements. Does not launch serving processes."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[2]
ROOT = WORKSPACE / 'experiments/agentx_pd_dynamic_20261030'
SELECTED = {
    'eight_gpu_col_ep4_durable_r3': ('E5/V3', 'col_ep4'),
    'eight_gpu_col_pp2_durable_m16_r3': ('E5/V3', 'col_pp2'),
    'eight_gpu_col_single_durable_kv40_r3': ('E5/V3', 'col_single'),
    'eight_gpu_pd_ep4_e7_t300_b7200_r1': ('E7/V3', 'pd_ep4'),
}
E5 = '37e0828c642ce4f3795c4c4394f4ce913bcd1a35'
E7 = '797f6488cf5fea2f88e38ccaeb1703443deec537'
SOURCES = {}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, required=True):
    if not path.exists() and not required:
        return {}
    digest = sha(path)
    SOURCES[str(path)] = digest
    return json.loads(path.read_text())


def csv_write(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def call_signature(row):
    return (row['source_outer_idx'], row['source_inner_idx'],
            row['input_tokens'], row['output_tokens'])


def main():
    episodes, inventory, files, comparisons, slo_rows, historical = [], [], [], [], [], []
    main_by_key, raw_sha_paths = {}, defaultdict(list)
    common_graph_sha = None
    imported = []
    for name, (cohort, arm) in SELECTED.items():
        run = ROOT/'runs'/name
        manifest = read(run/'launch_manifest.json')
        assert manifest['engine_commit'] == (E5 if cohort == 'E5/V3' else E7)
        progress, cleanup = read(run/'task/progress.json'), read(run/'task/cleanup.json')
        assert progress['stage'] == 'complete'
        assert cleanup['passed'] and cleanup['model_processes_exited'] and cleanup['gpu_idle']
        posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
        assert posts and read(posts[-1])['passed']
        graph_path = run/'agentx_graphs.json'
        graphs = read(graph_path)
        if common_graph_sha is None:
            common_graph_sha = sha(graph_path)
        assert sha(graph_path) == common_graph_sha
        graph_index = {g['id']:g for g in graphs['graphs']}
        case_index = {c['id']:c for c in manifest['cases']}
        for frozen in manifest.get('frozen_inputs', []):
            p = Path(frozen['path'])
            # Recheck the frozen workload/replayer, not unrelated Guard assets.
            if p.is_relative_to(run) and (p.name == 'agentx_graphs.json' or 'replay' in p.name):
                assert sha(p) == frozen['sha256'], p
                SOURCES[str(p)] = frozen['sha256']
        for rel, record in manifest.get('imported_episodes', {}).items():
            for item in record['artifacts']:
                if item['relative_path'] != rel:
                    continue
                original, current = Path(item['source']), run/rel
                assert sha(original) == sha(current) == item['sha256']
                imported.append(dict(original=str(original), imported=str(current),
                                     sha256=item['sha256']))
        formals = sorted(run.glob(f'results/{arm}/*/sticky_hash/formal.r*.json'))
        assert len(formals) == (6 if cohort == 'E5/V3' else 3)
        for path in formals:
            case = path.parent.parent.name
            repeat = path.stem.split('.')[-1]
            marker = path.with_name('completed.'+repeat+'.json')
            assert read(marker)['passed']
            d = read(path)
            requests, cfg = d['requests'], case_index[case]
            expected = {}
            for gid in cfg['graph_ids']:
                source = graph_index[gid]
                expected[gid] = Counter(call_signature(t)
                    for conversation in source['conversations'] for t in conversation['turns'])
                assert sum(expected[gid].values()) == source['retained_requests']
            by_root = defaultdict(list)
            request_keys = set()
            for req in requests:
                key = (req['root_id'], req['conversation_id'], req['turn'])
                assert key not in request_keys
                request_keys.add(key)
                by_root[req['root_id']].append(req)
                assert len(req['output_ids']) == req['output_tokens']
                assert req['meta_info']['completion_tokens'] == req['output_tokens']
                assert req['meta_info']['prompt_tokens'] == req['input_tokens']
                for metric in ['ttft_ms', 'e2e_ms', 'client_dispatch_lag_ms']:
                    assert math.isfinite(req[metric]) and req[metric] >= 0
                assert req['e2e_ms'] >= req['ttft_ms']
            root_graph_counts = Counter()
            for rid, rows in by_root.items():
                gid = rid.rsplit(':', 1)[-1]
                assert Counter(call_signature(t) for t in rows) == expected[gid]
                root_graph_counts[gid] += 1
            assert root_graph_counts == Counter({g:cfg['copies'] for g in cfg['graph_ids']})
            assert len(requests) == d['completed_requests'] == 171*cfg['copies']
            row = dict(cohort=cohort, run=name, arm=arm, case=case, repeat=repeat,
                       engine_commit=manifest['engine_commit'], roots=len(by_root),
                       completed_requests=len(requests), source_family_count=len(expected),
                       copies_per_family=cfg['copies'], wall_s=d['wall_s'],
                       idle_compressed_s=d['idle_compressed_s'])
            for metric in ['ttft_ms', 'tpot_ms', 'e2e_ms', 'client_dispatch_lag_ms']:
                values = [r[metric] for r in requests if r[metric] is not None]
                value = mean(values)
                assert math.isclose(value, d['means'][metric], rel_tol=1e-11, abs_tol=1e-7)
                row[metric] = value
            row['ready_to_last_ms'] = mean(r['client_dispatch_lag_ms']+r['e2e_ms'] for r in requests)
            row['ready_ttft_ms'] = mean(r['client_dispatch_lag_ms']+r['ttft_ms'] for r in requests)
            row['interval_tpot_ms'] = sum(r['e2e_ms']-r['ttft_ms'] for r in requests)/sum(r['output_tokens']-1 for r in requests)
            row['dispatch_lag_over_100ms_requests'] = sum(r['client_dispatch_lag_ms'] > 100 for r in requests)
            row['dispatch_lag_over_1000ms_requests'] = sum(r['client_dispatch_lag_ms'] > 1000 for r in requests)
            row['input_tokens'] = sum(r['input_tokens'] for r in requests)
            row['output_tokens'] = sum(r['output_tokens'] for r in requests)
            row['cached_tokens_per_request'] = mean(r['meta_info'].get('cached_tokens', 0) for r in requests)
            row['num_retractions'] = sum(r['meta_info'].get('num_retractions', 0) for r in requests)
            row['root_first_dispatch_to_last_response_s'] = mean(
                max(r['finished'] for r in rows)-min(r['started'] for r in rows)
                for rows in by_root.values())
            row['root_mean_s'] = mean(
                max(r['finished'] for r in rows)-min(r['started']-r['client_dispatch_lag_ms']/1000 for r in rows)
                for rows in by_root.values())
            row['source'] = str(path)
            row['sha256'] = SOURCES[str(path)]
            episodes.append(row)
            main_by_key[(arm, case, repeat)] = requests
            for old_slo in d['slo']:
                ttft, tpot = old_slo['ttft_ms'], old_slo['tpot_ms']
                before = sum(r['ttft_ms'] <= ttft and
                             (r['tpot_ms'] is None or r['tpot_ms'] <= tpot) for r in requests)
                after = sum(r['ttft_ms']+r['client_dispatch_lag_ms'] <= ttft and
                            (r['tpot_ms'] is None or r['tpot_ms'] <= tpot) for r in requests)
                assert before == old_slo['passed_requests']
                assert math.isclose(before/d['wall_s'], old_slo['goodput_rps'])
                slo_rows.append(dict(cohort=cohort, arm=arm, case=case, repeat=repeat,
                    ttft_threshold_ms=ttft, request_mean_tpot_threshold_ms=tpot,
                    dispatch_start_passed=before, ready_time_passed=after,
                    dispatch_start_goodput_rps=before/d['wall_s'],
                    ready_time_goodput_rps=after/d['wall_s']))

    assert len(episodes) == 21
    assert len(imported) == 4
    assert sum(r['completed_requests'] for r in episodes if r['cohort'] == 'E5/V3') == 26163
    assert sum(r['completed_requests'] for r in episodes if r['cohort'] == 'E7/V3') == 8208
    groups = defaultdict(list)
    for row in episodes:
        groups[(row['cohort'], row['arm'], row['case'])].append(row)
    means = []
    for (cohort, arm, case), rows in groups.items():
        assert len(rows) == 3
        row = dict(cohort=cohort, arm=arm, case=case, independent_episodes=3,
                   requests_per_episode=rows[0]['completed_requests'])
        for metric in ['ttft_ms', 'tpot_ms', 'interval_tpot_ms', 'e2e_ms', 'root_mean_s', 'root_first_dispatch_to_last_response_s', 'wall_s',
                       'idle_compressed_s', 'client_dispatch_lag_ms', 'ready_to_last_ms',
                       'ready_ttft_ms',
                       'cached_tokens_per_request', 'num_retractions',
                       'dispatch_lag_over_100ms_requests', 'dispatch_lag_over_1000ms_requests']:
            row[metric] = mean(r[metric] for r in rows)
        means.append(row)
    for source_dir in ['matrix_analysis', 'e7_t300_b7200_analysis']:
        source = ROOT/source_dir/'means.csv'
        SOURCES[str(source)] = sha(source)
        with source.open() as stream:
            old_rows = list(csv.DictReader(stream))
        for current in means:
            if (current['cohort'] == 'E5/V3') != (source_dir == 'matrix_analysis'):
                continue
            old = [r for r in old_rows if r['arm'] == current['arm'] and r['case'] == current['case']]
            assert len(old) == 1
            for new_key, old_key in [('ttft_ms','ttft_ms'), ('tpot_ms','request_tpot_ms'),
                                     ('e2e_ms','e2e_ms'), ('root_mean_s','root_mean_s'), ('wall_s','wall_s')]:
                assert math.isclose(current[new_key], float(old[0][old_key]), rel_tol=1e-10, abs_tol=1e-6)
    for (arm, case, repeat), requests in main_by_key.items():
        if arm == 'col_ep4':
            continue
        base = {(r['root_id'], r['conversation_id'], r['turn']):r
                for r in main_by_key[('col_ep4', case, repeat)]}
        count = Counter()
        for r in requests:
            x = base[(r['root_id'], r['conversation_id'], r['turn'])]
            assert call_signature(x) == call_signature(r)
            a, b = x['output_ids'], r['output_ids']
            count['paired_requests'] += 1
            count['whole_sequence_equal'] += a == b
            count['first_token_equal'] += a[0] == b[0]
            if not x['meta_info'].get('cached_tokens') and not r['meta_info'].get('cached_tokens'):
                count['cold_on_both'] += 1
                count['cold_first_token_mismatch'] += a[0] != b[0]
        comparisons.append(dict(arm=arm, case=case, repeat=repeat, **count))

    for run in sorted((ROOT/'runs').iterdir()):
        if not run.is_dir():
            continue
        manifest = read(run/'launch_manifest.json', False)
        cfg, progress = read(run/'config.json', False), read(run/'task/progress.json', False)
        cleanup = read(run/'task/cleanup.json', False)
        formals = sorted(run.glob('results/*/*/*/formal.r*.json'))
        completed = 0
        for path in formals:
            repeat = path.stem.split('.')[-1]
            marker = path.with_name('completed.'+repeat+'.json')
            passed = bool(read(marker, False).get('passed'))
            completed += passed
            digest = sha(path)
            raw_sha_paths[digest].append(str(path))
            files.append(dict(run=run.name, arm=path.parts[-4], case=path.parts[-3],
                              policy=path.parent.name, repeat=repeat, completion_marker_passed=passed,
                              selected_main=run.name in SELECTED, sha256=digest, path=str(path)))
            if run.name not in SELECTED:
                d = read(path)
                old_row = dict(run=run.name, engine_commit=manifest['engine_commit'],
                    arm=path.parts[-4], case=path.parts[-3], policy=path.parent.name,
                    repeat=repeat, completion_marker_passed=passed,
                    duplicate_of_main=any(x['sha256'] == digest for x in episodes),
                    requests=len(d['requests']), wall_s=d['wall_s'],
                    cohort='historical_separate_not_main_matrix', source=str(path), sha256=digest)
                assert old_row['requests'] == d['completed_requests']
                for metric in ['ttft_ms','tpot_ms','e2e_ms']:
                    values = [r[metric] for r in d['requests'] if r[metric] is not None]
                    old_row[metric] = mean(values)
                    assert math.isclose(old_row[metric], d['means'][metric], abs_tol=1e-7, rel_tol=1e-11)
                old_row['interval_tpot_ms'] = sum(r['e2e_ms']-r['ttft_ms'] for r in d['requests'])/sum(r['output_tokens']-1 for r in d['requests'])
                historical.append(old_row)
        if run.name in SELECTED:
            category = 'selected_accepted_cohort'
        elif formals:
            category = 'historical_or_diagnostic_not_main_matrix'
        elif '_v4_b7200_' in run.name:
            category = 'prepared_only_no_gpu_episode'
        else:
            category = 'no_complete_formal_unqualified_or_prepared'
        inventory.append(dict(run=run.name, category=category, stage=progress.get('stage', ''),
            engine_commit=manifest.get('engine_commit', cfg.get('engine_commit', '')),
            arms=';'.join(manifest.get('arms', cfg.get('arms', []))),
            expected_episodes=cfg.get('expected_points', progress.get('expected', '')),
            formal_artifacts=len(formals), completion_markers_passed=completed,
            cleanup_passed=cleanup.get('passed', ''), path=str(run)))
    duplicates = [dict(sha256=h, paths=paths) for h, paths in raw_sha_paths.items() if len(paths)>1]
    csv_write('accepted_episodes.csv', episodes)
    csv_write('accepted_means.csv', means)
    csv_write('output_agreement.csv', comparisons)
    csv_write('latency_scope_slo.csv', slo_rows)
    csv_write('run_inventory.csv', inventory)
    csv_write('formal_inventory.csv', files)
    csv_write('historical_episodes.csv', historical)
    old_groups = defaultdict(list)
    for row in historical:
        if not row['duplicate_of_main']:
            old_groups[(row['run'],row['arm'],row['case'],row['policy'])].append(row)
    old_means = []
    for (run,arm,case,policy), rows in old_groups.items():
        assert len(rows) == 3 and all(r['completion_marker_passed'] for r in rows)
        row = dict(run=run, engine_commit=rows[0]['engine_commit'], arm=arm, case=case,
            policy=policy, repeats=len(rows), requests_per_episode=rows[0]['requests'],
            cohort='historical_separate_not_main_matrix')
        row.update({k:mean(r[k] for r in rows) for k in ['ttft_ms','tpot_ms','interval_tpot_ms','e2e_ms','wall_s']})
        old_means.append(row)
    csv_write('historical_means.csv', old_means)
    result = dict(updated_at=datetime.now(ZoneInfo('Australia/Melbourne')).isoformat(),
                  offline_only=True, selected_accepted_episodes=len(episodes),
                  selected_requests=sum(r['completed_requests'] for r in episodes),
                  graph_sha256=common_graph_sha, imported_episodes=imported,
                  run_directories=len(inventory), formal_artifacts=len(files),
                  unique_formal_sha256=len(raw_sha_paths), duplicate_formal_artifacts=duplicates,
                  separate_historical_episodes=sum(len(rows) for rows in old_groups.values()),
                  separate_historical_points=len(old_means),
                  categories=dict(Counter(r['category'] for r in inventory)),
                  selected_completion_integrity='passed',
                  correctness_scope='length/completion/identity checks passed; numeric output equivalence unresolved',
                  measured_workload_scope='V3 custom compressed hash-token replay of eight source families',
                  sources=[dict(path=p, sha256=h) for p,h in sorted(SOURCES.items())],
                  analyzer_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','duplicate_formal_artifacts','imported_episodes']}))
    print(json.dumps([{k:r[k] for k in ['cohort','arm','case','ttft_ms','tpot_ms','e2e_ms',
                                      'client_dispatch_lag_ms']} for r in means]))


if __name__ == '__main__':
    main()
