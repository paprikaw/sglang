"""Audit completed immutable AgentX episodes and summarize arithmetic means."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from completed_episode_files import completed_formals, gate_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run/'config.json').read_text())
    graphs = {g['id']: g for g in json.loads(Path(cfg['graphs_file']).read_text())['graphs']}
    cases = {c['id']: c for c in cfg['cases']}
    trials, requests, sources, issues = [], [], [], []
    files, pending = completed_formals(run)
    for path in files:
        arm, case_id, policy = path.relative_to(run).parts[1:4]
        case = cases[case_id]
        data = json.loads(path.read_text())
        rows = data['requests']
        expected = sum(sum(len(c['turns']) for c in graphs[g]['conversations']) for g in case['graph_ids'])*case.get('copies', 1)
        assert data['completed_requests'] == len(rows) == expected, (path, 'request count')
        ids = [(r['root_id'], r['conversation_id'], r['turn']) for r in rows]
        assert len(set(ids)) == expected, (path, 'duplicate source turn')
        for row in rows:
            graph_id = next(g for g in case['graph_ids'] if row['root_id'].endswith(g))
            conv = next(c for c in graphs[graph_id]['conversations'] if row['conversation_id'].endswith(c['session_id']))
            source = conv['turns'][row['turn']]
            assert row['input_tokens'] == source['input_tokens']
            assert row['output_tokens'] == source['output_tokens']
            assert len(row['output_ids']) == source['output_tokens']
            if row['tpot_ms'] is not None:
                assert abs(row['e2e_ms']-row['ttft_ms']-(row['output_tokens']-1)*row['tpot_ms']) < 1e-5
            for side, meta in [('client', row['meta_info']), ('prefill', row.get('prefill_meta', {}))]:
                if (meta.get('cached_tokens') or 0) > row['input_tokens']:
                    issues.append(dict(path=str(path), conversation=row['conversation_id'], turn=row['turn'], side=side, issue='reported cached tokens exceed input'))
        means = {m: statistics.mean(r[m] for r in rows if r[m] is not None) for m in ('ttft_ms','tpot_ms','e2e_ms','client_dispatch_lag_ms')}
        assert all(abs(means[m]-data['means'][m]) < 1e-5 for m in means)
        record = dict(arm=arm, case=case_id, policy=policy, repeat=path.stem.split('.')[-1], requests=expected,
                      **means, wall_s=data['wall_s'], output_tokens_per_s=data['output_tokens_per_s'],
                      switched_requests=data['switched_requests'])
        trials.append(record)
        for row in rows:
            requests.append(dict(arm=arm, case=case_id, policy=policy, repeat=record['repeat'],
                **{k: row.get(k) for k in ('root_id','conversation_id','turn','source_kind','input_tokens','output_tokens',
                    'ttft_ms','tpot_ms','e2e_ms','client_dispatch_lag_ms','prefill_instance','decode_instance','routed_dp_rank','switched')},
                client_cached_tokens=row['meta_info'].get('cached_tokens'),
                prefill_cached_tokens=row.get('prefill_meta',{}).get('cached_tokens')))
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    grouped = {}
    for row in trials:
        grouped.setdefault((row['arm'],row['case'],row['policy']),[]).append(row)
    summaries = []
    for key, group in grouped.items():
        assert len({g['requests'] for g in group}) == 1
        summaries.append(dict(arm=key[0], case=key[1], policy=key[2], repeats=len(group),
            complete_point=len(group)==cfg['repeats'],
            **{k:statistics.mean(g[k] for g in group) for k in ('ttft_ms','tpot_ms','e2e_ms','wall_s','output_tokens_per_s','switched_requests')}))
    out = run/'analysis';out.mkdir(exist_ok=True)
    for name, records in [('trials.csv',trials),('requests.csv',requests),('means.csv',summaries)]:
        if records:
            with (out/name).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
        elif (out/name).exists():
            (out/name).unlink()
    evidence = dict(episode_audit_passed=True, finished_episodes=len(trials), planned_episodes=cfg['expected_points'],
        all_episodes_finished=len(trials)==cfg['expected_points'], complete_points=sum(s['complete_point'] for s in summaries),
        request_count=len(requests), sources=sources, cache_accounting_issues=issues,
        excluded_unmarked_formals=pending, completion_gate_sha256=gate_sha256(),
        scope='Audits completed episode content only; does not replace Guard cleanup, source qualification, or cross-arm completeness.')
    (out/'audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    lines=['# Eight-H100 AgentX episode means (incomplete until every planned episode finishes)','',
           'Absolute client latency in ms; TPOT in ms/token. Only episodes with a passed completion marker enter this table; an episode marker does not replace whole-run qualification and cleanup. Arithmetic means over the identical source request set. `sticky_hash` is the historical code label for first-arrival round-robin placement with subsequent conversation/rank affinity.','',
           '| Arm | Case | Policy | Repeats | TTFT ms | TPOT ms/token | E2E ms | Wall s |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['arm']} | {s['case']} | {s['policy']} | {s['repeats']} | {s['ttft_ms']:.2f} | {s['tpot_ms']:.3f} | {s['e2e_ms']:.2f} | {s['wall_s']:.3f} |")
    lines += ['',f"Completed episodes: {len(trials)}/{cfg['expected_points']}. The presence of one arm does not establish a P/D advantage.",
              'Client cache fields can combine P-side accounting with D-side reuse; they are not direct measurements of per-role transfer bytes.']
    (out/'MEANS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(finished_episodes=len(trials),complete_points=evidence['complete_points'],requests=len(requests),cache_accounting_issues=len(issues))))


if __name__ == '__main__':
    main()
