"""Offline proof for completed arms, preserving whole-run Guard failures.

Use the original Guard loader-delta validator. Never alter a frozen spec,
binding, result, or failed postcheck to make an incomplete run look successful.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for run in args.run:
        run = run.resolve()
        cfg = read(run / 'config.json')
        sys.path.insert(0, cfg['guard_scripts'])
        import model_cache_adapter
        spec = read(run / 'dashboard_spec.json')
        binding_path = run / 'dashboard_guard/model_cache_bindings.json'
        binding = read(binding_path)
        assert binding['experiment_id'] == cfg['id']
        assert binding['spec_sha256'] == model_cache_adapter._json_sha256(spec)
        assert binding['lease_spec_sha256'] == sha(run / 'dashboard_spec.json')
        contract = model_cache_adapter.validate_contract(spec, launch=False)
        assert binding['contract_sha256'] == model_cache_adapter._json_sha256(contract)
        assert binding['lease_receipt_sha256'] == sha(binding['lease_receipt_path'])
        assert set(binding['nodes']) == {cfg['node']}
        assert all(sha(d['path']) == d['sha256'] for d in spec['frozen_inputs'])
        node = binding['nodes'][cfg['node']]
        expected = cfg['model_cache']['usage_logs'][cfg['node']]
        actual = [{k: u[k] for k in ('id', 'path', 'expected_ranks')} for u in node['usage_logs']]
        assert actual == expected
        uses = {u['id']: u for u in node['usage_logs']}
        cleanup = read(run / 'task/cleanup.json')
        assert cleanup['passed'] and cleanup['model_processes_exited']
        arms = []
        for layout in cfg['layouts']:
            points = sorted((run / 'results' / layout['id']).glob('*/completed.json'))
            if not points:
                continue
            assert all(read(p)['passed'] for p in points)
            assert read(run / 'evidence' / layout['id'] / 'cleanup.json')['cleanup_verified']
            loaders, qualifications = [], []
            for replica, descriptor in enumerate(layout['usage_logs']):
                loaders.append(model_cache_adapter._verify_usage_log(
                    uses[descriptor['id']], node['group_key'], binding['admitted_at']))
                alias = 'ep2' if layout['ep'] > 1 else 'pp2'
                qfiles = sorted((run / 'evidence' / f'{layout["id"]}.r{replica}' / alias / 'qualification').glob('*.json'))
                assert len(qfiles) == layout['pp'] * layout['ep']
                values = [read(p) for p in qfiles]
                assert all(q['passed'] and q['weight_dtype'] == 'torch.bfloat16'
                           and q['expert_shape'][0] == 256 // layout['ep'] for q in values)
                assert sum(q['layers'] for q in values if q['tp_rank'] == 0) == 40
                qualifications.extend(dict(replica=replica, path=str(p), sha256=sha(p), result=q)
                                      for p, q in zip(qfiles, values))
            arms.append(dict(arm=layout['id'], completed_points=len(points),
                             accepted_completed_arms_only=True, loader_checks=loaders,
                             qualification=qualifications, cleanup_verified=True))
        posts = [read(p) for p in sorted((run / 'dashboard_guard/checks').glob('*-post.json'))]
        reports.append(dict(run=str(run), spec_sha256=sha(run / 'dashboard_spec.json'),
                            binding_sha256=sha(binding_path), frozen_inputs_match=True,
                            whole_run_guard_postpassed=bool(posts) and posts[-1].get('passed') is True,
                            whole_run_scientific_success=bool(posts) and posts[-1].get('scientific_success') is True,
                            progress=read(run / 'task/progress.json'), cleanup=cleanup,
                            completed_arm_loader_and_model_checks_passed=True, arms=arms,
                            limitation='A failed final startup or incomplete grid remains a failed whole run; only listed completed arms are accepted.'))
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(runs=reports), indent=2) + '\n')
    print(json.dumps(dict(runs=len(reports), arms=sum(len(r['arms']) for r in reports),
                          completed_points=sum(a['completed_points'] for r in reports for a in r['arms']),
                          whole_run_postpassed=[r['whole_run_guard_postpassed'] for r in reports],
                          output=str(path))))


if __name__ == '__main__':
    main()
