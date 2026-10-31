"""Join intrusive P/D fingerprints by transfer room and serial request time."""
import argparse
import collections
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    evidence = json.loads((run / 'evidence/pd/host_restore_isolation.json').read_text())
    snapshots = []
    for p in sorted((run / 'evidence').glob('pd.r*/state_probe.*.json')):
        for row in json.loads(p.read_text()):
            snapshots.append(dict(row, source=str(p)))
    rooms = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in snapshots:
        rooms[row['bootstrap_room']][row['point']].append(row)
    pairs = []
    for room, points in rooms.items():
        pre = points['prefill_before_final_send']
        dec = points['decode_after_transfer_commit']
        if not pre or not dec:
            continue
        p_layers = {layer: value for s in pre for layer, value in s['mamba'].items()}
        d_layers = {layer: value for s in dec for layer, value in s['mamba'].items()}
        p_kv = {layer: value for s in pre for layer, value in s['kv'].items()}
        d_kv = {layer: value for s in dec for layer, value in s['kv'].items()}
        assert p_layers.keys() == d_layers.keys(), room
        assert p_kv.keys() == d_kv.keys(), room
        mismatched_mamba = [layer for layer in p_layers if p_layers[layer] != d_layers[layer]]
        mismatched_kv = [layer for layer in p_kv if p_kv[layer] != d_kv[layer]]
        case_name = None
        token_match = None
        stamp = min(s['time'] for s in dec)
        for case in evidence['cases']:
            for i, response in enumerate(case['responses']):
                meta = response['meta_info']
                if meta['request_received_ts'] <= stamp <= meta['request_finished_ts']:
                    assert case_name is None
                    case_name, token_match = case['name'], case['matches'][i]
        pairs.append(dict(room=room, case=case_name, output_matches=token_match,
                          input_tokens=pre[0]['input_tokens'],
                          prefill_stages=len(pre), decode_stages=len(dec),
                          mamba_layers=len(p_layers), kv_layers=len(p_kv),
                          mismatched_mamba=mismatched_mamba, mismatched_kv=mismatched_kv,
                          prefill_slots=[s['mamba_slot'] for s in pre],
                          decode_slots=[s['mamba_slot'] for s in dec],
                          mamba_field_diffs={layer: [name for name in p_layers[layer] if p_layers[layer][name] != d_layers[layer][name]]
                                             for layer in mismatched_mamba},
                          time=stamp))
    out = dict(scope='Intrusive state-copy diagnostic; hashes alone do not establish numerical equivalence of independently recomputed KV',
               snapshots=len(snapshots), paired_requests=len(pairs),
               completed_cases=len(evidence['cases']),
               mismatch_outputs=sum(not match for c in evidence['cases'] for match in c['matches']),
               restore_pairs=[p for p in pairs if p['case'] is not None], all_pairs=pairs)
    (run / 'analysis').mkdir(exist_ok=True)
    (run / 'analysis/state_pairs.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ['all_pairs', 'restore_pairs']}, indent=2))
    for p in out['restore_pairs']:
        print(p['case'], p['input_tokens'], 'output_match', p['output_matches'],
              'GDN mismatches', p['mismatched_mamba'], 'KV mismatches', p['mismatched_kv'])


if __name__ == '__main__':
    main()
