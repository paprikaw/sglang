"""Compare full-model probabilities only when token histories are identical."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def samples(path):
    data = json.loads(path.read_text())
    rounds = []
    for group in data['rounds']:
        rows = {}
        for request, row in enumerate(group['requests']):
            meta = row['validation_logits']
            chosen = meta['output_token_logprobs']
            tops = meta['output_top_logprobs']
            assert len(chosen) == len(tops) == data['output_tokens']
            history = []
            for step, (token, top) in enumerate(zip(chosen, tops)):
                values = {int(t[1]):float(t[0]) for t in top}
                assert len(values) == 10
                rows[(request, tuple(history))] = dict(step=step, token=int(token[1]), top=values)
                history.append(int(token[1]))
        rounds.append(rows)
    return rounds


def compare(left, right, label):
    positions, deltas = [], []
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        common = a['top'].keys() & b['top'].keys()
        assert len(common) >= 5
        errors = [abs(a['top'][token] - b['top'][token]) for token in common]
        positions.append(dict(comparison=label, request=key[0], step=a['step'],
            matched_history=True, common_top10=len(common), argmax_equal=a['token'] == b['token'],
            mean_abs_logprob_delta=statistics.mean(errors), max_abs_logprob_delta=max(errors)))
        deltas.extend(errors)
    assert positions
    summary = dict(comparison=label, common_history_positions=len(positions),
        candidate_positions=len(right), reference_positions=len(left),
        identical_argmax_positions=sum(p['argmax_equal'] for p in positions),
        argmax_agreement=sum(p['argmax_equal'] for p in positions) / len(positions),
        mean_top10_overlap=statistics.mean(p['common_top10'] for p in positions),
        mean_abs_logprob_delta=statistics.mean(deltas), max_abs_logprob_delta=max(deltas))
    return summary, positions


def main(batched=False):
    ref_run = 'r6_native_batch_validation' if batched else 'r4_reference_fixed'
    ref_path = ROOT / 'runs' / ref_run / 'pp2/validation.probabilities.json'
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    reference = samples(ref_path)
    summaries, positions = [], []
    for i, row in enumerate(reference[1:], 1):
        a, b = compare(reference[0], row, 'native_pp_self_r' + str(i + 1))
        summaries.append(a); positions += b
    comparisons = [('r7_adaptive_batch_validation','pp2')] if batched else [('r4_reference_fixed', 'ep2'), ('r5_adaptive_confirm', 'pp2')]
    for name, arm in comparisons:
        path = ROOT / 'runs' / name / arm / 'validation.probabilities.json'
        if not path.exists():
            continue
        for i, row in enumerate(samples(path)):
            a, b = compare(reference[0], row, name + '.' + arm + '.r' + str(i + 1))
            summaries.append(a); positions += b
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=True)
    report = dict(protocol='Top-10 natural log probabilities, same prompt and exact generated-token history; untimed 8-token generation',
        atomic_batched_input=batched,
        reference=str(ref_path), summary=summaries,
        long_greedy_text_is_not_bitwise_equivalent=True)
    suffix = '_batched' if batched else ''
    (out / ('probability_validation' + suffix + '.json')).write_text(json.dumps(report, indent=2))
    with (out / ('probability_positions' + suffix + '.csv')).open('w') as h:
        writer = csv.DictWriter(h, fieldnames=list(positions[0]))
        writer.writeheader(); writer.writerows(positions)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--batched',action='store_true')
    main(ap.parse_args().batched)
