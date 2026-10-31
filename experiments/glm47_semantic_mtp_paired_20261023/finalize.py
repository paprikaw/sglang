"""Publish only a complete, cleaned, semantic pair; never import synthetic data."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run
    results=json.loads((run/'task/results.json').read_text())
    cleanup=json.loads((run/'task/cleanup.json').read_text())
    if not results['passed'] or not cleanup['passed'] or results['valid_points']!=2:
        raise RuntimeError('Both actual semantic arms and owned GPU cleanup must pass')
    points={}
    digests=[]
    for arm in results['arms']:
        point=arm['results'][0]
        if (not arm['passed'] or not point['old_context_constructed_by_model']
            or point['synthetic_kv'] or point['request_count']!=8
            or point['actual_cached_tokens_per_request']!=81856):
            raise RuntimeError('Unmatched or synthetic data rejected')
        digests.append([r['sha256'] for r in point['semantic_input_manifest']['requests']])
        points[arm['layout']]=point
    if digests[0]!=digests[1]:
        raise RuntimeError('EP8 and PP used different semantic prompts')
    value=dict(protocol='semantic_model_prefill_to_native_hicache_then_cold_mtp_decode',
               workload=dict(batch=8,raw_prefix=81920,query=1,output=128,
                             actual_cached_prefix=81856,initial_recompute=65),
               same_semantic_input_digests=True,synthetic_results_excluded=True,
               numerical_parity_with_non_speculative_not_claimed=True,points=points)
    (run/'comparison.json').write_text(json.dumps(value,indent=2)+'\n')
    text=['# GLM-4.7-AWQ semantic MTP Decode', '',
          '065/167, 8 A100; native EAGLE 2/1/3. B8, raw P81920 + Q1 -> 128.',
          'Actual cached prefix 81856/request; initial 65 tokens recomputed.',
          'Real model Prefill and warmup excluded; HiCache H2D included in TTFT.', '',
          '| Metric (request means) | EP8 | PP2 x EP4 |',
          '|---|---:|---:|']
    for label,key,factor in [('TTFT (s)','ttft_s',1),('TPOT (ms)','tpot_s',1000),
                             ('End-to-end (s)','end_to_end_s',1),
                             ('Reported accept length','spec_accept_length',1),
                             ('Reported draft acceptance (%)','spec_accept_rate',100)]:
        text.append(f"| {label} | {points['ep8'][key]['mean']*factor:.3f} | {points['pp2ep4'][key]['mean']*factor:.3f} |")
    text.append(f"| Global output tok/s | {points['ep8']['throughput_tok_s']:.3f} | {points['pp2ep4']['throughput_tok_s']:.3f} |")
    text.extend(['', 'Mean, median, nearest-rank sample P99, per-request streaming events,',
                 'acceptance counts and actual native graph-replay proofs are in comparison.json.',
                 'N=8 P99 is only the sample maximum. No synthetic baseline is compared.',
                 'Short semantic sanity outputs are preserved separately in each arm.',
                 'Component-profile measurements are separate from these production metrics.', ''])
    (run/'RESULTS.md').write_text('\n'.join(text))
    print('\n'.join(text))


if __name__=='__main__':
    main()
