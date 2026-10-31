"""Keep the first completed E8/V4 pilot separate from the 21-episode profile."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RAW = WORK / ('experiments/agentx_pd_dynamic_20261030/runs/'
    'two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot/results/'
    'pd_2p6d_pp2_ep4ep2/agentx_c8/sticky_hash/formal.r1.json')


def main():
    payload = RAW.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == '24dc6cc2f2724baf0be0b54841f853e7c09434ff6ae2cada784eb2cef7268b4d'
    marker = RAW.with_name('completed.r1.json')
    assert json.loads(marker.read_text())['passed']
    raw = json.loads(payload)
    rows = raw['requests']
    assert len(rows) == raw['completed_requests'] == 171
    with (HERE.parent/'agentx_workload_profile_20261031/request_profiles.csv').open() as src:
        reference = {(r['source_conversation'],int(r['turn'])):
                     (int(r['context_tokens']),int(r['decode_tokens']))
                     for r in csv.DictReader(src) if r['arm']=='col_ep4' and
                     r['case']=='agentx_c8' and r['repeat']=='r1'}
    for r in rows:
        identity = r['conversation_id'].split(':',2)[-1], r['turn']
        assert reference[identity] == (r['input_tokens'],r['output_tokens'])
        meta = r['prefill_meta']
        details = meta.get('cached_tokens_details') or dict(device=0,host=0)
        assert details['device']+details['host'] == meta['cached_tokens']
    total_input = sum(r['input_tokens'] for r in rows)
    cached = sum(r['prefill_meta']['cached_tokens'] for r in rows)
    device = sum((r['prefill_meta'].get('cached_tokens_details') or {}).get('device',0) for r in rows)
    host = sum((r['prefill_meta'].get('cached_tokens_details') or {}).get('host',0) for r in rows)
    point = dict(cohort='E8/V4',arm='PP2 P + EP4/EP2 D, 2P6D',case='agentx_c8',repeats=1,
        requests=171,mean_input_tokens=total_input/171,
        mean_output_tokens=statistics.fmean(r['output_tokens'] for r in rows),
        mean_new_prefill_tokens=(total_input-cached)/171,
        token_weighted_prefix_hit_pct=100*cached/total_input,
        token_weighted_device_hit_pct=100*device/total_input,
        token_weighted_host_hit_pct=100*host/total_input,
        any_prefix_hit_requests_pct=100*sum(r['prefill_meta']['cached_tokens']>0 for r in rows)/171,
        status='Completed episode marker; overall 4-point pilot not yet complete at this observation')
    with (HERE/'pilot_cache_supplement.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(point));writer.writeheader();writer.writerow(point)
    evidence=dict(point=point,path=str(RAW),sha256=digest,marker=str(marker),
                  marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
                  policy=raw['branch_timing_policy'],
                  limitation='One corrected-timing episode; separate from historical V3, no pooled mean or layout-benefit claim')
    (HERE/'pilot_cache_supplement.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(point,ensure_ascii=False))


if __name__=='__main__':
    main()
