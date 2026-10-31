"""Describe completed replay exposure and paired first-use effects.

Intervals come from client observations, not GPU kernel timings or actual
scheduler batches. All formal repeats remain in the experiment mean.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
from completed_episode_files import completed_formals, gate_sha256


def intervals(rows, decode_only=False):
    return [(r['started'] + (r['ttft_ms']/1000 if decode_only else 0), r['finished'])
            for r in rows if not decode_only or r['output_tokens'] > 1]


def exposure(spans, wall):
    events = collections.defaultdict(int)
    for start, end in spans:
        assert end >= start
        events[start] += 1
        events[end] -= 1
    durations = collections.defaultdict(float)
    count = peak = 0
    previous = None
    for timestamp, delta in sorted(events.items()):
        if previous is not None:
            durations[count] += timestamp-previous
        count += delta
        peak = max(peak, count)
        previous = timestamp
    assert count == 0
    busy = sum(t for n,t in durations.items() if n)
    area = sum(n*t for n,t in durations.items())
    assert busy <= wall + 1e-5
    return dict(time_mean_over_episode=area/wall, peak=peak, any_active_s=busy,
                no_active_s=max(0,wall-busy),
                seconds_at_or_above={str(n): sum(t for c,t in durations.items() if c >= n)
                                    for n in (2,4,8,16,32,64)})


def source_key(row):
    # Independent repetitions use separate synthetic token namespaces.
    normalize = lambda s: re.sub(r'^agentx-[^:]+-repeat\d+:', '', s)
    return (normalize(row['root_id']), normalize(row['conversation_id']), row['turn'])


def paired(first, later):
    a={source_key(r):r for r in first['requests']}
    b={source_key(r):r for r in later['requests']}
    assert a.keys() == b.keys()
    rows=[]
    for key,x in a.items():
        y=b[key]
        assert (x['input_tokens'],x['output_tokens']) == (y['input_tokens'],y['output_tokens'])
        rows.append(dict(source=list(key), input_tokens=x['input_tokens'],
                         cached_tokens=x['meta_info'].get('cached_tokens',0),
                         first_ttft_ms=x['ttft_ms'], later_ttft_ms=y['ttft_ms'],
                         delta_ttft_ms=x['ttft_ms']-y['ttft_ms'],
                         instance=x['decode_instance'], rank=x['routed_dp_rank']))
    return dict(requests=len(rows),
        routing_changes=sum((x['prefill_instance'],x['decode_instance'],x['routed_dp_rank']) !=
                            (b[k]['prefill_instance'],b[k]['decode_instance'],b[k]['routed_dp_rank'])
                            for k,x in a.items()),
        cached_token_changes=sum(x['meta_info'].get('cached_tokens') != b[k]['meta_info'].get('cached_tokens')
                                 for k,x in a.items()),
        mean_ttft_delta_ms=statistics.mean(r['delta_ttft_ms'] for r in rows),
        largest_positive_deltas=sorted(rows,key=lambda r:r['delta_ttft_ms'],reverse=True)[:12],
        limitation='Matched source shapes, routing and cache counters; synthetic token namespaces differ by repeat. This does not trace or prove JIT compilation.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args()
    run=args.run.resolve()
    trials=[]; groups=collections.defaultdict(dict); sources=[]
    files, pending = completed_formals(run)
    for path in files:
        data=json.loads(path.read_text());rows=data['requests']
        arm,case,policy=path.relative_to(run).parts[1:4]
        repeat=path.stem.split('.')[-1]
        groups[arm,case,policy][repeat]=data
        caches=[r['meta_info'].get('cached_tokens',0) or 0 for r in rows]
        assert all(0 <= c <= r['input_tokens'] for c,r in zip(caches,rows))
        instances=sorted({r['decode_instance'] for r in rows})
        instance_decode={i:exposure(intervals([r for r in rows if r['decode_instance']==i],True),data['wall_s'])
                         for i in instances}
        trials.append(dict(arm=arm,case=case,policy=policy,repeat=repeat,requests=len(rows),
            client_inflight=exposure(intervals(rows),data['wall_s']),
            client_after_first_token=exposure(intervals(rows,True),data['wall_s']),
            instance_after_first_token=instance_decode,
            mean_cached_tokens=statistics.mean(caches),
            mean_new_input_tokens=statistics.mean(r['input_tokens']-c for c,r in zip(caches,rows)),
            cache_token_fraction=sum(caches)/sum(r['input_tokens'] for r in rows),
            reported_host_hit_requests=sum((r['meta_info'].get('cached_tokens_details') or {}).get('host',0)>0 for r in rows),
            request_retractions=sum(r['meta_info'].get('num_retractions',0) or 0 for r in rows)))
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pairs={':'.join(key)+':r1_vs_'+repeat:paired(values['r1'],values[repeat])
           for key,values in groups.items() if 'r1' in values
           for repeat in ('r2','r3') if repeat in values}
    out=run/'analysis';out.mkdir(exist_ok=True)
    (out/'exposure.json').write_text(json.dumps(dict(trials=trials,paired_first_use=pairs,sources=sources,
        excluded_unmarked_formals=pending, completion_gate_sha256=gate_sha256(),
        scope='Client in-flight and after-first-token intervals only; these include queuing/streaming and are not actual GPU batches. All formal samples retained.'),indent=2)+'\n')
    lines=['# AgentX replay exposure and first-use audit','',
        'Only episodes with passed completion markers. Client intervals only: after-first-token concurrency is not a measured GPU Decode batch. Means below are time-weighted over the whole episode.','',
        '| Arm / case / repeat | In-flight mean | After-first-token mean | After-first-token peak | No in-flight seconds | Mean cached input tokens | Mean new input tokens |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t in trials:
        lines.append(f"| {t['arm']} / {t['case']} / {t['repeat']} | {t['client_inflight']['time_mean_over_episode']:.2f} | {t['client_after_first_token']['time_mean_over_episode']:.2f} | {t['client_after_first_token']['peak']} | {t['client_inflight']['no_active_s']:.2f} | {t['mean_cached_tokens']:.2f} | {t['mean_new_input_tokens']:.2f} |")
    lines += ['','First-use audit:','',
        '| Paired source requests | Routing changes | Cache counter changes | Mean first-minus-later TTFT ms |',
        '|---|---:|---:|---:|']
    for name,p in pairs.items():
        lines.append(f"| {name} ({p['requests']}) | {p['routing_changes']} | {p['cached_token_changes']} | {p['mean_ttft_delta_ms']:.2f} |")
    lines += ['', 'All repeats remain in the formal mean. Differences here motivate explicit untimed all-rank warmup; they do not prove a specific compilation or transfer cause.',
              'See [raw exposure and paired evidence](exposure.json) for instance-level durations, host-hit/retraction counters, matched request IDs and source hashes.']
    (out/'EXPOSURE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(trials=len(trials),paired_audits=len(pairs))))


if __name__=='__main__':
    main()
