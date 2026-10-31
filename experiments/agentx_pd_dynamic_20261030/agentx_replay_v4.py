"""Finite, dependency-preserving AgentX-derived replay against real SGLang.

All times are client wall time. Authored waits use a virtual clock that advances
normally during inference; only system-wide idle gaps above the declared cap are
compressed. Request lengths, all retained child turns, and graph edges remain.
"""
import concurrent.futures as cf
import hashlib
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from agentx_branch_timing import POLICY as BRANCH_POLICY, build_vertices, due_time


def http(host, port, path, key, payload=None, timeout=300):
    request = urllib.request.Request(f'http://{host}:{port}{path}',
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()[:500]}') from None
    return json.loads(body) if path != '/flush_cache?timeout=60' else body.decode()


def stream(endpoint, key, payload, timeout=600):
    start = time.perf_counter()
    request = urllib.request.Request(f"http://{endpoint['host']}:{endpoint['port']}/generate",
        data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    first = last = final = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                if line[6:].strip() == b'[DONE]':
                    break
                value = json.loads(line[6:])
                meta = value.get('meta_info', {})
                if 'error' in value:
                    raise RuntimeError(str(value['error']))
                finish = meta.get('finish_reason')
                if isinstance(finish, dict) and finish.get('type') in ('abort', 'error'):
                    raise RuntimeError(str(finish))
                if meta.get('completion_tokens', 0):
                    last = time.perf_counter()
                    first = last if first is None else first
                final = value
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()[:500]}') from None
    if first is None or final is None:
        raise RuntimeError('Incomplete response stream')
    meta = final['meta_info']
    count = meta['completion_tokens']
    if count != payload['sampling_params']['max_new_tokens']:
        raise RuntimeError(f'Incomplete output: {count}')
    return dict(started=start, finished=last, ttft_ms=1000*(first-start),
                tpot_ms=1000*(last-first)/(count-1) if count > 1 else None,
                e2e_ms=1000*(last-start), output_ids=final.get('output_ids'),
                output_tokens=count, meta_info=meta)


def request_one(prefill, decode, key, tokens, output, lane, return_logprob=False):
    target = decode or prefill
    payload = dict(input_ids=tokens, stream=True, sampling_params={
        'temperature': 0, 'max_new_tokens': output, 'ignore_eos': True})
    if return_logprob:
        payload['return_logprob'] = True
        payload['logprob_start_len'] = len(tokens) - 1
    if target['ep'] > 1:
        payload['routed_dp_rank'] = target.get('rank', lane % target['ep'])
    if decode is None:
        return stream(target, key, payload)
    payload.update(bootstrap_host=prefill['host'], bootstrap_port=prefill['bootstrap_port'],
                   bootstrap_room=random.randrange(1, 2**62), disagg_prefill_dp_rank=prefill.get('rank', lane % prefill['ep']))
    pre = dict(payload, stream=False)
    if prefill['ep'] > 1:
        pre['routed_dp_rank'] = prefill.get('rank', lane % prefill['ep'])
    else:
        pre.pop('routed_dp_rank', None)
    start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(http, prefill['host'], prefill['port'], '/generate', key, pre, 600)
        result = stream(decode, key, payload)
        pre_result = future.result()
    if 'error' in pre_result:
        raise RuntimeError(str(pre_result['error']))
    elapsed = 1000*(result['started']-start)
    result['ttft_ms'] += elapsed
    result['e2e_ms'] += elapsed
    result['started'] = start
    result['prefill_meta'] = pre_result.get('meta_info', {})
    return result


def token_ids(graph, turn, namespace, body, cache):
    tokens = []
    for hash_id in turn['hash_ids']:
        index = (namespace, graph['id'], hash_id)
        if index not in cache:
            digest = hashlib.sha256(':'.join(map(str, index)).encode()).digest()
            marker = [1000 + int.from_bytes(digest[i:i+2], 'little') % 20000 for i in range(0,16,2)]
            cache[index] = (marker + body * 64)[:graph['block_size']]
        tokens.extend(cache[index])
    n = turn['input_tokens']
    # Trace hash IDs describe full blocks; a non-block-aligned tail stays unique
    # to its actual source turn. No input length is truncated.
    if len(tokens) < n:
        seed = f"{namespace}:{graph['id']}:{turn['source_outer_idx']}:{turn.get('source_inner_idx')}"
        rng = random.Random(hashlib.sha256(seed.encode()).digest())
        tokens += [1000+rng.randrange(20000) for _ in range(n-len(tokens))]
    tokens = tokens[:n]
    assert len(tokens) == n
    return tokens


def lcp(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


def pool_clone_lane(sid, lane, endpoints, family_width, copies):
    """Remove first-wave clone aliasing for non-power-of-two role pools.

    Replay first placements arrive in lexical copy blocks of family_width.
    For pools dividing that width, preserve V2 exactly. Otherwise choose a
    co-prime stride over observed first-placement blocks. Later child placements
    also use only their arrival index and retain the resulting cache owner.
    This balances cloned initial families; it is not a workload-cost predictor.
    """
    if copies <= 1 or not family_width:
        return lane
    period = len(endpoints) * math.lcm(*(e['ep'] for e in endpoints))
    if family_width % period == 0:
        return lane
    copy_id = int(sid.split(':copy', 1)[1].split(':', 1)[0])
    arrival_index = lane - copy_id
    assert arrival_index >= 0
    stride = next(s for s in range(1, period + 1)
                  if math.gcd(family_width + s, period) == 1)
    return arrival_index + (arrival_index // family_width) * stride


class Router:
    def __init__(self, endpoints, policy, family_width=None, copies=1):
        self.endpoints = endpoints
        self.family_width = family_width
        self.copies = copies
        self.policy = policy
        self.active = {e['id']: 0 for e in endpoints}
        self.pending_output = {e['id']: 0 for e in endpoints}
        self.affinity = {}
        self.history = {}
        self.tpot = {e['id']: (7.0 if e['ep'] > 1 else 8.0) for e in endpoints}

    def pick(self, sid, turn, lane):
        lane = pool_clone_lane(sid, lane, self.endpoints, self.family_width, self.copies)
        old = self.affinity.get(sid)
        if self.policy == 'sticky_hash':
            index = lane % len(self.endpoints)
            chosen = self.endpoints[index]
        else:
            def cost(e):
                rank = (lane // len(self.endpoints)) % e['ep']
                history = self.history.get((sid, e['id'], rank), [])
                cached = lcp(history, turn['hash_ids']) * 64
                fresh = max(0, turn['input_tokens'] - cached)
                if self.policy == 'least_outstanding':
                    return (self.active[e['id']], 0 if e['id'] == old else 1, e['id'])
                # Uses current requests and completed observations only. This
                # policy is intentionally simple and is compared with all fixed
                # pools; no future trace lengths or future routing statistics.
                n = self.active[e['id']] + 1
                batching = min(n, 8) ** .55
                wait = self.pending_output[e['id']] * self.tpot[e['id']] / batching
                cache_cost = fresh * .012
                stay_bonus = 0 if e['id'] == old else 20
                return (wait + cache_cost + stay_bonus, e['id'])
            chosen = min(self.endpoints, key=cost)
        switched = old is not None and old != chosen['id']
        self.affinity[sid] = chosen['id']
        self.active[chosen['id']] += 1
        self.pending_output[chosen['id']] += turn['output_tokens']
        return dict(chosen, rank=(lane // len(self.endpoints)) % chosen['ep'], clone_pool_lane=lane), switched

    def finish(self, sid, endpoint, turn, lane, result):
        name = endpoint['id']
        self.active[name] -= 1
        self.pending_output[name] -= turn['output_tokens']
        self.history[sid, name, endpoint.get('rank', lane % endpoint['ep'])] = turn['hash_ids']
        if result.get('tpot_ms') is not None:
            self.tpot[name] = .8*self.tpot[name] + .2*result['tpot_ms']




def assign_clone_balanced_lane(lanes, sid, namespace):
    """Rotate replicated families across logical cache owners, once per session.

    This removes the periodic alias between K selected roots and K GPU lanes.
    Only the replay copy number is used, never token counts or future work.
    The resulting owner remains fixed throughout the conversation. Copy zero
    keeps the original placement so unreplicated low-load results are intact.
    """
    if sid not in lanes:
        marker, copy = namespace.rsplit(':copy', 1)
        assert marker and copy.endswith(':')
        copy_id = int(copy[:-1])
        assert copy_id >= 0
        lanes[sid] = len(lanes) + copy_id
    return lanes[sid]


def replay(graphs, endpoints, policy, concurrency, key, body, health,
           namespace, idle_gap_cap_s=10, copies=1):
    # Every arm replays the same finite trees. Concurrency changes only admission.
    trees = [(g, f'{namespace}:copy{copy}:') for copy in range(copies) for g in graphs]
    queued = list(trees)
    pending = {}
    active_roots = set()
    remaining = {}
    complete = {}
    issued = {}
    completed_rows = []
    futures = {}
    blocks = {}
    lanes = {}
    prefill_eps = [e for e in endpoints if e['role'] in ('colocated', 'prefill')]
    decode_eps = [e for e in endpoints if e['role'] == 'decode']
    p_router = Router(prefill_eps, 'sticky_hash' if decode_eps else policy, len(graphs), copies)
    d_router = Router(decode_eps, policy, len(graphs), copies) if decode_eps else None
    start = time.perf_counter()
    shift = 0.0
    skipped_idle = 0.0
    deadline = start + 2400
    next_health = start
    with cf.ThreadPoolExecutor(max_workers=128) as pool:
        while queued or pending or futures:
            now = time.perf_counter()
            if now > deadline:
                raise TimeoutError('Finite AgentX episode exceeded 2400 seconds')
            if now >= next_health:
                health()
                next_health = now + 2
            while queued and len(active_roots) < concurrency:
                graph, prefix = queued.pop(0)
                vertices = build_vertices(graph, prefix)
                root = prefix + graph['id']
                active_roots.add(root)
                remaining[root] = len(vertices)
                for v in vertices.values():
                    v['admitted_at'] = now-start+shift
                pending.update(vertices)
            for future in list(futures):
                if not future.done():
                    continue
                vkey, vertex, p, d, lane, switched, ready_at = futures.pop(future)
                result = future.result()
                finish = result['finished']-start+shift
                complete[vkey] = finish
                p_router.finish(vertex['sid'], p, vertex['turn'], lane, result)
                if d_router:
                    d_router.finish(vertex['sid'], d, vertex['turn'], lane, result)
                result.update(root_id=vertex['root_id'], conversation_id=vertex['sid'], turn=vkey[1],
                              prefill_instance=p['id'], decode_instance=d['id'] if d else p['id'],
                              routed_dp_rank=(d or p)['rank'], routing_lane=lane, switched=switched,
                              prefill_pool_lane=p['clone_pool_lane'], decode_pool_lane=(d or p)['clone_pool_lane'],
                              source_outer_idx=vertex['turn']['source_outer_idx'],
                              source_inner_idx=vertex['turn'].get('source_inner_idx'),
                              source_kind=vertex['turn']['source_kind'],
                              input_tokens=vertex['turn']['input_tokens'],
                              ready_virtual_s=ready_at,
                              credit_issued_virtual_s=issued[vkey],
                              client_dispatch_lag_ms=max(0, 1000*(result['started']-start+shift-ready_at)))
                completed_rows.append(result)
                remaining[vertex['root_id']] -= 1
                if remaining[vertex['root_id']] == 0:
                    active_roots.remove(vertex['root_id'])
            virtual_now = time.perf_counter()-start+shift
            ready = []
            next_due = None
            for vkey, vertex in pending.items():
                due = due_time(vertex, complete, issued)
                if due is None:
                    continue
                if due <= virtual_now:
                    ready.append((vkey, vertex, due))
                else:
                    next_due = due if next_due is None else min(next_due, due)
            for vkey, vertex, due in sorted(ready, key=lambda x: (x[2], x[0])):
                lane = assign_clone_balanced_lane(lanes, vertex['sid'], vertex['namespace'])
                p, p_switch = p_router.pick(vertex['sid'], vertex['turn'], lane)
                d, d_switch = d_router.pick(vertex['sid'], vertex['turn'], lane) if d_router else (None, False)
                tokens = token_ids(vertex['graph'], vertex['turn'], vertex['namespace'], body, blocks)
                issued[vkey] = time.perf_counter()-start+shift
                future = pool.submit(request_one, p, d, key, tokens, vertex['turn']['output_tokens'], lane)
                futures[future] = (vkey, vertex, p, d, lane, p_switch or d_switch, due)
                del pending[vkey]
            if not futures and next_due is not None:
                gap = next_due-(time.perf_counter()-start+shift)
                if gap > idle_gap_cap_s:
                    delta = gap-idle_gap_cap_s
                    shift += delta
                    skipped_idle += delta
            if futures:
                cf.wait(futures, timeout=.01, return_when=cf.FIRST_COMPLETED)
            else:
                time.sleep(.01)
    wall = time.perf_counter()-start
    means = {metric: statistics.mean(r[metric] for r in completed_rows if r[metric] is not None)
             for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'client_dispatch_lag_ms')}
    slo = []
    for ttft in (200, 500, 1000, 2000):
        for tpot in (10, 15, 20, 30):
            passed = sum(r['ttft_ms'] <= ttft and (r['tpot_ms'] is None or r['tpot_ms'] <= tpot) for r in completed_rows)
            slo.append(dict(ttft_ms=ttft, tpot_ms=tpot, passed_requests=passed, goodput_rps=passed/wall))
    return dict(requests=completed_rows, means=means, wall_s=wall,
                output_tokens_per_s=sum(r['output_tokens'] for r in completed_rows)/wall,
                completed_requests=len(completed_rows), switched_requests=sum(r['switched'] for r in completed_rows),
                idle_compressed_s=skipped_idle, idle_gap_cap_s=idle_gap_cap_s, slo=slo,
                clone_lane_policy='pool_counterbalanced_copy_offset_v2',
                branch_timing_policy=BRANCH_POLICY,
                protocol='Pinned official branch issue/return offsets; finite-root admission remains custom; clone-copy cyclic offset with conversation affinity; full selected AgentX DAGs; exact source lengths; native GPU/host cache; global idle compression only; source hash-prefix synthetic input_ids; not official AgentX score.')
