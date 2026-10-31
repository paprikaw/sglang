"""Private diagnostic: synchronize Prefill host-restore completion before forward.

Only an explicit idle-only admin command demotes GPU prefixes. Formal workload
replay never invokes it. Memory budgets and native cache policy are unchanged.
"""
from functools import wraps
import time

_INSTALLED = False


def qualify(base, run, cfg, arm, key, one, smoke, reference, health):
    """Isolate P, D, and combined native host restore; preserve all outcomes."""
    evidence = {'formal_performance_sample': False, 'diagnostic': 'Prefill H2D finish-event CPU fence; not a production performance sample', 'cases': []}
    cases = [(f'{label}_r{repeat+1}', demote) for repeat in range(3)
             for label, demote in [('both_first', [0, 1]), ('prefill_only', [0]), ('decode_only', [1]), ('both_last', [0, 1])]]
    for label, demote in cases:
        base.flush(cfg, arm, key)
        warm = [one(cfg, arm, key, t, 32, i) for i, t in enumerate(smoke)]
        cold_match = [r['output_ids'] for r in warm] == reference
        case = {'name': label, 'warm_matches_reference': cold_match,
                'demotion': [], 'responses': [], 'matches': []}
        evidence['cases'].append(case)
        for replica in demote:
            deadline = time.monotonic() + 45
            while True:
                health()
                answer = base.h.http_json(base.port(cfg, arm, replica), '/set_internal_state', key,
                    {'server_args': {'agentx_cache_cmd': 2}}, 60)
                if isinstance(answer, list) and answer and all(v is True for v in answer):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('Idle GPU demotion was not acknowledged')
                time.sleep(.2)
            info = base.h.http_json(base.port(cfg, arm, replica), '/get_server_info', key, timeout=30)
            case['demotion'].append({'replica': replica, 'states': info['internal_states']})
        for index in (4, 5):
            response = one(cfg, arm, key, smoke[index], 32, index)
            case['responses'].append(response)
            case['matches'].append(response['output_ids'] == reference[index])
            base.h.save(run / 'evidence' / arm / 'host_restore_isolation.json', evidence)
        case['after'] = []
        for replica in (0, 1):
            info = base.h.http_json(base.port(cfg, arm, replica), '/get_server_info', key, timeout=30)
            case['after'].append({'replica': replica, 'states': info['internal_states']})
    events = [e for state in evidence['cases'][-1]['after'][1]['states'] for e in state.get('agentx_restores', [])]
    evidence['positive_decode_restores'] = sum(e['loaded_tokens'] > 0 and not e['copy_mamba_to_request'] for e in events)
    stage_proofs = []
    for replica in (0, 1):
        for pp in (0, 1):
            root = run / 'evidence' / f'{arm}.r{replica}'
            proof = __import__('json').loads((root / f'prefill_fence.pp{pp}.json').read_text())
            restores = __import__('json').loads((root / f'cache_restore.pp{pp}.json').read_text())
            stage_proofs.append(dict(replica=replica, pp_rank=pp, fences=proof['fences'],
                positive_decode_restores=sum(e['loaded_tokens'] > 0 and not e['copy_mamba_to_request'] for e in restores)))
    evidence['stage_proofs'] = stage_proofs
    evidence['fence_applied'] = all(p['fences'] >= 9 for p in stage_proofs if p['replica'] == 0)
    evidence['both_decode_stages_restored'] = all(p['positive_decode_restores'] >= 9 for p in stage_proofs if p['replica'] == 1)
    evidence['passed'] = all(c['warm_matches_reference'] and all(c['matches']) for c in evidence['cases']) and evidence['fence_applied'] and evidence['both_decode_stages_restored']
    base.h.save(run / 'evidence' / arm / 'host_restore_isolation.json', evidence)
    if not evidence['passed']:
        raise RuntimeError('Isolated host restore output mismatch; per-role evidence saved')


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import os
    import json
    from pathlib import Path
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    old_set = Scheduler.set_internal_state
    old_get = Scheduler.get_internal_state
    old_load = UnifiedRadixCache.init_load_back
    old_start = UnifiedRadixCache.ready_to_load_host_cache

    @wraps(old_start)
    def start_loading(self):
        from sglang.srt.runtime_context import get_disagg
        from sglang.srt.disaggregation.utils import DisaggregationMode
        index = old_start(self)
        if index >= 0 and DisaggregationMode(get_disagg().disaggregation_mode) == DisaggregationMode.PREFILL:
            self.cache_controller.layer_done_counter.events[index].finish_event.synchronize()
            self._agentx_h2d_fences = getattr(self, '_agentx_h2d_fences', 0) + 1
        return index

    UnifiedRadixCache.ready_to_load_host_cache = start_loading

    @wraps(old_load)
    def load(self, params):
        result = old_load(self, params)
        events = getattr(self, '_agentx_restores', [])
        events.append({'time': time.time(), 'copy_mamba_to_request': params.copy_mamba_to_request,
                       'loaded_tokens': len(result[0]) if result is not None else 0})
        self._agentx_restores = events[-128:]
        return result

    @wraps(old_set)
    def set_state(self, req):
        if set(req.server_args) != {'agentx_cache_cmd'}:
            return old_set(self, req)
        cmd = int(req.server_args['agentx_cache_cmd'])
        if cmd not in (1, 2):
            raise ValueError('Unknown cache diagnostic command')
        if not self.is_fully_idle():
            return SetInternalStateReqOutput(updated=False)
        cache = self.tree_cache
        if not isinstance(cache, UnifiedRadixCache) or cache.cache_controller is None:
            raise RuntimeError('Hybrid HiCache is required')
        cache.writing_check(write_back=True)
        before = {'full': cache.full_evictable_size(), 'mamba': cache.mamba_evictable_size()}
        if cmd == 2:
            cache.evict(EvictParams(num_tokens=1 << 50, mamba_num=1 << 30))
        after = {'full': cache.full_evictable_size(), 'mamba': cache.mamba_evictable_size()}
        self._agentx_cache_probe = {'time': time.time(), 'command': cmd, 'before': before,
                                   'after': after, 'ongoing_writes': len(cache.ongoing_write_through)}
        return SetInternalStateReqOutput(updated=True)

    @wraps(old_get)
    def get_state(self, *args, **kwargs):
        result = old_get(self, *args, **kwargs)
        result.internal_state['agentx_prefill_h2d_fences'] = getattr(self.tree_cache, '_agentx_h2d_fences', 0)
        result.internal_state['agentx_cache_probe'] = getattr(self, '_agentx_cache_probe', None)
        result.internal_state['agentx_restores'] = getattr(self.tree_cache, '_agentx_restores', [])
        path = Path(os.environ['EP2_PP2_RUN']) / f'cache_restore.pp{self.tree_cache.pp_rank}.json'
        path.write_text(json.dumps(getattr(self.tree_cache, '_agentx_restores', [])))
        (path.parent / f'prefill_fence.pp{self.tree_cache.pp_rank}.json').write_text(json.dumps({'pp_rank': self.tree_cache.pp_rank, 'fences': getattr(self.tree_cache, '_agentx_h2d_fences', 0), 'performance_sample': False}))
        return result

    Scheduler.set_internal_state = set_state
    Scheduler.get_internal_state = get_state
    UnifiedRadixCache.init_load_back = load
    _INSTALLED = True
