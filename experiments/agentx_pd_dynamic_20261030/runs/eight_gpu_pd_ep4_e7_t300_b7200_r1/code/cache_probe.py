"""Private diagnostic hook: native HiCache eviction and restore evidence.

Only an explicit idle-only admin command demotes GPU prefixes. Formal workload
replay never invokes it. Memory budgets and native cache policy are unchanged.
"""
from functools import wraps
import time

_INSTALLED = False


def qualify(base, run, cfg, arm, key, one, smoke, reference, health):
    """Demote both roles once, then verify exact output plus an actual D loadback."""
    evidence = {'formal_performance_sample': False, 'demotion': [], 'responses': []}
    for replica in range(base.layout(cfg, arm)['replicas']):
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
        evidence['demotion'].append([s.get('agentx_cache_probe') for s in info['internal_states']])
    for index in (4, 5):
        response = one(cfg, arm, key, smoke[index], 32, index)
        evidence['responses'].append(response)
        if response['output_ids'] != reference[index]:
            base.h.save(run / 'evidence' / arm / 'host_restore_qualification.json', evidence)
            raise RuntimeError('Output changed after actual CPU HiCache restore')
    info = base.h.http_json(base.port(cfg, arm, 1), '/get_server_info', key, timeout=30)
    events = [s.get('agentx_restores', []) for s in info['internal_states']]
    evidence['decode_loadback_events'] = events
    evidence['passed'] = any(e['loaded_tokens'] > 0 and not e['copy_mamba_to_request']
                             for rank in events for e in rank)
    base.h.save(run / 'evidence' / arm / 'host_restore_qualification.json', evidence)
    if not evidence['passed']:
        raise RuntimeError('Host test did not exercise Decode local restore')


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    old_set = Scheduler.set_internal_state
    old_get = Scheduler.get_internal_state
    old_load = UnifiedRadixCache.init_load_back

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
        result.internal_state['agentx_cache_probe'] = getattr(self, '_agentx_cache_probe', None)
        result.internal_state['agentx_restores'] = getattr(self.tree_cache, '_agentx_restores', [])
        return result

    Scheduler.set_internal_state = set_state
    Scheduler.get_internal_state = get_state
    UnifiedRadixCache.init_load_back = load
    _INSTALLED = True
