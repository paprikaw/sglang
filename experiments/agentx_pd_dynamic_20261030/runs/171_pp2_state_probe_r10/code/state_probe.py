"""Intrusive correctness diagnostic: fingerprint native P/D state by room.

GPU fences and CPU copies are deliberate. These runs are never performance
samples. Read native state after final Prefill and after Decode transfer commit.
"""
from functools import wraps

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import hashlib
    import json
    import os
    from pathlib import Path
    import time
    import torch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.disaggregation.decode import DecodeTransferQueue

    def digest(tensor):
        host = tensor.detach().contiguous().cpu()
        return {'shape': list(host.shape), 'dtype': str(host.dtype),
                'sha256': hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'finite': bool(torch.isfinite(host).all())}

    def capture(scheduler, req, point):
        length = len(req.origin_input_ids)
        if length not in (4096, 4097):
            return
        torch.cuda.synchronize()
        pool = scheduler.req_to_token_pool
        slot = int(pool.translate_mamba_indices(pool.req_index_to_mamba_index_mapping[req.req_pool_idx]).item())
        states = pool.mamba_pool.mamba_cache
        mamba = {}
        for global_layer, local_layer in sorted(pool.mamba_map.items()):
            mamba[str(global_layer)] = {
                'temporal': digest(states.temporal[local_layer, slot]),
                'conv': [digest(c[local_layer, slot]) for c in states.conv],
            }
        cache = scheduler.token_to_kv_pool_allocator.get_kvcache()
        indices = pool.req_to_token[req.req_pool_idx, :length].to(torch.long)
        kv = {}
        # Access the underlying dense pool directly: diagnostic CUDA fence above
        # completed local streams, and this must not alter HiCache consumers.
        for global_layer, local_layer in sorted(cache.full_attention_layer_id_mapping.items()):
            kv[str(global_layer)] = {
                'key': digest(cache.full_kv_pool.get_key_buffer(local_layer).index_select(0, indices)),
                'value': digest(cache.full_kv_pool.get_value_buffer(local_layer).index_select(0, indices)),
            }
        rows = getattr(scheduler, '_agentx_state_snapshots', [])
        rows.append({'time': time.time(), 'point': point, 'bootstrap_room': str(req.bootstrap_room),
                     'input_tokens': length, 'pp_rank': scheduler.tree_cache.pp_rank,
                     'request_pool_idx': int(req.req_pool_idx), 'mamba_slot': slot,
                     'mamba': mamba, 'kv': kv, 'performance_sample': False})
        scheduler._agentx_state_snapshots = rows

    old_send = Scheduler.send_kv_chunk

    @wraps(old_send)
    def send(self, req, last_chunk=False, end_idx=None):
        if last_chunk:
            capture(self, req, 'prefill_before_final_send')
        return old_send(self, req, last_chunk=last_chunk, end_idx=end_idx)

    old_commit = DecodeTransferQueue._commit_transfer_to_req

    @wraps(old_commit)
    def commit(self, decode_req):
        result = old_commit(self, decode_req)
        capture(self.scheduler, decode_req.req, 'decode_after_transfer_commit')
        return result

    old_get = Scheduler.get_internal_state

    @wraps(old_get)
    def get(self, *args, **kwargs):
        result = old_get(self, *args, **kwargs)
        rows = getattr(self, '_agentx_state_snapshots', [])
        path = Path(os.environ['EP2_PP2_RUN']) / f'state_probe.pp{self.tree_cache.pp_rank}.tp{self.ps.tp_rank}.json'
        path.write_text(json.dumps(rows))
        result.internal_state['agentx_state_snapshot_count'] = len(rows)
        return result

    Scheduler.send_kv_chunk = send
    DecodeTransferQueue._commit_transfer_to_req = commit
    Scheduler.get_internal_state = get
    _INSTALLED = True
