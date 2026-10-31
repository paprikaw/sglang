"""SM80 Verify qualification. Synthetic Q/K/V; no model or expert compute."""
import importlib
import math
import statistics
import time


def run(case):
    import torch
    import triton
    ext = importlib.import_module('sglang.kernels.ops.attention.extend_attention')
    dec = importlib.import_module('sglang.kernels.ops.attention.decode_attention')
    split = importlib.import_module('sglang.kernels.ops.attention.verify_splitkv')
    b, p, qlen, h, hk, d = case['batch'], case['prefix'], case['new_tokens'], 24, 2, 128
    torch.manual_seed(20261023)
    dev = 'cuda'
    pages_per_req = triton.cdiv(p + qlen, 64)
    pages = torch.randperm(b * pages_per_req, device=dev)
    logical = (pages[:, None] * 64 + torch.arange(64, device=dev)[None, :]).reshape(b, -1)
    logical = logical[:, :p + qlen].contiguous()
    pool = b * pages_per_req * 64
    kb = torch.randn(pool, hk, d, device=dev, dtype=torch.bfloat16)
    vb = torch.randn_like(kb)
    qs = torch.randn(b * qlen, h, d, device=dev, dtype=torch.bfloat16)
    qo = torch.arange(b + 1, device=dev, dtype=torch.int32) * qlen
    ki = torch.arange(b + 1, device=dev, dtype=torch.int32) * (p + qlen)
    indices = logical.flatten()
    prefix_lens = torch.full((b,), p, device=dev, dtype=torch.int32)
    prefix_ptr = torch.arange(b + 1, device=dev, dtype=torch.int32) * p
    prefix_ids = logical[:, :p].reshape(-1).contiguous()
    new_ids = logical[:, p:].reshape(-1)
    ke, ve = kb[new_ids].contiguous(), vb[new_ids].contiguous()
    out = torch.empty_like(qs)
    mask_one = torch.arange(p + qlen, device=dev)[None, :] <= p + torch.arange(qlen, device=dev)[:, None]
    mask = mask_one.expand(b, -1, -1).reshape(-1).contiguous()
    mask_ptr = torch.arange(b + 1, device=dev, dtype=torch.int64) * qlen * (p + qlen)

    def unified():
        ext.extend_attention_fwd_unified(qs, out, kb, vb, 1., 1., qo, ki, indices,
                                        prefix_lens, qlen, custom_mask=mask,
                                        mask_indptr=mask_ptr, page_size=64)

    # Reuse the exact unified implementation and KV reduction block; only shrink
    # its query tile for this locally qualified short-query shape.
    original_config = ext._get_block_sizes_for_extend_attention
    original_shape = original_config(d, d)

    def smaller_config(lq, lv):
        values = list(original_config(lq, lv))
        values[3] = 16
        return tuple(values)

    def unified_small():
        ext._get_block_sizes_for_extend_attention = smaller_config
        try:
            unified()
        finally:
            ext._get_block_sizes_for_extend_attention = original_config

    # Treat each causal verification query as one Decode query. This deliberately
    # keeps Decode's request-local 256-token split size, independent of batch.
    # Metadata is prepared once here, not once per layer; measure its footprint
    # explicitly, rather than claiming these buffers are free in a live server.
    cpu_ids = logical.cpu().tolist()

    def decode_state(qtensor, sequences, context_cap=None):
        lengths = [len(x) for x in sequences]
        ptr = torch.tensor([0, *list(__import__('itertools').accumulate(lengths))], device=dev, dtype=torch.int32)
        ids = torch.tensor([x for seq in sequences for x in seq], device=dev, dtype=torch.int64)
        nsplit = torch.tensor([triton.cdiv(n, 256) for n in lengths], device=dev, dtype=torch.int32)
        maximum = triton.cdiv(context_cap, 256) if context_cap else max(triton.cdiv(n, 256) for n in lengths)
        scratch = torch.empty(qtensor.shape[0], h, maximum, d, device=dev, dtype=torch.float32)
        lse = torch.empty(qtensor.shape[0], h, maximum, device=dev, dtype=torch.float32)
        result = torch.empty_like(qtensor)

        def call():
            dec.decode_attention_fwd(qtensor, kb, vb, result, ptr, ids, scratch, lse,
                                     nsplit, maximum, d ** -.5, 1., 1., page_size=64)
        return call, result, ids.numel() * ids.element_size() + ptr.numel() * ptr.element_size()

    decode_call, decode_out, meta_bytes = decode_state(qs, [row[:p + i + 1] for row in cpu_ids for i in range(qlen)])
    fullcap_call, fullcap_out, _ = decode_state(qs, [row[:p + i + 1] for row in cpu_ids for i in range(qlen)], context_cap=131072)
    from verify_as_decode import CausalDecodeState
    state = CausalDecodeState(b, qlen, h, d, 131072, 256, dev)
    integrated_out = torch.empty_like(qs)
    def integrated_call():
        state.run(qs, integrated_out, kb, vb, prefix_ptr, prefix_ids, new_ids, d ** -.5, 64)
    oneq = qs.reshape(b, qlen, h, d)[:, 0].contiguous()
    one_call, one_out, _ = decode_state(oneq, [row[:p + 1] for row in cpu_ids])
    references = []
    for request in range(b):
        k = kb[logical[request]].repeat_interleave(h // hk, dim=1).float()
        v = vb[logical[request]].repeat_interleave(h // hk, dim=1).float()
        query = qs[request * qlen:(request + 1) * qlen].float()
        scores = torch.einsum('qhd,khd->hqk', query, k) * d ** -.5
        scores.masked_fill_(~mask_one[None, :, :], float('-inf'))
        references.append(torch.einsum('hqk,khd->qhd', scores.softmax(-1), v))
    reference = torch.cat(references).to(torch.bfloat16)
    unified(); torch.cuda.synchronize()
    baseline = out.clone()

    def error(value, target):
        delta = (value.float() - target.float()).abs()
        return dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()), exact=bool(torch.equal(value, target)))

    def check(value):
        torch.testing.assert_close(value, reference, atol=2e-3, rtol=2e-2)
        return dict(fp32_reference=error(value, reference), original_unified=error(value, baseline))

    def measure(call):
        for _ in range(4): call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): call()
        graph.replay(); torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(32): graph.replay()
            end.record(); end.synchronize()
            samples.append(start.elapsed_time(end) / 32)
        return dict(graph_ms=statistics.median(samples), samples_ms=samples,
                    repetitions=3, graph_replays_per_sample=32)

    rows = {}
    for name, call, result in (('unified_original', unified, out),
                               ('unified_m16', unified_small, out),
                               ('causal_parallel_decode', decode_call, decode_out),
                               ('parallel_decode_131k_cap', fullcap_call, fullcap_out),
                               ('integrated_metadata_131k_cap', integrated_call, integrated_out)):
        call(); torch.cuda.synchronize()
        rows[name] = dict(passed=True, correctness=check(result), **measure(call))

    # CUDA has a minimum tensor-core query tile. Keep the upstream implementation
    # untouched, qualify a derived class with M>=16 for this NVIDIA-only test.
    vk = split.VerifySplitKV(b, h, hk, d, d, qlen, device=dev,
                             n_splits=split.choose_n_splits(p), block_n=32, num_warps=4)
    vk.l_pad = max(16, vk.l_pad)
    vk._alloc(b)

    def verify_split():
        vk(qs, ke, ve, kb, vb, qo, prefix_ptr, prefix_ids, d ** -.5, o_out=out)

    try:
        verify_split(); torch.cuda.synchronize()
        rows['upstream_splitkv_m16'] = dict(passed=True, correctness=check(out), **measure(verify_split))
    except Exception as exc:
        rows['upstream_splitkv_m16'] = dict(passed=False, error=f'{type(exc).__name__}: {exc}')

    # Compare the batched causal Decode result to B*Q independent Decode calls.
    independent = []
    for request in range(b):
        for token in range(qlen):
            index = request * qlen + token
            call, value, _ = decode_state(qs[index:index + 1], [cpu_ids[request][:p + token + 1]])
            call(); independent.append(value)
    torch.cuda.synchronize()
    independent = torch.cat(independent)
    invariance = error(decode_out, independent)
    if not invariance['exact']:
        raise AssertionError(f'Parallel causal Decode changed independent query outputs: {invariance}')
    torch.testing.assert_close(integrated_out, decode_out, atol=0, rtol=0)
    # One captured graph must consume changed, ragged prefix metadata, including
    # an empty prefix, without stale Python-object or cache-slot assumptions.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): integrated_call()
    ragged = [p, p // 2, p // 4, 0]
    original_prefix_ids, original_prefix_ptr = prefix_ids.clone(), prefix_ptr.clone()
    packed = [value for row, length in zip(cpu_ids, ragged) for value in row[:length]]
    ptr = [0, *list(__import__('itertools').accumulate(ragged))]
    prefix_ids[:len(packed)].copy_(torch.tensor(packed, device=dev, dtype=prefix_ids.dtype))
    prefix_ptr.copy_(torch.tensor(ptr, device=dev, dtype=prefix_ptr.dtype))
    ragged_sequences = [row[:length] + row[p:p + i + 1]
                        for row, length in zip(cpu_ids, ragged) for i in range(qlen)]
    ragged_call, ragged_out, _ = decode_state(qs, ragged_sequences, context_cap=131072)
    ragged_call(); graph.replay(); torch.cuda.synchronize()
    ragged_error = error(integrated_out, ragged_out)
    if not ragged_error['exact']:raise AssertionError('Graph replay used stale/ragged/wrong physical KV: ' + str(ragged_error))
    prefix_ids.copy_(original_prefix_ids);prefix_ptr.copy_(original_prefix_ptr)
    graph.replay(); torch.cuda.synchronize()
    restored_error = error(integrated_out, decode_out)
    if not restored_error['exact']:raise AssertionError('Graph replay failed after restoring prefix')
    one_full_call, one_full_out, _ = decode_state(oneq, [row[:p + 1] for row in cpu_ids], context_cap=131072)
    one_call(); torch.cuda.synchronize()
    return dict(case=case, status='passed', model_loaded=False, batch_size=b,
                num_heads=h, head_dim=d, page_size=64, num_kv_heads=hk,
                original_tile=dict(zip(('D', 'DPE', 'DV', 'M', 'N', 'warps'), original_shape)),
                candidates=rows, independent_decode_bitwise=invariance,
                normal_q1_decode=measure(one_call), normal_q1_decode_131k_cap=measure(one_full_call),
                graph_ragged_prefix_replay=ragged_error, graph_restored_prefix_replay=restored_error,
                expanded_metadata_bytes=meta_bytes,
                note='Kernel Graph timing, not full-model TPOT. Expanded index preparation excluded; integration must build once per Verify step, not per layer.')
