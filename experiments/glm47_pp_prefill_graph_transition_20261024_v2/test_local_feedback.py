"""CPU checks exercise the *pinned real* exchange body, with queue transport.

No CUDA/model load. This covers protocol ancestry, startup and draining, not
NCCL performance. GPU transport and exact real-model outputs are separate gates.
"""
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace as NS

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent / 'runtime'))
from pp_local_feedback import TailFeedbackSlots, attach, receipt

ENGINE = Path(__file__).resolve().parents[2] / '.engine-worktrees/glm47-pp-spec-microbatch-invariant-20261023'


def real_exchange_class():
    source = ENGINE / 'python/sglang/srt/managers/scheduler_pp_mixin.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SchedulerPPMixin')
    names = {'_pp_send_output_to_next_stage', '_pp_send_recv_and_preprocess_output_tensors'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in methods:
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
    # Future annotations also covers annotated local assignments in the body.
    code = 'from __future__ import annotations\n' + ast.unparse(ast.Module(body=methods, type_ignores=[]))
    ns = dict(torch=NS(profiler=NS(record_function=lambda _: nullcontext())),
              is_xpu=lambda: False, PPProxyTensors=lambda data: NS(tensors=data),
              _pp_can_skip_output_comm=lambda batch: batch.skip)
    exec(compile(code, str(source), 'exec'), ns)
    return type('PinnedExchange', (), {key: ns[key] for key in names})


def test_slot_ownership_and_ancestry_fail_closed():
    slots = TailFeedbackSlots()
    payload = object()
    slots.put(0, 'event', payload, ('a', 'b'))
    with pytest.raises(RuntimeError, match='overwritten'):
        slots.put(0, 'new', object(), ('a', 'b'))
    with pytest.raises(RuntimeError, match='ancestry'):
        slots.pop(0, ('b', 'a'))
    assert 0 in slots.pending
    assert slots.pop(0, ('a', 'b')).payload is payload
    with pytest.raises(RuntimeError, match='without a producer'):
        slots.pop(0, ('a', 'b'))


@pytest.mark.parametrize('lengths', [(3, 3), (1, 4), (4, 1), (1, 0), (0, 1)])
def test_real_exchange_two_slots_no_echo_and_independent_drain(lengths):
    mixin = real_exchange_class()
    mixin.init_pp_loop_state = lambda self: None
    mixin._pp_recv_dict_from_prev_stage = lambda self: feedback.get(timeout=3)
    mixin._pp_requires_immediate_output_relay = lambda self: False
    attach(mixin, can_skip=lambda batch: batch.skip,
           current_stream=lambda self: NS(wait_event=lambda event: None))
    proxies, feedback = Queue(), Queue()
    observed = []
    launches = {slot: lengths[slot] for slot in (0, 1)}
    steps = max(lengths) * 2 + 1
    seqs = [{0: 100, 1: 100}, {0: 100, 1: 100}]
    peers = []

    for rank in (0, 1):
        peer = mixin()
        peer.ps = NS(pp_rank=rank, pp_size=2)
        peer.pp_loop_size = 2
        peer._pp_spec_microbatch = peer._pp_spec_relay = True
        peer.pp_group = NS(is_last_rank=rank == 1)
        peer.copy_stream_ctx = nullcontext()
        peer.copy_stream = NS(wait_stream=lambda _: None)
        peer.schedule_stream = None
        peer.device_module = NS(Event=lambda: NS(record=lambda _: None), current_stream=lambda: None)
        peer.mb_metadata = [None, None]
        peer._pp_send_dict_to_next_stage = lambda data, **kw: feedback.put(data) or []
        peer.init_pp_loop_state()
        peers.append(peer)

    def run(rank):
        peer = peers[rank]
        mbs, queue, previous = [None, None], deque(), None

        def prep(target, metadata, packet):
            slot, round_id = target.slot, target.round
            data = packet.tensors
            assert data['slot'] == slot and data['round'] == round_id
            assert data['rids'] == tuple(req.rid for req in target.reqs)
            assert data['new_seq'] == 100 + 2 * (round_id + 1)
            seqs[rank][slot] = data['new_seq']
            observed.append(('retire', rank, slot, round_id))
            return NS()
        peer._pp_prep_batch_result = prep
        for step in range(steps):
            slot, round_id = step % 2, step // 2
            exists = round_id < launches[slot]
            if exists:
                batch = NS(slot=slot, round=round_id, skip=False,
                           forward_mode=NS(is_prebuilt=lambda: False),
                           reqs=[NS(rid=f'slot{slot}-r{i}') for i in range(4)])
                assert seqs[rank][slot] == 100 + 2 * round_id
                if rank == 0:
                    proxies.put((slot, round_id))
                else:
                    assert proxies.get(timeout=3) == (slot, round_id)
                    payload = NS(tensors=dict(slot=slot, round=round_id,
                        rids=tuple(req.rid for req in batch.reqs), new_seq=seqs[rank][slot] + 2))
                    queue.append((object(), payload))
                observed.append(('launch', rank, slot, round_id))
                mbs[slot] = batch
                peer.mb_metadata[slot] = NS(fwd_batch=batch)
            else:
                mbs[slot] = None
                peer.mb_metadata[slot] = None
            previous, _, _, _ = peer._pp_send_recv_and_preprocess_output_tensors(
                slot, 1 - slot, mbs, peer.mb_metadata, queue, previous)
        assert not queue

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run, rank) for rank in (0, 1)]
        for job in jobs:
            job.result(timeout=10)
    for rank in (0, 1):
        assert seqs[rank] == {slot: 100 + 2 * lengths[slot] for slot in (0, 1)}
    assert proxies.empty() and feedback.empty()
    tail = receipt(peers[1]); head = receipt(peers[0])
    assert tail['produced'] == tail['retired'] == sum(lengths)
    assert head['head_feedback_receives'] == sum(lengths)
    assert head['echo_sends_skipped'] >= sum(lengths) - 1


def test_unsupported_scheduler_fails_before_protocol_change():
    mixin = real_exchange_class()
    mixin.init_pp_loop_state = lambda self: None
    mixin._pp_recv_dict_from_prev_stage = lambda self: None
    attach(mixin, can_skip=lambda _: False, current_stream=lambda _: None)
    peer = mixin()
    peer._pp_spec_microbatch = False
    with pytest.raises(RuntimeError, match='qualified PP2'):
        peer.init_pp_loop_state()
