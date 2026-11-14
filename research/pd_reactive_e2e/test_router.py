import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

spec=importlib.util.spec_from_file_location('reactive_router',Path(__file__).with_name('router.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_admission_preserves_available_affinity_but_uses_idle_new_replica():
    p=m.Admission([NS(name='d0'),NS(name='d1')],1)
    assert p.select('a').name=='d0'
    p.enabled.add('d1')
    assert p.select('a').name=='d1'
    assert p.moves==1
    assert p.select('b') is None

def test_closed_gate_defers_new_handoff_and_measures_wait():
    async def check():
        p=m.Admission([NS(name='d0')],1);p.paused=True
        waiting=asyncio.create_task(p.acquire('a'))
        await asyncio.sleep(.025)
        assert not waiting.done() and p.state()['waiting']==1 and p.active['d0']==0
        async with p.condition:
            p.paused=False;p.condition.notify_all()
        endpoint,wait=await asyncio.wait_for(waiting,.5)
        assert endpoint.name=='d0' and wait>=.02 and not p.waiting
        await p.release('d0')
        with pytest.raises(RuntimeError):await p.release('d0')
    asyncio.run(check())

def test_cancelled_waiter_does_not_leak_admission():
    async def check():
        p=m.Admission([NS(name='d0')],1);p.select('a')
        waiting=asyncio.create_task(p.acquire('b'));await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):await waiting
        assert not p.waiting and p.active=={'d0':1}
    asyncio.run(check())

def test_router_wait_is_in_latency_but_not_in_decode_tpot():
    import json
    p=m.Admission([NS(name='d0')]);q=m.AuditQueue(p)
    token=p.context.set(dict(admission_wait_s=3,started_unix_ns=100,endpoint=NS(name='d0'),enabled=['d0']))
    q.put_nowait(json.dumps(dict(elapsed_s=10,first_content_s=2,started_unix_ns=999)))
    row=json.loads(q.get_nowait())
    assert row['elapsed_s']==13 and row['first_content_s']==5
    assert row['elapsed_s']-row['first_content_s']==8 and row['started_unix_ns']==100
    p.context.reset(token)


def test_cache_affinity_distinguishes_cloned_sessions_but_survives_turn_changes():
    spec=importlib.util.spec_from_file_location('session_placement',Path(__file__).with_name('stable_placement.py'))
    placement=importlib.util.module_from_spec(spec);spec.loader.exec_module(placement)
    a=NS(conversation_id='same-source-conversation',x_correlation_id='clone-a',turn_index=1)
    b=NS(conversation_id='same-source-conversation',x_correlation_id='clone-b',turn_index=1)
    first=placement.placement_key(a);a.turn_index=7
    assert first==placement.placement_key(a) and first!=placement.placement_key(b)


def test_deterministic_prefill_chunk_must_be_large_enough_to_admit_long_requests():
    spec=importlib.util.spec_from_file_location('launch_checks',Path(__file__).with_name('launch_checks.py'))
    checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
    args=NS(enable_deterministic_inference=True,attention_backend='triton',chunked_prefill_size=512)
    with pytest.raises(ValueError,match='4096'):checks.validate_prefill_alignment(args)
    args.chunked_prefill_size=4096;checks.validate_prefill_alignment(args)
