import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import aiohttp
from aiohttp import web
import pytest

spec=importlib.util.spec_from_file_location('reactive_lifecycle',Path(__file__).with_name('lifecycle.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_control_endpoint_survives_workload_exit_until_controller_releases(tmp_path):
    async def check():
        app=web.Application();app.router.add_get('/state',lambda _:web.json_response(dict(active=0)))
        server=web.AppRunner(app);await server.setup()
        site=web.TCPSite(server,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        task=asyncio.create_task(m.hold_control_after_workload(tmp_path,0,timeout_s=2))
        try:
            while not (tmp_path/'replay_workload_done.json').exists():await asyncio.sleep(.001)
            assert not task.done()
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://127.0.0.1:{port}/state') as reply:
                    assert reply.status==200 and await reply.json()==dict(active=0)
            (tmp_path/'release_router').touch()
            await asyncio.wait_for(task,1)
        finally:
            if not task.done():task.cancel()
            await server.cleanup()
    asyncio.run(check())


def test_failed_workload_does_not_wait_for_controller_release(tmp_path):
    asyncio.run(m.hold_control_after_workload(tmp_path,7))
    assert json.loads((tmp_path/'replay_workload_done.json').read_text())['exit_code']==7


def test_router_hold_has_a_bounded_lifetime(tmp_path):
    with pytest.raises(TimeoutError,match='controller'):
        asyncio.run(m.hold_control_after_workload(tmp_path,0,timeout_s=0))


def test_guard_cancellation_interrupts_router_hold(tmp_path):
    async def check():
        task=asyncio.create_task(m.hold_control_after_workload(tmp_path,0))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
    asyncio.run(check())


def finite_run(tmp_path, *, missing_last=False, cancelled=False, drained=True):
    (tmp_path/'config.json').write_text(json.dumps(dict(expected_profile_requests=2, expected_tasks=1)))
    directory=tmp_path/'replay'
    (directory/'agentx').mkdir(parents=True)
    (directory/'events').mkdir()
    records=[dict(metadata=dict(benchmark_phase='profiling', x_request_id=str(i), was_cancelled=cancelled))
             for i in range(1 if missing_last else 2)]
    (directory/'agentx/profile_export.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    events=[dict(event='tree_drained')]
    if drained:events.append(dict(event='finite_stream_drained'))
    (directory/'events/client.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
    (directory/'routing.jsonl').write_text(''.join(json.dumps(dict(x_request_id=str(i),success=True))+'\n' for i in range(2)))


def test_finite_replay_requires_all_successful_requests_and_native_drain(tmp_path):
    finite_run(tmp_path)
    assert m.validate_finite_replay(tmp_path)['passed']


@pytest.mark.parametrize('change',[dict(missing_last=True),dict(cancelled=True),dict(drained=False)])
def test_zero_harness_exit_does_not_qualify_a_partial_replay(tmp_path,change):
    finite_run(tmp_path,**change)
    with pytest.raises(RuntimeError,match='replay incomplete'):
        asyncio.run(m.hold_control_after_workload(tmp_path,0,timeout_s=0))
    assert json.loads((tmp_path/'replay_workload_done.json').read_text())['exit_code']==1
    assert not json.loads((tmp_path/'replay_completion_audit.json').read_text())['passed']


def test_validation_waits_for_buffered_audit_close_without_closing_control(tmp_path):
    async def check():
        finite_run(tmp_path)
        path=tmp_path/'replay/routing.jsonl'
        rows=path.read_text().splitlines();path.write_text('')
        queue=asyncio.Queue();written=asyncio.Event()
        async def writer():
            # The real adapter also uses a buffered file through aiofiles.
            with path.open('a',buffering=1024*1024) as output:
                for row in rows:output.write(row+'\n')
                written.set()
                assert await queue.get() is None
        task=asyncio.create_task(writer());await written.wait()
        assert path.stat().st_size==0
        with pytest.raises(RuntimeError,match='incomplete'):
            m.validate_finite_replay(tmp_path)
        adapter=NS(active={},reactive=NS(active={'d0':0},waiting={}),audit_queue=queue,audit_task=task)
        (tmp_path/'release_router').touch()
        await m.hold_control_after_workload(tmp_path,0,adapter=adapter)
        assert adapter.audit_task is None and task.done()
        assert json.loads((tmp_path/'replay_completion_audit.json').read_text())['passed']
    asyncio.run(check())


def test_audit_flush_waits_for_the_last_http_handler(tmp_path):
    async def check():
        queue=asyncio.Queue();closed=asyncio.Event()
        async def writer():
            assert await queue.get()=='last row'
            assert await queue.get() is None
            closed.set()
        adapter=NS(active={'last':1},reactive=NS(active={'d0':1},waiting={}),audit_queue=queue,
                   audit_task=asyncio.create_task(writer()))
        flushing=asyncio.create_task(m.flush_routing_audit(adapter))
        await asyncio.sleep(.01);assert not closed.is_set()
        queue.put_nowait('last row');adapter.active.clear();adapter.reactive.active['d0']=0
        await flushing
        assert closed.is_set()
    asyncio.run(check())
