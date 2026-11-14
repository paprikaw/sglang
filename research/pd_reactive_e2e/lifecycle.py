"""Keep the router alive while the controller finishes its scale-down."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time


def validate_finite_replay(run: Path):
    """A zero harness exit can still mean a timed-out, partially sent DAG."""
    config = json.loads((run/'config.json').read_text())
    def records(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    formal = [r for r in records(run/'replay/agentx/profile_export.jsonl')
              if r['metadata'].get('benchmark_phase') == 'profiling']
    events = [r for path in (run/'replay/events').glob('*.jsonl') for r in records(path)]
    routes = {r['x_request_id']: r for r in records(run/'replay/routing.jsonl')}
    good = [r for r in formal if routes.get(r['metadata']['x_request_id'], {}).get('success')
            and not r['metadata'].get('was_cancelled')
            and not r['metadata'].get('context_overflow_skip')]
    tasks = sum(r['event'] == 'tree_drained' for r in events)
    drained = sum(r['event'] == 'finite_stream_drained' for r in events)
    result = dict(requests=len(formal), successful=len(good), tasks=tasks,
                  finite_stream_drained=drained,
                  expected_requests=config['expected_profile_requests'],
                  expected_tasks=config['expected_tasks'])
    result['passed'] = (result['requests'] == result['successful'] == result['expected_requests']
                        and tasks == result['expected_tasks'] and drained == 1)
    (run/'replay_completion_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise RuntimeError('finite AgentX replay incomplete: '+json.dumps(result))
    return result


async def flush_routing_audit(adapter, timeout_s=30):
    """Close only the audit writer, keeping the HTTP control plane alive."""
    async with asyncio.timeout(timeout_s):
        # Native request accounting can finish just before a handler's finally
        # block queues its audit row. Wait for those handlers before the sentinel.
        while adapter.active or any(adapter.reactive.active.values()) or adapter.reactive.waiting:
            await asyncio.sleep(.01)
        if adapter.audit_task is not None:
            adapter.audit_queue.put_nowait(None)
            await asyncio.shield(adapter.audit_task)
            adapter.audit_task = None  # Native stop must not send a second sentinel.


async def hold_control_after_workload(run: Path, exit_code: int, timeout_s=480, *, adapter=None):
    def publish(code, error=None):
        done = run / 'replay_workload_done.json'
        temporary = done.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(exit_code=code, unix=time.time(), error=error))+'\n')
        temporary.replace(done)
    try:
        if not exit_code and adapter is not None:
            await flush_routing_audit(adapter)
        if not exit_code and (run/'config.json').exists():
            validate_finite_replay(run)
    except Exception as error:
        publish(1,repr(error))
        raise
    publish(exit_code)
    if exit_code:
        return
    deadline = time.monotonic()+timeout_s
    while not (run/'release_router').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('controller did not release the post-workload router')
        await asyncio.sleep(.1)
