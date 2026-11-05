"""Check the real selected dataset with the released CLI on CPU before loading models."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import threading


async def preflight(run, output=None):
    cfg = json.loads((run/'config.json').read_text())
    packet = json.loads(sys.stdin.readline())
    env = dict(os.environ, **packet['environment'])
    sys.path.insert(0, str(run/'harness'))
    from test_adapter import Backend, serve
    from sglang_adapter import Adapter
    from run import command
    from aiohttp import web
    output = output or run/'evidence/dataset_preflight'
    output.mkdir(parents=True, exist_ok=False)

    class FastBackend(Backend):
        async def chat(self, request):
            body = await request.json()
            self.calls.append(dict(max_tokens=body.get('max_completion_tokens', body.get('max_tokens')),
                messages=len(body['messages'])))
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            count = body.get('max_completion_tokens', body.get('max_tokens'))
            events = [dict(id=body['rid'], object='chat.completion.chunk', model=body['model'],
                choices=[dict(index=0, delta=dict(content='x'), finish_reason=None)]),
                dict(id=body['rid'], object='chat.completion.chunk', model=body['model'],
                choices=[dict(index=0, delta={}, finish_reason='length')],
                usage=dict(prompt_tokens=100, completion_tokens=count, total_tokens=100+count))]
            for event in events:
                await response.write(b'data: '+json.dumps(event).encode()+b'\n\n')
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
            return response

    backend = FastBackend()
    async with serve(backend.app()) as target:
        adapter = Adapter(dict(mode='colocated', colocated=[dict(name='CPU-fake', url=target,
            attention_dp_size=4)]), frontend_key='CPU-qualification-only', backend_key='CPU-fake',
            audit_path=output/'routing.jsonl')
        async with serve(adapter.app()) as url:
            args = argparse.Namespace(model='CPU-qualification', tokenizer=cfg['agentx_tokenizer'],
                concurrency=2, seed=42, duration=2, entries=cfg['dataset_entries'], output=output,
                dataset=None, local_dataset=Path(cfg['agentx_dataset']), smoke=True, warmup_grace=30)
            cmd = command(args, url)+['--workers-max', '1']
            cmd[cmd.index('--benchmark-grace-period')+1] = '5'
            env.update(AGENTX_WARMUP_PREFILL_LIMIT=str(cfg['warmup_only_prefill_limit']),
                AGENTX_WARMUP_ADMISSION_EVIDENCE=str(output/'warmup_admission'),
                AGENTX_FRONTEND_KEY='CPU-qualification-only', HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', HF_HUB_CACHE=cfg['agentx_tokenizer_cache'],
                AIPERF_DATASET_WEKA_PARALLEL_WORKERS='4', TOKENIZERS_PARALLELISM='false',
                AIPERF_HTTP_KEEPALIVE_TIMEOUT='4')
            child = None
            def terminated(signum, frame):
                if child and child.returncode is None:
                    os.killpg(child.pid, signal.SIGTERM)
                raise SystemExit(128+signum)
            signal.signal(signal.SIGTERM, terminated)
            def parent_closed():
                sys.stdin.read()
                os.kill(os.getpid(), signal.SIGTERM)
            threading.Thread(target=parent_closed, daemon=True).start()
            with (output/'harness.log').open('wb') as log:
                child = await asyncio.create_subprocess_exec(*cmd, env=env, stdout=log,
                    stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                try:
                    code = await asyncio.wait_for(child.wait(), timeout=210)
                finally:
                    if child.returncode is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(child.wait(), 10)
                        except TimeoutError:
                            os.killpg(child.pid, signal.SIGKILL)
                            await child.wait()
            assert code == 0, 'Actual selected dataset CLI failed; see CPU preflight log'
            result = json.loads((output/'agentx/profile_export_aiperf.json').read_text())
            assert backend.calls and not result['error_summary']
            passed = dict(passed=True, source_families=cfg['dataset_entries'],
                selected_dataset=cfg['agentx_dataset'], calls=len(backend.calls),
                cpu_only=True, fake_backend=True, gpu_model_loaded=False,
                performance_measurement=False, cli_exit_code=code,
                profile_requests=result['request_count']['avg'],
                warmup_requests=result['warmup_metrics']['request_count']['avg'])
            (output/'summary.json').write_text(json.dumps(passed, indent=2)+'\n')
            print(json.dumps(passed), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    asyncio.run(preflight(args.run.resolve(), args.output.resolve() if args.output else None))
