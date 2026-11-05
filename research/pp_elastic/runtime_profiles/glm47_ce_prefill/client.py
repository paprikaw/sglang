"""C4 cold-cache API measurements; short profiling is a separate diagnostic."""
import argparse
import asyncio
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path

import aiohttp
import numpy as np

from common import save, visible_json, cfg, case

INPUT_TOKENS = 8192
OUTPUT_TOKENS = 128
CONCURRENCY = 4
INPUT_SEED = 20260920 + CONCURRENCY * 1000000 + INPUT_TOKENS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--base-url', default='http://172.26.92.180:31933')
    parser.add_argument('--ready-timeout', type=float, default=3600)
    return parser.parse_args()


class Client:
    def __init__(self, args):
        self.args = args
        self.case = case(args.arm, args.run)
        self.out = args.run.resolve() / args.arm
        self.out.mkdir(parents=True, exist_ok=True)
        self.base = args.base_url.rstrip('/')
        self.inputs = np.random.default_rng(INPUT_SEED).integers(
            0, 10000, size=(CONCURRENCY, INPUT_TOKENS), dtype=np.int32)
        self.input_sha = hashlib.sha256(self.inputs.tobytes()).hexdigest()
        self.started = time.monotonic()
        self.completed = 0
        self.formal_completed = 0
        self.phase_completed = {}

    def progress(self, phase, **fields):
        save(self.out / 'client_progress.json', dict(timestamp=time.time(), arm=self.args.arm,
            phase=phase, completed_requests=self.completed,
            formal_completed_requests=self.formal_completed,
            expected_formal_requests=sum(w['concurrency'] for w in cfg(self.args.run)['workloads'].values()) * cfg(self.args.run)['benchmark']['formal_trials'], elapsed_s=time.monotonic() - self.started, **fields))

    async def json_request(self, session, method, endpoint, payload=None, key=None):
        async with session.request(method, self.base + endpoint, json=payload) as response:
            text = await response.text()
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text
            if key:
                save(self.out / f'{key}.json', dict(status=response.status,
                     request=payload, response=value, timestamp=time.time()))
            assert response.status == 200, (endpoint, response.status, text[:2000])
            return value

    async def wait_ready(self, session):
        self.progress('waiting_for_server')
        deadline = time.monotonic() + self.args.ready_timeout
        while time.monotonic() < deadline:
            try:
                async with session.get(self.base + '/health', timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        save(self.out / 'client_ready.json', dict(passed=True,
                             timestamp=time.time(), arm=self.args.arm, endpoint='/health'))
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(3)
        raise TimeoutError('server readiness deadline')

    async def verify_service(self, session):
        # These reads and the one API request happen after server health, outside
        # measured samples. Remote-produced JSON always uses the GPFS barrier.
        self.progress('service_validation')
        weights = []
        for rank in range(self.case['world']):
            row = await asyncio.to_thread(visible_json, self.out / f'weights_rank{rank}.json',
                                          self.out / 'artifact_visibility/client', 30)
            assert row.get('passed') is True and len(row['layers']) == self.case['moe_layers_by_rank'][rank], rank
            weights.append(dict(rank=rank, passed=True, layers=len(row['layers'])))
        await self.json_request(session, 'GET', '/v1/models', key='models')
        await self.json_request(session, 'GET', '/get_server_info', key='server_info')
        payload = dict(input_ids=self.inputs[0, :32].tolist(),
            sampling_params=dict(temperature=0, max_new_tokens=4, min_new_tokens=4, ignore_eos=True))
        if self.case['dp'] > 1:
            payload['routed_dp_rank'] = 0
        result = await self.json_request(session, 'POST', '/generate', payload, 'service_validation')
        meta = result['meta_info']
        assert meta['prompt_tokens'] == 32 and meta['completion_tokens'] == 4, meta
        assert meta.get('cached_tokens') == 0, meta
        save(self.out / 'service_validation_result.json', dict(passed=True,
             weights=weights, prompt_tokens=32, completion_tokens=4, cached_tokens=0))

    async def request_one(self, session, tokens, label, index, output_tokens, batch_started):
        dp_rank = index % self.case['dp'] if self.case['dp'] > 1 else None
        payload = dict(input_ids=tokens, stream=True,
            sampling_params=dict(temperature=0, max_new_tokens=output_tokens,
                                 min_new_tokens=output_tokens, ignore_eos=True))
        if dp_rank is not None:
            payload['routed_dp_rank'] = dp_rank
        started = time.perf_counter()
        previous_count, previous_time = 0, started
        first_time, first_count = None, None
        gaps, chunks, final = [], [], None
        try:
            async with session.post(self.base + '/generate', json=payload) as response:
                assert response.status == 200, (response.status, (await response.text())[:2000])
                async for raw in response.content:
                    raw = raw.strip()
                    if not raw or raw == b'data: [DONE]':
                        continue
                    assert raw.startswith(b'data: '), raw[:100]
                    event = json.loads(raw[6:])
                    now = time.perf_counter()
                    count = event.get('meta_info', {}).get('completion_tokens', 0)
                    assert previous_count <= count <= output_tokens, (previous_count, count)
                    chunks.append(dict(t=now - started, batch_t=now - batch_started, data=event))
                    if count > previous_count:
                        if first_time is None:
                            first_time, first_count = now, count
                        else:
                            gaps.extend([(now - previous_time) / (count - previous_count)] *
                                        (count - previous_count))
                        previous_time, previous_count = now, count
                    final = event
            finished = time.perf_counter()
            assert final is not None and first_time is not None
            assert gaps or output_tokens == 1
            meta = final['meta_info']
            assert meta['completion_tokens'] == output_tokens and meta['prompt_tokens'] == len(tokens), meta
            assert meta.get('cached_tokens') == 0 and meta.get('num_retractions') == 0, meta
            assert first_count == 1, ('first streamed token must be separately observed', first_count)
            if dp_rank is not None:
                assert meta.get('dp_rank') == dp_rank, (dp_rank, meta)
            generated_ids = final.get('output_ids')
            assert isinstance(generated_ids, list) and len(generated_ids) == output_tokens
            row = dict(request_index=index, requested_dp_rank=dp_rank,
                input_sha256=hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest(),
                input_tokens=len(tokens), output_tokens=output_tokens, cached_tokens=meta['cached_tokens'],
                ttft_s=first_time - started, first_chunk_tokens=first_count,
                tpot_s=(previous_time - first_time) / (output_tokens - first_count) if output_tokens > first_count else None,
                median_adjusted_itl_s=statistics.median(gaps) if gaps else None, e2e_s=finished - started,
                request_start_from_batch_s=started - batch_started,
                first_token_from_batch_s=first_time - batch_started,
                last_token_from_batch_s=previous_time - batch_started,
                completion_from_batch_s=finished - batch_started,
                decode_post_first_tail_s=previous_time - first_time,
                http_post_last_token_tail_s=finished - previous_time,
                final_meta=meta, generated_text=final.get('text'), generated_ids=generated_ids, passed=True)
            save(self.out / f'{label}_request{index}.json', row)
            self.completed += 1
            self.phase_completed[label] = self.phase_completed.get(label, 0) + 1
            if label.startswith('formal'):
                self.formal_completed += 1
            self.progress(label, completed_in_trial=self.phase_completed[label],
                          latest_completed_index=index)
            return row
        finally:
            # Preserve any partial stream on failure as well as every successful
            # SSE event; timing reconstruction does not require server logs.
            save(self.out / f'{label}_request{index}_stream.json', chunks)

    async def trial(self, session, label, warmup=False, output_tokens=OUTPUT_TOKENS, diagnostic=False):
        self.progress(label, trial_requests=self.current['concurrency'], output_tokens=output_tokens)
        started = time.perf_counter()
        tasks = [asyncio.create_task(self.request_one(session, tokens.tolist(), label, index,
                  output_tokens, started)) for index, tokens in enumerate(self.inputs)]
        try:
            rows = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        wall = time.perf_counter() - started
        all_first = max(row['first_token_from_batch_s'] for row in rows)
        last_token = max(row['last_token_from_batch_s'] for row in rows)
        result = dict(arm=self.args.arm, label=label, batch=self.current['concurrency'], warmup=warmup,
            diagnostic=diagnostic, input_seed=self.current['seed'], input_sha256=self.input_sha,
            input_tokens=self.current['input_tokens'], output_tokens=output_tokens, wall_s=wall,
            mean_ttft_s=statistics.mean(row['ttft_s'] for row in rows),
            mean_tpot_s=statistics.mean(row['tpot_s'] for row in rows) if output_tokens > 1 else None,
            median_request_median_itl_s=statistics.median(row['median_adjusted_itl_s'] for row in rows) if output_tokens > 1 else None,
            mean_e2e_s=statistics.mean(row['e2e_s'] for row in rows),
            mean_decode_post_first_tail_s=statistics.mean(row['decode_post_first_tail_s'] for row in rows),
            time_until_all_first_tokens_s=all_first, time_until_last_output_token_s=last_token,
            post_all_first_tokens_to_last_token_s=last_token - all_first,
            batch_post_all_first_tail_s=wall - all_first,
            input_tokens_per_s=self.current['concurrency'] * self.current['input_tokens'] / wall,
            output_tokens_per_s=self.current['concurrency'] * output_tokens / wall,
            total_tokens_per_s=self.current['concurrency'] * (self.current['input_tokens'] + output_tokens) / wall,
            requests=rows, passed=True)
        save(self.out / f'{label}.json', result)
        print('TRIAL_PASS ' + json.dumps({key: value for key, value in result.items()
                                        if key != 'requests'}), flush=True)
        return result

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=1200)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self.wait_ready(session)
            await self.verify_service(session)
            configuration = cfg(self.args.run)
            results = []
            for name, workload in configuration['workloads'].items():
                self.current = workload
                seed = workload['seed']
                self.inputs = np.random.default_rng(seed).integers(0,10000,
                    size=(workload['concurrency'],workload['input_tokens']),dtype=np.int32)
                self.input_sha = hashlib.sha256(self.inputs.astype('<i4',copy=False).tobytes()).hexdigest()
                assert self.input_sha == workload['input_sha256']
                if self.case['ep'] in (4,8):
                    for rank in range(8):
                        route = await asyncio.to_thread(visible_json,self.out/f'route_rank{rank}.json',
                            self.out/'artifact_visibility/client',60)
                        assert route['passed'] and route['low_latency']
                for trial in range(configuration['benchmark']['warmup_trials']):
                    results.append(await self.trial(session,f'warmup_{name}_{trial}',warmup=True,
                        output_tokens=workload['output_tokens']))
                for trial in range(configuration['benchmark']['formal_trials']):
                    row = await self.trial(session,f'formal_{name}_{trial}',
                        output_tokens=workload['output_tokens'])
                    row.update(workload=name,trial=trial)
                    results.append(row)
                save(self.out/'benchmark_result.json',dict(passed=False,arm=self.args.arm,
                    formal_trials=configuration['benchmark']['formal_trials'],formal_requests=sum(w['concurrency'] for w in configuration['workloads'].values()) * configuration['benchmark']['formal_trials'],results=results,profiler_in_formal_samples=False))
            save(self.out/'benchmark_result.json',dict(passed=True,arm=self.args.arm,
                formal_trials=configuration['benchmark']['formal_trials'],formal_requests=sum(w['concurrency'] for w in configuration['workloads'].values()) * configuration['benchmark']['formal_trials'],results=results,profiler_in_formal_samples=False))
            self.progress('complete', formal_completed=True, formal_requests=self.formal_completed)
            save(self.out / 'client_result.json', dict(passed=True, formal_passed=True,
                 timestamp=time.time(), arm=self.args.arm))
            return 0


def main():
    client = Client(parse_args())
    try:
        return asyncio.run(client.run())
    except BaseException as error:
        save(client.out / 'client_failure.json', dict(timestamp=time.time(), error=str(error),
             error_type=type(error).__name__, traceback=traceback.format_exc()))
        client.progress('failed')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
