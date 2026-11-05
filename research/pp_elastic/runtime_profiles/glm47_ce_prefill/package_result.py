#!/usr/bin/env python3
"""CPU-only validated GLM CE Prefill archival; never launch or read GPU state.

The serving harness is frozen separately. This tool only packages completed,
cleaned runs, independently recomputes metrics from raw client/SSE receipts,
and labels historical RDMA/SM references as noncontemporaneous comparisons.
"""
import argparse
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
MANAGER = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager')
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
HISTORY_AUDIT = WORKSPACE / 'experiments/glm47_a100_pcie_rdma_comparison_20261010_analysis/comparison.json'
ARMS = ('pp2ep4', 'ep8')
PRIVATE_KEYS = {'token', 'auth_token', 'authorization_token', 'signature', 'secret',
                'guard_env', 'guard_lease_id', 'lease_id', 'model_cache_bindings',
                'admission_id', 'binding'}
CODE_SUFFIXES = {'.py', '.cu', '.cuh', '.cpp', '.hpp', '.h'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f'Artifact changed while reading: {path}')
    return json.loads(payload)


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if k.lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2) + '\n')


def equal(left, right, message):
    require(math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-8), message)


def stats(values):
    values = sorted(values)
    require(bool(values), 'Empty latency distribution')
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p99=values[max(0, math.ceil(.99 * len(values)) - 1)],
                minimum=values[0], maximum=values[-1])


def contract(config):
    require(tuple(config.get('arms', config['cases'])) == ARMS, 'Both layout arms are required')
    require(config['nodes'] == ['spartan-gpgpu168', 'spartan-gpgpu169'], 'Different node pair')
    require(config['model_path'].endswith('/GLM-4.7-AWQ'), 'Different model')
    workload = config['workloads']['prefill_c4']
    require((workload['input_tokens'], workload['output_tokens'], workload['concurrency']) ==
            (8192, 1, 4), 'Different Prefill workload')
    require(config['benchmark']['formal_trials'] == 1, 'This report is one formal C4 batch')
    for arm, expected in [('pp2ep4', (4, 4, 2, 1, 1024)), ('ep8', (8, 8, 1, 2, 2048))]:
        case = config['cases'][arm]
        require((case['tp'], case['ep'], case['pp'], case['dp'], case['global_prefill_chunk']) ==
                expected, f'{arm}: different parallel layout or global chunk')
        require(case['world'] == 8 and case['attn_tp'] == 4 and case['ep_capacity'] == 256 and
                case['local_prefill_chunk'] == 1024, f'{arm}: different capacity/local chunk')
    return workload


def postcheck(run):
    cleanup = read(run / 'a100_pair/task/cleanup.json')
    require(all(cleanup.get(key) is True for key in
                ('passed', 'model_processes_exited', 'gpu_idle')), 'Task cleanup incomplete')
    candidates = sorted((run / 'dashboard_guard/checks').glob('*-post.json'))
    good = [read(path) for path in candidates if read(path).get('passed') is True]
    require(bool(good), 'No successful Guard postcheck')
    post = good[-1]
    require(post.get('scientific_success') is True and post.get('cleanup_verified') is True,
            'Guard scientific/cleanup success absent')
    cache = post.get('model_cache', {})
    require(cache.get('passed') is True, 'CPU weight-cache usage verification absent')
    require(len(cache.get('nodes', [])) == 2, 'Two CPU cache nodes not verified')
    for node in cache['nodes']:
        require(len(node.get('uses', [])) == 2, 'Both model loads per node must use CPU cache')
        for usage in node['uses']:
            require(set(usage['observed_ranks']) == set(usage['expected_ranks']) and
                    len(usage['observed_ranks']) == 4, 'CPU cache rank usage mismatch')
            require(usage['delta_bytes'] > 0 and len(usage['delta_sha256']) == 64,
                    'No fresh CPU cache loader evidence')
    return sanitize(post), sanitize(cleanup)


def audit_counter_receipts(output, arm):
    """Only accept the post-formal, CPU-only owned Unix socket protocol.

    Controller and worker wall clocks are on different hosts. The unique
    request nonce and source ordering establish the post-formal boundary;
    timestamps are ordered only within each worker/supervisor node.
    """
    request = read(output / 'CE_SNAPSHOT_REQUEST.json')
    receipts = read(output / 'ce_counter_receipts.json')
    require(request['passed'] is True and request['arm'] == arm and
            request['after_formal_requests'] is True and request['no_other_model_requests'] is True and
            request['timed'] is False and math.isfinite(request['created_at']) and request['created_at'] > 0,
            'CE post-formal request incomplete')
    require(receipts['passed'] is True and receipts['request'] == request and
            receipts['ranks'] == list(range(8)) and receipts['all_workers_received'] is True and
            receipts['transport'] == 'unix_socket' and receipts['cpu_only'] is True and receipts['timed'] is False,
            'CE receipt protocol incomplete')
    nodes = receipts['node_socket_receipts']
    require(len(nodes) == 2 and {row['node_rank'] for row in nodes} == {0, 1},
            'CE node socket receipt set incomplete')
    acknowledgements = {}
    for node in nodes:
        node_rank = node['node_rank']
        require(node == read(output / f'ce_socket_node{node_rank}.json'), 'CE node receipt/aggregate mismatch')
        require(node['passed'] is True and node['request_id'] == request['request_id'] and
                node['transport'] == 'unix_socket' and node['cpu_only'] is True and node['timed'] is False and
                math.isfinite(node['completed_at']) and node['completed_at'] > 0,
                'CE node socket receipt invalid')
        require(len(node['responses']) == 4 and {row['rank'] for row in node['responses']} ==
                set(range(node_rank * 4, node_rank * 4 + 4)), 'CE node rank ACK set incomplete')
        for ack in node['responses']:
            rank = ack['rank']
            setup = read(output / f'ce_config_rank{rank}.json')
            final_path = output / f'ce_final_rank{rank}.json'
            final = read(final_path)
            require(all(ack[key] == setup[key] for key in ('rank', 'pid', 'process_start_ticks', 'counter_socket')) and
                    ack['request_id'] == request['request_id'] and ack['receipt_sha256'] == digest(final_path),
                    'CE socket ACK identity/nonce/hash mismatch')
            require(final['passed'] is True and final['source'] == 'unix_socket_listener' and
                    final['after_formal_requests'] is True and final['cpu_only'] is True and final['timed'] is False and
                    final['request_id'] == request['request_id'] and
                    all(final[key] == setup[key] for key in ('rank', 'pid', 'process_start_ticks')) and
                    math.isfinite(final['received_at']) and math.isfinite(final['completed_at']) and
                    0 < final['received_at'] <= final['completed_at'] <= node['completed_at'],
                    'CE final counter receipt identity or same-node ordering invalid')
            socket_path = Path(setup['counter_socket'])
            require(socket_path.is_absolute() and socket_path.parent.parent == Path('/tmp') and
                    socket_path.parent.name == f'sglang-ce-counters-{setup["counter_socket_uid"]}' and
                    setup['counter_socket_protocol'] == 1 and setup['counter_socket_mode'] == 0o600 and
                    setup['counter_socket_directory_mode'] == 0o700 and
                    socket_path.name.endswith(f'-r{rank}-p{setup["pid"]}-t{setup["process_start_ticks"]}.sock') and
                    len(str(socket_path).encode()) < 108,
                    'CE local counter socket path identity invalid')
            acknowledgements[rank] = ack
    require(set(acknowledgements) == set(range(8)), 'CE eight-rank ACK coverage absent')
    return request


def audit_ranks(run, config, ce):
    result = []
    native = config['deepep_native']
    for arm in ARMS:
        case = config['cases'][arm]
        service = read(run / arm / 'service_validation_result.json')
        require(service['passed'] is True and service['cached_tokens'] == 0,
                f'{arm}: service validation failed')
        if ce:
            request = audit_counter_receipts(run / arm, arm)
        for node_rank, node in enumerate(config['nodes']):
            end = read(run / arm / f'node{node_rank}_end.json')
            require(end['node'] == node and end['arm'] == arm and end['node_rank'] == node_rank,
                    'Cleanup node identity mismatch')
            require(end['returncode'] == 0 and all(end.get(key) is True for key in
                        ('worker_exited', 'owned_processes_exited', 'gpu_idle')) and
                    end['gpu_query']['passed'] is True, f'{arm}: node cleanup failed')
        for rank in range(8):
            weight = read(run / arm / f'weights_rank{rank}.json')
            route = read(run / arm / f'route_rank{rank}.json')
            check = read(run / arm / f'comm_check_rank{rank}.json')
            ep_rank = rank % 4 if case['pp'] == 2 else rank
            ep = case['ep']
            require(weight['passed'] is True and weight['rank'] == rank and
                    weight['attn_tp'] == 4 and weight['attn_dp'] == case['dp'] and
                    weight['moe_ep'] == ep and weight['moe_tp'] == 1, 'Weight rank/layout mismatch')
            require(len(weight['layers']) == case['moe_layers_by_rank'][rank], 'Layer count mismatch')
            require(all(layer['experts'] == 160 // ep and layer['intermediate'] == 1536 and
                        layer['quant_method'] == 'HummingMoEMethod' for layer in weight['layers']),
                    'Expert computation backend changed')
            require('A100' in weight['gpu'] and weight['nvlink_peers'] ==
                    [peer in (rank % 4, (rank % 4) ^ 1) for peer in range(4)], 'NVLink topology mismatch')
            require(route['passed'] is True and route['rank'] == rank and route['low_latency'] is True and
                    route['local_ep_rank'] == ep_rank and route['native_sha256'] == native['sha256'] and
                    route['native_path'] == native['path'] and route['nvlink_pair_only_enabled'] is True,
                    'Native/route identity mismatch')
            mapped = [peer // 4 == ep_rank // 4 for peer in range(ep)]
            require(route['peer_mapped'] == mapped, 'Remote peers unexpectedly mapped locally')
            require(check['passed'] is True and check.get('global_rank', rank) == rank and
                    all(row['passed'] is True for row in check['cases']), 'Communication check failed')
            record = dict(arm=arm, rank=rank, ep_rank=ep_rank, native_sha256=native['sha256'],
                          local_experts=160 // ep, moe_layer_count=len(weight['layers']),
                          pcie_threshold=route['pcie_p2p_max_tokens'], peer_mapped=mapped)
            if ce:
                setup = read(run / arm / f'ce_config_rank{rank}.json')
                before = read(run / arm / f'ce_after_validation_rank{rank}.json')
                after = read(run / arm / f'ce_final_rank{rank}.json')
                require(setup['passed'] is True and setup['rank'] == rank and setup['local_ep_rank'] == ep_rank and
                        setup['world'] == ep and setup['capacity'] == 256 and setup['cuda_graph'] is False and
                        setup['native'] == native and setup['peer_mapping'] == mapped,
                        'CE constructor identity/layout mismatch')
                expected = dict(dispatch_dedup=1, matched_peer_order=1, skip_local_drain=int(ep == 4), local_notify=1)
                require(setup['settings'] == expected and route['pcie_p2p_max_tokens'] == 256,
                        'CE settings or threshold mismatch')
                local = 160 // ep
                header = 16 + ((local * 4 + 15) // 16) * 16
                frame = header + 256 * local * (5120 * 2 + 32)
                extra = ((ep * frame + 2 * ep * 128 + 127) // 128) * 128
                require(setup['geometry'] == dict(local_experts=local, header_bytes=header,
                        frame_bytes=frame, extra_bytes=extra), 'CE frame geometry mismatch')
                require(before['passed'] is True and after['passed'] is True and after['after_formal_requests'] is True,
                        'CE final runtime receipt absent')
                require(after['request_id'] == request['request_id'] and
                        all(after[key] == setup[key] for key in ('pid', 'process_start_ticks')),
                        'CE final receipt nonce/worker identity mismatch')
                low, high = before['counters'], after['counters']
                delta = [b - a for a, b in zip(low['counters'], high['counters'])]
                require(len(delta) == 6 and delta[0] > 2 and delta[1] > 4 and delta[2] > 0 and
                        delta[3] > 0 and delta[4] > 1 and delta[5] > 1, 'CE not exercised beyond startup checks')
                require(high['epoch'] - low['epoch'] == delta[0] and delta[0] == delta[4] + delta[5] and
                        delta[4] == delta[5] and delta[1] == 2 * delta[0], 'CE dispatch/combine accounting inconsistent')
                require(all(row.get('rank', rank) == rank for row in (before, after)), 'CE rank receipt mismatch')
                require(check.get('graph_captured') is False, 'CE communication must remain eager')
                record.update(ce_counters_before=low, ce_counters_after=high, ce_delta=delta,
                              counter_scope='after standalone comm checks through service validation, warmup and formal batch')
            result.append(record)
        if ce:
            step = case['ep']
            arm_records = [row for row in result if row['arm'] == arm]
            for start in range(0, 8, step):
                block = arm_records[start:start + step]
                require(len({row['ce_counters_after']['epoch'] for row in block}) == 1 and
                        len({row['ce_delta'][0] for row in block}) == 1, 'EP group collective count/epoch mismatch')
    return result


def audit_deployments(run, config, ce):
    deployments = [read(path) for path in (run / 'deployments').glob('*/manifest.json')]
    require(len(deployments) == 2 and {row['arm'] for row in deployments} == set(ARMS),
            'Exactly two completed deployments are required')
    result = []
    for row in deployments:
        require(row['status'] == 'benchmark_complete_stopped', 'Deployment not completed/stopped')
        case = config['cases'][row['arm']]
        args = row['server']['launch_args']
        expected = {'--attention-backend': 'triton', '--quantization': 'humming',
            '--moe-runner-backend': 'humming', '--context-length': '16384',
            '--mem-fraction-static': '0.80', '--max-total-tokens': '65536',
            '--chunked-prefill-size': str(case['global_prefill_chunk']), '--max-prefill-tokens': '1024',
            '--cuda-graph-backend-prefill': 'disabled', '--deepep-mode': 'low_latency',
            '--deepep-dispatcher-output-dtype': 'bf16'}
        for key, value in expected.items():
            require(key in args and args[args.index(key) + 1] == value, f'{row["arm"]}: launch {key} changed')
        decode = args[args.index('--cuda-graph-backend-decode') + 1]
        require(decode == ('disabled' if ce else 'full'), 'Unexpected Decode Graph startup mode')
        require(all(flag in args for flag in ('--disable-radix-cache', '--disable-overlap-schedule',
                    '--disable-shared-experts-fusion', '--disable-custom-all-reduce')), 'Serving feature flags changed')
        result.append(dict(arm=row['arm'], deployment_id=row['deployment_id'],
            actual_arguments=args, prefill_graph='disabled', decode_graph=decode))
    return result


def recompute(run, config, cohort):
    workload = contract(config)
    input_path = Path(workload['input_artifact'])
    require(digest(input_path) == workload['input_sha256'], 'Frozen request input SHA changed')
    input_data = input_path.read_bytes()
    require(len(input_data) == 4 * 8192 * 4, 'Unexpected input binary shape')
    rows, request_rows = [], []
    for arm in ARMS:
        result = read(run / arm / 'benchmark_result.json')
        require(result['passed'] is True and result['formal_trials'] == 1, 'Incomplete benchmark')
        batches = [b for b in result['results'] if b['label'] == 'formal_prefill_c4_0']
        require(len(batches) == 1, 'Exactly one formal Prefill batch is required')
        batch = batches[0]
        require(batch['passed'] is True and not batch['warmup'] and not batch.get('diagnostic') and
                batch['input_sha256'] == workload['input_sha256'] and len(batch['requests']) == 4,
                'Formal batch completeness/input mismatch')
        batch_file = read(run / arm / 'formal_prefill_c4_0.json')
        require(all(key in batch and batch[key] == value for key, value in batch_file.items()) and
                set(batch) - set(batch_file) <= {'trial', 'workload'}, 'Batch/aggregate receipt mismatch')
        for request in batch['requests']:
            index = request['request_index']
            require(index in range(4), 'Unexpected request index')
            raw = read(run / arm / f'formal_prefill_c4_0_request{index}.json')
            require(raw == request and raw['passed'] is True and
                    (raw['input_tokens'], raw['output_tokens'], raw['cached_tokens']) == (8192, 1, 0),
                    'Standalone request completion/token mismatch')
            expected_input = hashlib.sha256(input_data[index * 8192 * 4:(index + 1) * 8192 * 4]).hexdigest()
            require(raw['input_sha256'] == expected_input, 'Request input differs from frozen binary')
            stream = read(run / arm / f'formal_prefill_c4_0_request{index}_stream.json')
            positive = [event for event in stream if event['data'].get('meta_info', {}).get('completion_tokens', 0) > 0]
            require(bool(positive), 'No output token observed')
            first, last = positive[0], positive[-1]
            require(last['data']['meta_info']['completion_tokens'] == 1 and
                    last['data']['output_ids'] == raw['generated_ids'] and len(raw['generated_ids']) == 1,
                    'Raw SSE output-token receipt mismatch')
            equal(first['t'], raw['ttft_s'], 'Raw SSE TTFT mismatch')
            equal(last['batch_t'], raw['last_token_from_batch_s'], 'Raw SSE last-token timestamp mismatch')
            equal(raw['completion_from_batch_s'] - raw['request_start_from_batch_s'], raw['e2e_s'],
                  'HTTP completion E2E mismatch')
            for event in stream:
                equal(event['batch_t'] - event['t'], raw['request_start_from_batch_s'], 'SSE clocks disagree')
            require(raw['tpot_s'] is None and raw['final_meta'].get('num_retractions') == 0,
                    'One output token has no TPOT / no eviction expected')
            request_rows.append(dict(cohort=cohort, run=str(run), arm=arm, request_index=index,
                input_sha256=raw['input_sha256'], input_tokens=8192, output_tokens=1, cached_tokens=0,
                ttft_s=first['t'], ttvot_s=last['t'], e2e_s=raw['e2e_s'],
                completion_from_batch_s=raw['completion_from_batch_s']))
        subset = [row for row in request_rows if row['arm'] == arm]
        require(len({row['request_index'] for row in subset}) == 4, 'Duplicate or missing request IDs')
        http_wall = max(row['completion_from_batch_s'] for row in subset)
        archive_wall = batch['wall_s']
        require(archive_wall >= http_wall > 0, 'Client makespan ordering invalid')
        input_total = sum(row['input_tokens'] for row in subset)
        equal(input_total / archive_wall, batch['input_tokens_per_s'], 'Recorded input/wall throughput mismatch')
        row = dict(cohort=cohort, run=str(run), arm=arm, request_count=4, input_tokens_total=input_total,
                   http_completion_batch_wall_s=http_wall, archival_inclusive_batch_wall_s=archive_wall,
                   input_tok_s=input_total / archive_wall, input_tok_s_archival_inclusive=input_total / archive_wall,
                   input_tok_s_http_completion=input_total / http_wall,
                   archival_tail_s=archive_wall - http_wall, input_sha256=workload['input_sha256'])
        for metric in ('ttft_s', 'ttvot_s', 'e2e_s'):
            row[metric] = stats([request[metric] for request in subset])
        row['tpot_ms'] = None
        rows.append(row)
    return rows, request_rows


def historical_runs():
    audit = read(HISTORY_AUDIT)
    matches = [Path(row['run']) for row in audit['datasets']
               if Path(row['run']).name == 'glm47_a100_equal_local_chunk_20261010_r1']
    require(len(matches) == 1, 'Historical equal-local RDMA reference ambiguous')
    return [('historical_rdma', matches[0]),
            ('historical_sm_pcie', MANAGER / 'runtime/glm47_a100_pcie_prefill_20261010_r2')]


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def copy_json(source, dest):
    write(dest, read(source))


def archive_evidence(run, dest, config):
    # This is a positive allowlist. Never recursively copy runtime or Guard.
    for name in ('config.json', 'frozen_inputs.json', 'completed_run.json', 'kv_cache_policy.json'):
        copy_json(run / name, dest / name)
    for name in ('summary.json', 'summary.csv'):
        source = run / 'compare' / name
        if name.endswith('.json'):
            copy_json(source, dest / 'compare' / name)
        else:
            (dest / 'compare').mkdir(exist_ok=True)
            shutil.copy2(source, dest / 'compare' / name)
    for arm in ARMS:
        for pattern in ('benchmark_result.json', 'service_validation_result.json', 'formal_prefill_c4_0*.json',
                        'node*_end.json', 'route_rank*.json', 'weights_rank*.json', 'comm_check_rank*.json',
                        'ce_config_rank*.json', 'ce_after_validation_rank*.json', 'ce_final_rank*.json',
                        'CE_SNAPSHOT_REQUEST.json', 'ce_counter_receipts.json', 'ce_socket_node*.json',
                        'preload_rank*.json', 'commwarm_rank*.json', 'boundary_rank*.json', 'formal_dispatch_rank*.json'):
            for source in sorted((run / arm).glob(pattern)):
                copy_json(source, dest / arm / source.name)
    for source in (run / 'deployments').glob('*/manifest.json'):
        copy_json(source, dest / 'deployments' / source.parent.name / source.name)
    specs = config['model_cache']['node_specs']
    for node, record in specs.items():
        source = Path(record['path'])
        require(digest(source) == record['sha256'], 'CPU cache spec changed')
        copy_json(source, dest / 'cpu_cache_specs' / f'{node}.json')
    frozen = read(run / 'frozen_inputs.json')['files']
    copied = []
    for index, receipt in enumerate(frozen):
        source = Path(receipt['path'])
        require(digest(source) == receipt['sha256'], f'Frozen source changed: {source}')
        if source.suffix in CODE_SUFFIXES:
            target = dest / 'frozen_source' / f'{index:04d}_{source.name}'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(dict(original_path=str(source), snapshot=str(target.relative_to(dest)), sha256=receipt['sha256']))
    write(dest / 'frozen_source/index.json', dict(files=copied, frozen_file_count=len(frozen),
        scope='Code text only. Model weights, Guard runtime bindings/leases, binaries and logs are excluded.'))
    native = Path(config['deepep_native']['path'])
    require(digest(native) == config['deepep_native']['sha256'], 'Native binary SHA changed')
    build = native.parents[1] / 'build_result.json'
    require(build.exists(), f'Native build receipt missing: {build}')
    receipt = read(build)
    require(receipt['passed'] is True and receipt['native_sha256'] == config['deepep_native']['sha256'] and
            receipt.get('cuda_override_commit') == config['deepep_source']['commit'],
            'Native build receipt identity mismatch')
    copy_json(build, dest / 'native/build_result.json')
    write(dest / 'native/identity.json', dict(wrapper=config['deepep_source'], native=config['deepep_native'],
        actual_host_commit=receipt.get('source_commit'), actual_cuda_commit=receipt.get('cuda_override_commit'),
        host_base_native_sha256=receipt.get('host_base_native_sha256'), binary_copied=False))
    (dest / 'tools').mkdir(exist_ok=True)
    shutil.copy2(__file__, dest / 'tools/package_result.py')


def report(rows):
    current = {row['arm']: row for row in rows if row['cohort'] == 'current_ce'}
    p, e = current['pp2ep4'], current['ep8']
    lines = ['# GLM-4.7-AWQ A100 CE Prefill comparison', '',
        f"PP2×EP4 / EP8 throughput ratio: **{p['input_tok_s'] / e['input_tok_s']:.3f}×**; "
        f"mean TTFT {p['ttft_s']['mean']:.3f} / {e['ttft_s']['mean']:.3f} s.", '',
        '| Cohort | Layout | Input tok/s ↑ | Benchmark wall s ↓ | TTFT mean / median / P99 s ↓ | E2E mean / median / P99 s ↓ |',
        '|---|---|---:|---:|---|---|']
    for row in rows:
        ttft, end = row['ttft_s'], row['e2e_s']
        lines.append(f"| {row['cohort']} | {row['arm']} | {row['input_tok_s']:.2f} | "
            f"{row['archival_inclusive_batch_wall_s']:.4f} | {ttft['mean']:.4f} / {ttft['median']:.4f} / {ttft['p99']:.4f} | "
            f"{end['mean']:.4f} / {end['median']:.4f} / {end['p99']:.4f} |")
    lines += ['', 'All rows use nodes168/169, 8 A10080GB PCIe, GLM-4.7-AWQ/Humming, Attention TP4 per node, '
        'cold-prefix C4, 8192 input →1 output token. PP2×EP4 uses 43/46 MoE layers per stage; EP8 uses89 layers '
        'and Attention DP2. Local scheduler chunk1024 and source capacity256 are equal; PP global1024, EP8 global2048.', '',
        'Primary throughput = completed input tokens / the original benchmark wall_s, matching historical displayed throughput. '
        'This client wall includes trailing receipt writes. The CSV separately reports input / common batch start to last HTTP completion '
        'as input_tok_s_http_completion and its wall as http_completion_batch_wall_s. '
        'Earlier synchronous writes may still delay later observation; no correction is invented. '
        'Request TTFT/E2E are client measured. TTVOT equals TTFT here because there is one output token; TPOT is undefined.', '',
        'Each configuration has one formal batch of four completed requests. P99 uses nearest rank '
        'ceil(0.99×4), hence equals the maximum; it is descriptive across these requests, not a repeated-service tail.', '',
        'Current CE retains pair NVLink, uses Copy Engine for local cross-pair payload and original RDMA for cross-node traffic. '
        'EP8 retains the RDMA drain; PP local EP4 skips it. CE runtime counters establish actual traffic beyond standalone '
        'communication checks, but include service validation/warmup/formal inference and do not give a per-request wire profile.', '',
        'Historical RDMA and forced SM PCIe are noncontemporaneous references. They share old native9dc34887 with each other '
        'and differ in threshold0 versus256; current CE uses a different LL CUDA object. This is not a same-native CE/RDMA '
        'ablation. The old threshold1 cohort is deliberately excluded.', '',
        'Prefill Graph is off in all compared rows. Historical Decode Graph was full while current Decode Graph is disabled; '
        'this Prefill-only workload contains no measured Decode sequence, so these rows do not establish a Graph effect.', '',
        'Model load/CPU weight cache startup, Humming preload, correctness tests and warmup are excluded from formal latency. '
        'Raw SSE, all request receipts, 16 rank layout/transport receipts, CE counters, cleanup/cache checks, frozen source '
        'hashes and actual native build provenance are retained in the snapshot. No model weights or private Guard bindings/leases are copied.']
    return '\n'.join(lines) + '\n'


def register(config, dest, rows):
    marker = f"<!-- begin {config['id']} -->"
    p, e = [next(row for row in rows if row['cohort'] == 'current_ce' and row['arm'] == arm) for arm in ARMS]
    entry = (f"\n{marker}\n## {config['id']}\n\n- Snapshot: `{dest}`; complete C4 Prefill8192→1, "
             'nodes168/169, 8 A100, GLM4.7-AWQ/Humming; equal localchunk1024/sourcecap256, PPglobal1024/EP8global2048.\n'
             f"- Actual CE CUDA `{config['deepep_source']['commit']}` / native `{config['deepep_native']['sha256']}`; "
             'pair NVLink + local cross-pair CE + remote RDMA; eager, both layouts completed and cleaned.\n'
             f"- Input throughput PP2×EP4 {p['input_tok_s']:.2f}, EP8 {e['input_tok_s']:.2f} tok/s; "
             f"mean TTFT {p['ttft_s']['mean']:.4f}/{e['ttft_s']['mean']:.4f}s. Original benchmark wall_s; "
             'nearest-rank P99 over4 completed requests, not repeated-service tails.\n'
             '- Historical equal-local RDMA and forced SM PCIe retained as noncontemporaneous references, '
             'not same-native CE/RDMA causal evidence. Startup/frozen/native/cache/cleanup evidence validated.\n'
             f"<!-- end {config['id']} -->\n")
    with REGISTRY.open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        if marker not in handle.read():
            handle.seek(0, 2)
            handle.write(entry)
            handle.flush()
            os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--no-register', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    completed = read(run / 'completed_run.json')
    require(completed['passed'] is True and tuple(completed['arms']) == ARMS, 'Both CE layouts must finish')
    config = read(run / 'config.json')
    require(config['id'] == run.name, 'Run identity mismatch')
    contract(config)
    summary = read(run / 'compare/summary.json')
    require(summary['passed'] is True and summary['config_id'] == config['id'], 'Comparison analyzer did not pass')
    post, cleanup = postcheck(run)
    ranks = audit_ranks(run, config, ce=True)
    deployments = audit_deployments(run, config, ce=True)
    rows, requests = recompute(run, config, 'current_ce')
    refs = []
    for label, historical in historical_runs():
        old = read(historical / 'config.json')
        contract(old)
        require(old['source_commit'] == config['source_commit'] and old['model_revision'] == config['model_revision'] and
                old['workloads']['prefill_c4']['input_sha256'] == config['workloads']['prefill_c4']['input_sha256'],
                'Historical model/engine/input mismatch')
        expected_threshold = 0 if label == 'historical_rdma' else 256
        require(old['transport_policy']['pcie_p2p_max_tokens'] == expected_threshold, 'Historical policy mismatch')
        old_ranks = audit_ranks(historical, old, ce=False)
        old_deployments = audit_deployments(historical, old, ce=False)
        require(all(row['pcie_threshold'] == expected_threshold for row in old_ranks), 'Historical runtime policy mismatch')
        old_rows, old_requests = recompute(historical, old, label)
        rows.extend(old_rows)
        requests.extend(old_requests)
        refs.append(dict(cohort=label, run=str(historical), config=old, rank_checks=old_ranks,
                         deployments=old_deployments))
    for row in rows[:2]:
        saved = next(item for item in summary['rows'] if item['arm'] == row['arm'] and item['workload'] == 'prefill_c4')
        equal(saved['input_tokens_per_s'], row['input_tok_s_archival_inclusive'], 'Analyzer throughput/raw mismatch')
        equal(saved['ttft_mean_ms'] / 1000, row['ttft_s']['mean'], 'Analyzer TTFT/raw mismatch')
        equal(saved['ttft_p99_ms'] / 1000, row['ttft_s']['p99'], 'Analyzer P99/raw mismatch')
    if args.check_only:
        print(json.dumps(dict(passed=True, run=str(run), rows=rows, snapshot_written=False), ensure_ascii=False))
        return
    dest = (args.out or MANAGER / 'experiments' / config['id']).resolve()
    require(not dest.exists(), f'Refusing to overwrite an existing experiment snapshot: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + config['id'] + '-packaging-', dir=dest.parent))
    try:
        archive_evidence(run, temporary, config)
        write(temporary / 'guard_postcheck_public.json', post)
        write(temporary / 'cleanup_public.json', cleanup)
        write(temporary / 'cpu_cache_usage_receipt.json', post['model_cache'])
        write(temporary / 'rank_invariants.json', dict(passed=True, ranks=ranks))
        write(temporary / 'deployment_invariants.json', dict(passed=True, deployments=deployments))
        write(temporary / 'historical_references.json', dict(passed=True, source_audit=str(HISTORY_AUDIT), datasets=refs))
        write(temporary / 'comparison.json', dict(passed=True, rows=rows, requests=requests,
            percentiles='Nearest rank ceil(p*n), P99=max for n4. Descriptive, not repeated-service tails.',
            primary_timing_boundary='original benchmark wall_s including trailing receipt writes',
            secondary_timing_boundary='common batch start to last HTTP completion',
            ce_counter_scope='after standalone comm check through validation/warmup/formal requests',
            same_native_ce_rdma_ablation=False))
        csv_write(temporary / 'full_metrics.csv', rows)
        csv_write(temporary / 'request_metrics.csv', requests)
        (temporary / 'summary.md').write_text(report(rows))
        manifest = dict(id=config['id'], title='GLM A100 CE Prefill PP2×EP4 versus EP8',
            created_at=datetime.now(timezone.utc).isoformat(), status='complete', original_runtime=str(run),
            implementation=dict(engine='sglang', worktree=config['source_worktree'], commit=config['source_commit'],
                annotated_tag=config['source_tag'], deepep_source=config['deepep_source'], native=config['deepep_native']),
            model_path=config['model_path'], model_revision=config['model_revision'], nodes=config['nodes'], jobs=config['jobs'],
            workload=config['workloads']['prefill_c4'], cases=config['cases'], key_results=rows[:2],
            archive_policy='positive allowlist, private Guard keys redacted; no weights/binaries/raw logs copied',
            cpu_cache_verified=True, cleanup_verified=True, historical_references_noncontemporaneous=True,
            cuda_graph_prefill=False, cuda_graph_decode=False, raw_formal_request_count=8,
            input_tokens_per_formal_batch=32768, percentile_method='nearest-rank, four requests per layout')
        write(temporary / 'manifest.json', manifest)
        checksums = {str(path.relative_to(temporary)): digest(path) for path in sorted(temporary.rglob('*')) if path.is_file()}
        write(temporary / 'checksums.json', dict(passed=True, files=checksums))
        require(all(digest(temporary / name) == expected for name, expected in checksums.items()), 'Snapshot hash verification failed')
        os.replace(temporary, dest)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    if not args.no_register:
        register(config, dest, rows)
    print(json.dumps(dict(passed=True, snapshot=str(dest), files=len(checksums), rows=rows,
                          registry_appended=not args.no_register), ensure_ascii=False))


if __name__ == '__main__':
    main()
