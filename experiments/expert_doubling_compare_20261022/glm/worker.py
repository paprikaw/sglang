"""Guard-only single-layer real-weight compute microbenchmark.

No GPU import happens until the active lease and narrow experiment contract
have been authenticated. The required expert_layer_only policy is not yet
implemented in the canonical Guard; this worker must not be run around it.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import socket
import stat
import sys
import time

from layouts import (HIDDEN, INTERMEDIATE, CAPACITY, TOTAL_ROWS, ACTIVE_EXPERTS, point, summarize)

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
ENGINE = WORKSPACE / '.engine-worktrees/glm47-ep16-canonical-profile-20261004'
MODEL = Path('/data/gpfs/projects/punim2715/huggingface/GLM-4.7-AWQ')
LEGACY = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep4_local_expert_profile_20261005_r2/harness')
GUARD_SCRIPTS = ROOT.parent / 'guard/skills/llm-experiment-guard/scripts'


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticate(run):
    sys.path.insert(0, str(GUARD_SCRIPTS))
    from artifact_visibility import wait_for_stable_json

    def read(path):
        value, _ = wait_for_stable_json(Path(path), timeout_s=10)
        return value

    receipt_path = Path(os.environ['EXPERIMENT_GUARD_LEASE_RECEIPT'])
    mode = receipt_path.stat()
    assert mode.st_uid == os.getuid() and stat.S_IMODE(mode.st_mode) == 0o600
    receipt = read(receipt_path)
    registry = read(Path(receipt['registry_dir']) / 'registry.json')
    lease = next(e for e in registry['leases'] if e['lease_id'] == receipt['lease_id'])
    spec = read(run / 'dashboard_spec.json')
    assert lease['state'] == 'active'
    assert lease['token_sha256'] == hashlib.sha256(receipt['token'].encode()).hexdigest()
    assert time.time() - lease['heartbeat_at'] <= lease['ttl_s']
    assert lease['spec_sha256'] == receipt['spec_sha256'] == sha(run / 'dashboard_spec.json')
    assert lease['experiment_id'] == receipt['experiment_id'] == spec['id']
    assert len(lease['nodes']) == 1
    assert socket.gethostname().split('.')[0] in lease['nodes']
    binding = read(os.environ['EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS'])
    assert binding['lease_id'] == receipt['lease_id']
    assert binding['policy'] == spec['model_cache']['policy'] == 'expert_layer_only'
    assert spec['model_cache']['model_path'] == str(MODEL)
    assert spec['model_cache']['layer'] == 3
    assert spec['model_cache']['expert_ids'] == list(range(40))
    for file in (MODEL / 'config.json', MODEL / 'model.safetensors.index.json',
                 ROOT / 'worker.py', ROOT / 'layouts.py', LEGACY / 'profile_entry.py',
                 LEGACY / 'profile_utils.py', LEGACY / 'common.py'):
        match = [e for e in spec['frozen_inputs'] if e['path'] == str(file)]
        assert len(match) == 1 and match[0]['sha256'] == sha(file)
    return {'experiment_id': spec['id'], 'lease_id': lease['lease_id'],
            'spec_sha256': receipt['spec_sha256'], 'node': lease['nodes'][0]}


def legacy_helpers():
    # Only borrow the independently checked AWQ loader/reference; never its
    # historical experiment geometry or timing results.
    sys.path.insert(0, str(LEGACY))
    spec = importlib.util.spec_from_file_location('validated_expert_loader', LEGACY / 'profile_entry.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(run):
    authorization = authenticate(run)
    logging.basicConfig(level=logging.INFO, format='[TP0] %(message)s')
    import torch
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.humming import (
        HummingRunnerCore, HummingMoeQuantInfo, HummingRunnerInput, HummingGemmType,
    )

    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    set_global_server_args_for_scheduler(ServerArgs(
        model_path=str(MODEL), dtype='bfloat16', quantization='humming',
        moe_runner_backend='humming', attention_backend='triton', disable_radix_cache=True,
    ))
    selection_path = run / 'selection.json'
    selection = json.loads(selection_path.read_text())
    helper = legacy_helpers()
    helper.LOCAL_EXPERTS = 40
    helper.EXPECTED_CHECKPOINT_TENSOR_BYTES = 490_291_200
    layer, raw_weights, provenance = helper.load_real_experts(MODEL)
    for expert, projections in raw_weights.items():
        for projection, tensors in projections.items():
            for field, tensor in tensors.items():
                name = f'model.layers.3.mlp.experts.{expert}.{projection}.{field}'
                actual = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                assert actual == selection['tensors'][name]['sha256'], name
    core = HummingRunnerCore(MoeRunnerConfig(
        num_experts=160, num_local_experts=40, hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE, top_k=8, activation='silu',
    ))
    quant = HummingMoeQuantInfo(layer)
    torch.manual_seed(20261022)
    recv = torch.randn((40, CAPACITY, HIDDEN), device='cuda', dtype=torch.bfloat16) * .125
    numerical, results, tuning = [], [], {}
    for total in TOTAL_ROWS:
        points = [point(total, active) for active in ACTIVE_EXPERTS if active <= total]
        prepared = []
        for p in points:
            inp = HummingRunnerInput(
                hidden_states=recv,
                topk_ids=torch.zeros((max(1, total // 8), 8), device='cuda', dtype=torch.int32),
                topk_weights=torch.full((max(1, total // 8), 8), 1 / 8,
                                        device='cuda', dtype=torch.float32),
                expert_num_tokens=torch.tensor(p['counts'], device='cuda', dtype=torch.int32),
                expected_m=p['expected_m'], gemm_type=HummingGemmType.GROUPED_MASKED,
                apply_routed_scaling_factor=False,
            )
            def invoke(inp=inp):
                return core.run(inp, quant, {}).hidden_states
            for _ in range(5):
                output = invoke()
            torch.cuda.synchronize()
            for expert in (p['active_ids'][0], p['active_ids'][-1]):
                n = min(2, p['counts'][expert])
                expected = helper.reference_expert(recv[expert, :n], raw_weights[expert])
                numerical.append(dict(total_assignments=total,
                                      active_experts=p['active_experts'], expert=expert,
                                      **helper.error_metrics(output[expert, :n], expected)))
            prepared.append(invoke)

        for mode in ('eager', 'cuda_graph'):
            graphs, callables = [], []
            for invoke in prepared:
                if mode == 'eager':
                    callables.append(invoke)
                else:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = invoke()
                    graphs.append((graph, output))
                    callables.append(graph.replay)
            samples = [[] for _ in points]
            block_medians = [[] for _ in points]
            # Paired blocks, rotating/reversed order, all shapes already warm.
            size = len(points)
            orders = [list(range(size)), list(reversed(range(size))),
                      list(range(size//2, size)) + list(range(size//2))]
            for order in orders:
                for j in order:
                    # Immediate warmup gives every active weight set the same
                    # warm-cache protocol, rather than favoring adjacent shapes.
                    for _ in range(3):
                        callables[j]()
                    starts = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
                    ends = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
                    for start, end in zip(starts, ends):
                        start.record()
                        callables[j]()
                        end.record()
                    ends[-1].synchronize()
                    values = [s.elapsed_time(t) for s, t in zip(starts, ends)]
                    samples[j].extend(values)
                    block_medians[j].append(summarize(values)['median_ms'])
            for p, invoke, values, blocks in zip(points, prepared, samples, block_medians):
                output = invoke()
                finite = all(bool(torch.isfinite(output[e, :p['counts'][e]]).all())
                             for e in p['active_ids'])
                assert finite
                timing = summarize(values)
                results.append(dict(**p, mode=mode, timing=timing,
                                    block_median_ms=blocks, cuda_event_ms=values,
                                    finite=finite,
                                    effective_tflops=p['gemm_flops'] / timing['median_ms'] / 1e9))
            torch.cuda.synchronize()
            del graphs, callables, output
        tuning[str(total)] = core.get_humming_gemm_configs(HummingGemmType.GROUPED_MASKED)
        save(run / 'profile/partial.json', {'passed': None, 'rows': results})
        print('EXPERT_FOOTPRINT_POINT_COMPLETE', total, flush=True)

    assert len(results) == 126 and len(numerical) == 126
    assert all(x['passed'] for x in numerical)
    save(run / 'profile/result.json', dict(
        selection_sha256=sha(selection_path), passed=True, rows=results,
        numerical=numerical, authorization=authorization, weight_provenance=provenance,
        backend='Humming AWQ W4A16 GROUPED_MASKED', tuning=tuning,
        scope='one layer, one GPU, fixed received token-expert rows; no communication',
        protocol='warm-cache; 40 resident experts; fixed capacity/tuning estimate within each T',
        fixed_expert_capacity=CAPACITY, hardware_counters_measured=False,
        gpu=dict(name=torch.cuda.get_device_name(0),
                 total_bytes=torch.cuda.get_device_properties(0).total_memory,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated()),
    ))
    print('EXPERT_FOOTPRINT_COMPLETE', flush=True)
    while True:
        time.sleep(1)  # Owned supervisor performs verified shutdown.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())

