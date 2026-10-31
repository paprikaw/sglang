"""Freeze matched 122B FP8 runs using the qualified H100/Guard harness."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
ENGINE = WORKSPACE / '.engine-worktrees/qwen35b-decode-ep-overlap-20261027'
COMMIT = '0f2f1990994774472b1089d2b0682f0bf926a6b4'
TAG = 'experiment/sglang/qwen122b-latest-decode-20261027-cpu'
MODEL = Path('/data/gpfs/projects/punim2715/huggingface/Qwen3.5-122B-A10B-FP8')
REVISION = 'a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9'


def read(p): return json.loads(p.read_text())
def save(p, v): p.write_text(json.dumps(v, indent=2) + '\n')
def desc(p): return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--variant', choices=['native', 'overlap', 'pp'], required=True)
    a = ap.parse_args()
    run = a.run.resolve()
    assert not run.exists(), 'Use a fresh immutable run directory'
    subprocess.run([sys.executable, str(ROOT.parent/'h100_ep_decode_overlap_20261027/prepare.py'),
                    '--run', str(run), '--variant', a.variant], check=True, stdout=subprocess.DEVNULL)
    cfg = read(run/'config.json')
    code = run/'code'
    assert subprocess.check_output(['git', '-C', str(ENGINE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert subprocess.check_output(['git', '-C', str(ENGINE), 'rev-parse', TAG+'^{commit}'], text=True).strip() == COMMIT
    # This full-weight native FP8 loader was already qualified on this model.
    adapter = (ROOT.parent/'ep2_pp2_qwen_h100_20261025/runtime_adapter.py').read_text()
    adapter = adapter.replace("native_kernel=True,", "native_kernel=True, actual_moe_runner=str(experts[0].quant_method.runner.runner_backend),")
    adapter += '''
_base_install = install
def install():
    _base_install()
    import diagnostic_switch
    diagnostic_switch.install()
    if os.environ['EP2_PP2_ARM'] == 'pp2':
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
    from sglang.srt.layers.moe import decode_overlap
    decode_overlap.install()
'''
    (code/'runtime_adapter.py').write_text(adapter)
    hopper = code/'hopper_reference.py'
    text = hopper.read_text()
    text = text.replace("'--mem-fraction-static', '.94'", "'--mem-fraction-static', '.945', '--max-total-tokens', '81920' if arm == 'ep2' else '163840'")
    text = text.replace("'--context-length', '262144'", "'--quantization', 'fp8', '--context-length', '262144'")
    text = text.replace("'168' if arm == 'ep2' else '164'", "'104' if arm == 'ep2' else '100'")
    text = text.replace("'--mamba-max-states-per-path', '1'", "'--mamba-max-states-per-path', '1', '--mamba-radix-cache-strategy', 'extra_buffer_lazy' if arm == 'ep2' else 'extra_buffer'")
    text = text.replace(", '--deepep-dispatcher-output-dtype', 'bf16'", '')
    hopper.write_text(text)
    benchmark = (ROOT.parent/'h100_ep_decode_overlap_20261027/decode_benchmark.py').read_text()
    benchmark = benchmark.replace('len(q)==40', "len(q)==cfg['num_hidden_layers']")
    benchmark = benchmark.replace("result=base.cohort(current,key,probe);check(result,4096,8)", "result=base.cohort(current,key,probe);base.h.save(root/'numerical_probe.json',dict(performance_sample=False,result=result));check(result,4096,8)")
    benchmark = benchmark.replace("            if arm=='ep2':\n                set_diagnostic", "            if arm=='pp2':\n                native_reference=[r['output_ids'] for r in group['requests']]\n                base.h.save(root/'native_output_reference.json',dict(performance_sample=False,policy=0,output_ids=native_reference))\n            if arm=='ep2':\n                set_diagnostic")
    benchmark = benchmark.replace("            for _ in range(3):health();check(base.cohort(current,key,dec))", "            for _ in range(3):\n                health();warm=base.cohort(current,key,dec);check(warm)\n                if arm=='pp2':\n                    assert [r['output_ids'] for r in warm['requests']]==native_reference, 'PP lookahead changed the native fixed-group output'\n            if arm=='pp2':base.h.save(root/'native_lookahead_equivalence.json',dict(passed=True,performance_sample=False,full_sequences_compared=case['requests']*3,output_tokens_per_request=128))")
    # Preserve any rejected cohort before raising the exact-cache admission gate.
    benchmark = benchmark.replace("                    assert row['output_tokens']==output and (row.get('cached_tokens') or 0)==cached", "                    if row['output_tokens']!=output or (row.get('cached_tokens') or 0)!=cached:\n                        base.h.save(root/'rejected_cohort.json',dict(performance_sample=False,expected_cache=cached,expected_output=output,result=group))\n                        raise AssertionError('Exact cache/output admission failed; rejected cohort preserved')")
    # Capture resolved allocator/graph/parallel state for each workload.
    benchmark = benchmark.replace("tokens=prompts(current,case)", "base.h.save(root/'server_info.json',base.h.safe_server_info(base.h.http_json(cfg['port'],'/get_server_info',key,timeout=30)))\n            tokens=prompts(current,case)")
    (code/'decode_benchmark.py').write_text(benchmark)
    cfg.update(model_path=str(MODEL), model_revision=REVISION, engine_tag=TAG,
               port=27621, dist_port=27622, num_hidden_layers=48,
               precision='Native FP8 W8A8/DeepGEMM on BOTH layouts; BF16 KV/residuals; FP32 GDN',
               expert_mode='native_fp8', shared_small_gemm=None, ep_layout_adapter=None,
               hybrid_state_cache_strategy='EP lazy ping-pong / PP single extra buffer', hybrid_state_dtype='float32', mem_fraction_static=.945, kv_pool_tokens_per_rank=81920 if a.variant!='pp' else 163840, kv_pool_target_gib_per_rank=1.875,
               generic_skip_redundant_decode_prefix_state_lock=True,
               generic_expert_optimization_applied_to_both=True,
               comparison_scope='Same native FP8/DeepGEMM backend; no BF16-only Triton substitution. PP9 lookahead; EP native/shared overlap and optional two-chunk MoE overlap.',
               reference_run=str(ROOT.parent/'ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7'),
               protocol='H100178 NVLink GPU0/1; complete Qwen3.5-122B-A10B FP8 (48 layers). '
                        'Exactly the prior B8/B16/B32 repeated and B32 diverse Decode prompts; '
                        'GPUcached4096+new1 -> output128. EP2 AttentionDP2/TP1, PP2 24+24 layers '
                        'and two fixed B/2 slots, same even/odd request groups. Both native FP8/DeepGEMM. '
                        'PP9 early send/lookahead; EP native shared-expert overlap ON; '
                        'static EP native or two-chunk layer-local overlap captured before timing. '
                        'Native FP32 GDN prefix caches; common skip-redundant-Decode-prefix-lock optimization ON. '
                        'EP lazy ping-pong with native overlap schedule, PP single extra buffer with native PP schedule; '
                        'both use three state slots/request, page64, mem_fraction_static=0.945 and 1.875GiB KV/rank (81920 EP tokens / 163840 PP tokens). '
                        'Three full warmups, five profiler-off formal cohorts, independent profiles; no CPU KV restore.')
    for key in ['SGLANG_DEEPEP_BF16_DISPATCH', 'SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED']:
        cfg['launch_env'].pop(key, None)
    cfg['launch_env']['SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP'] = '1'
    cfg['launch_env']['SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK'] = '1'
    node_file = run/'cache/node.json'
    node = read(node_file)
    node['model_path'] = str(MODEL)
    save(node_file, node)
    cfg['model_cache']['node_specs'][cfg['node']] = desc(node_file)
    save(run/'config.json', cfg)
    spec = read(run/'dashboard_spec.json')
    inputs = []
    for item in spec['frozen_inputs']:
        path = Path(item['path'])
        if path.is_relative_to(code) or path in (run/'config.json', node_file): continue
        if path.is_relative_to(Path('/data/gpfs/projects/punim2715/huggingface')):
            candidate = MODEL/path.name
            if candidate.is_file(): inputs.append(desc(candidate))
        else: inputs.append(item)
    # Every imported private adapter is sealed, including unchanged baseline helpers.
    inputs += [desc(p) for p in sorted(code.glob('*.py'))] + [desc(run/'config.json'), desc(node_file)]
    for rel in ['python/sglang/srt/layers/quantization/fp8.py',
                'python/sglang/srt/layers/moe/moe_runner/deep_gemm.py']:
        inputs.append(desc(ENGINE/rel))
    spec.update(title='H100 122B FP8 latest EP2 / PP2: '+a.variant, port=27620,
                model_cache=cfg['model_cache'], frozen_inputs=list({d['path']:d for d in inputs}.values()))
    spec['sources'] = [s for s in spec['sources'] if s['worktree'] != str(ENGINE)] + [dict(worktree=str(ENGINE),commit=COMMIT,tag=TAG)]
    save(run/'dashboard_spec.json', spec)
    manifest = read(run/'launch_manifest.json')
    manifest.update(model_path=str(MODEL), model_revision=REVISION, model_layers=48,
                    source_tag=TAG, source_commit=COMMIT, engine_source_tag=TAG,
                    precision=cfg['precision'], expert_mode=cfg['expert_mode'],
                    shared_small_gemm=None, bf16_only_optimization_not_applicable=True,
                    comparison_scope=cfg['comparison_scope'], protocol=cfg['protocol'],
                    private_adapters=[desc(p) for p in sorted(code.glob('*.py'))])
    save(run/'launch_manifest.json', manifest)
    registry = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/kv_checkpoints/registry.json')
    data = read(registry)
    assert 'Qwen3.5-122B-A10B' not in str(data), 'A candidate now exists; live compatibility lookup is required'
    save(run/'kv_checkpoint_discovery.json',dict(registry=desc(registry),registered_model_candidate=False,
         model_path=str(MODEL),reason='No registered checkpoint for this hybrid GDN model; actual native GPU prefix/GDN state is produced outside timing.'))
    for path in code.glob('*.py'): path.chmod(0o600)
    print(json.dumps(dict(run=str(run),variant=a.variant,model=str(MODEL),points=cfg['expected_points'])))


if __name__ == '__main__': main()
