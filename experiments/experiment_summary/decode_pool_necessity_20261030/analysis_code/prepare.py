"""Freeze a GPU-hot, equal-four-GPU Decode comparison; no launch here."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
OLD = WS / 'experiments/h100_five_layouts_hicache_20261028/runs/ep_178_r4'
LATEST = WS / 'experiments/h100_ep_decode_overlap_20261027/runs/pp2_lookahead_r1'
GUARD = Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
QUALIFIED_BUNDLE = WS / 'experiments/h100_small_batch_layout_verify_20261029/guard_bundle_posix_v1'
QUALIFIED_GUARD = QUALIFIED_BUNDLE / 'skills/llm-experiment-guard/scripts'


def read(p): return json.loads(p.read_text())
def save(p, v): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(v, indent=2) + '\n')
def desc(p): return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--node', default='spartan-gpgpu011')
    ap.add_argument('--job', default='33758541')
    ap.add_argument('--layouts', default='ep4,two_pp2,four_ep1,pp4')
    ap.add_argument('--dashboard-port', type=int, default=27820)
    ap.add_argument('--profiles', default='short,dense,long')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--cases', help='Comma-separated existing case IDs for targeted confirmation')
    ap.add_argument('--reverse-layout-order', action='store_true')
    ap.add_argument('--mem-fraction-static', type=float, default=.95)
    ap.add_argument('--dense-kv-token-cap', type=int, default=147456)
    a = ap.parse_args(); run = a.run.resolve()
    selected = a.layouts.split(',')
    assert selected and len(selected)==len(set(selected)) and set(selected)<=set(('ep4','two_ep2','two_pp2','four_ep1','pp4'))
    profiles=a.profiles.split(',');assert profiles and set(profiles)<=set(('short','dense','long'))
    cases=[]
    for p in (256,4096):
        cases.extend(dict(id=f'p{p}_n{n}_repeated',profile='short',P=p,requests=n,family='repeated') for n in (8,16,32,64))
        cases.append(dict(id=f'p{p}_n128_repeated',profile='dense',P=p,requests=128,family='repeated'))
    cases.extend(dict(id=f'p4096_n{n}_diverse',profile='short' if n==8 else 'dense',P=4096,requests=n,family='diverse') for n in (8,128))
    for p in (32768,131072):
        cases.extend(dict(id=f'p{p}_n{n}_repeated',profile='long',P=p,requests=n,family='repeated') for n in (8,32))
    cases=[c for c in cases if c['profile'] in profiles]
    if a.cases:
        selected_cases=a.cases.split(',')
        assert len(selected_cases)==len(set(selected_cases))
        assert set(selected_cases)<=set(c['id'] for c in cases)
        cases=[c for c in cases if c['id'] in selected_cases]
    assert cases and a.repeats>=3 and .8<=a.mem_fraction_static<=.995
    assert all(c['requests']*(c['P']+192)//4<=a.dense_kv_token_cap for c in cases if c['profile']=='dense')
    if run.exists(): raise RuntimeError('Use a fresh immutable run directory')
    subprocess.run([sys.executable, str(GUARD/'backfill_trigger.py')], check=True, stdout=subprocess.DEVNULL)
    run.mkdir(parents=True)
    shutil.copytree(OLD/'code', run/'code', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('task', 'cache', 'evidence', 'results'): (run/name).mkdir()
    for name in ('gap_schedule.py', 'pp_fixed_cohort.py', 'pp_schedule_opt.py', 'pp_transfer_audit.py'):
        shutil.copyfile(LATEST/'code'/name, run/'code'/name)
    for name in ('verify_workload.py', 'forward_observer.py'):
        shutil.copyfile(ROOT/name, run/'code'/name)
    cfg = read(LATEST/'config.json')
    cfg.update(id=run.name, node=a.node, job_id=a.job, code_root=str(run/'code'),
               guard_scripts=str(QUALIFIED_GUARD), admin_key_file=str(run/'admin.key'),
               port=27721, dist_port=30721, prefix_tokens=4096, output_tokens=128,
               repeats=a.repeats, expected_points=len(cases)*len(selected), max_context=131328, requests=128,
               local_requests=32, local_mamba_slots=128, mem_fraction_static=a.mem_fraction_static,
               common_context=4096, large_context=4096, hicache_gib_per_rank=0,
               workload_contexts=[4096], new_prefill_tokens=1,
               workloads=cases,
               profiles={'short':dict(max_context=8192,max_requests=64,mamba_equiv=80,kv_token_cap_per_full_gpu=73728),
                         'dense':dict(max_context=8192,max_requests=128,mamba_equiv=160,kv_token_cap_per_full_gpu=a.dense_kv_token_cap),
                         'long':dict(max_context=131328,max_requests=32,mamba_equiv=48,kv_token_cap_per_full_gpu=1100032)},
               protocol='Single node four H100 GPUs per layout, complete Qwen3.6-35B-A3B BF16; global N8..128; '
               'P256/4096/32768/131072 GPU-hot prefixes, Q1 -> output128. No HiCache or CPU KV restore. '
               f'{a.repeats} profiler-off cohorts after separate warmup. EP4 AttentionDP4/TP1; '
               '2xEP2; 2xPP2 policy9 with two fixed slots per replica; 4xEP1; PP4 native scheduling. '
               f'Short/dense/long memory profiles separately launched; {a.mem_fraction_static} static fraction. '
               'KV upper bounds cover the largest requested cell per GPU and leave unused memory for Graph/NCCL workspaces. '
               'Actual native capacity can be lower than these explicit demand bounds. '
               'Capacity-infeasible cells are recorded as such, never turned into Decode timings. '
               'Same small-token Expert Triton family, native Attention/GDN and shared Expert overlap. '
               'Report arithmetic mean request TPOT, mean request E2E, and complete-cohort makespan separately. '
               'Read actual forward/graph histograms outside timing; never infer global tokens from local ranks.')
    fields = {'policy_path':'pp_ready_policy.bin', 'diagnostic_eager_policy_path':'eager_policy.bin',
              'fragment_policy_path':'fragment_policy.bin', 'gap_policy_path':'gap_policy.bin',
              'fixed_cohort_control':'fixed_cohort.bin', 'fixed_cohort_size':'fixed_cohort_size.bin',
              'forward_counter_control':'forward_counter.bin',
              'transfer_audit_control':'transfer_audit.bin', 'expert_workload_path':'active_workload.txt'}
    for key, name in fields.items(): cfg[key] = str(run/name)
    for name in ('pp_ready_policy.bin','eager_policy.bin','fragment_policy.bin','gap_policy.bin','transfer_audit.bin'):
        p=run/name; p.write_bytes(b'0'); p.chmod(0o600)
    for name in ('fixed_cohort.bin', 'fixed_cohort_size.bin','forward_counter.bin'):
        p=run/name; p.write_bytes(bytes([0])); p.chmod(0o600)
    (run/'active_workload.txt').write_text('startup\n')
    (run/'admin.key').write_text(secrets.token_urlsafe(32)); (run/'admin.key').chmod(0o600)
    cfg['launch_env'].update(EP_STEADY_HARDWARE='H100', DEEPEP_LL_NVLINK_PAIR_ONLY='0',
        EP_DECODE_OVERLAP_POLICY='0', PP_FIXED_COHORT_CONTROL=cfg['fixed_cohort_control'],
        PP_FIXED_COHORT_SIZE=cfg['fixed_cohort_size'], PP_TRANSFER_AUDIT_CONTROL=cfg['transfer_audit_control'],
        BENCH_FORWARD_COUNTER_CONTROL=cfg['forward_counter_control'],
        PP_GAP_SEPARATE_CHANNELS='1', BENCH_REPLICAS='2', NCCL_DEBUG='WARN')
    cfg['layouts'] = []
    definitions=[('ep4',4,1,[[0,1,2,3]]),('two_ep2',2,1,[[0,1],[2,3]]),
                               ('two_pp2',1,2,[[0,1],[2,3]]),
                               ('four_ep1',1,1,[[0],[1],[2],[3]]),
                               ('pp4',1,4,[[0,1,2,3]])]
    if a.reverse_layout_order:definitions.reverse()
    for profile in profiles:
        for kind,ep,pp,groups in definitions:
            if kind not in selected:continue
            arm=kind+'_'+profile
            logs=[dict(id=f'{cfg["id"]}.{arm}.r{r}', path=str(run/'cache'/f'{arm}.r{r}.log'),
                       expected_ranks=[f'PP{i} TP0' for i in range(pp)] if pp>1 else [f'TP{i}' for i in range(ep)])
                  for r in range(len(groups))]
            cfg['layouts'].append(dict(id=arm,kind=kind,profile=profile,ep=ep,tp=ep,dp=ep,pp=pp,replicas=len(groups),gpu_groups=groups,usage_logs=logs))
    nodecfg=read(OLD/'cache/node.json'); nodecfg.update(node=a.node,expected_job_id=a.job)
    save(run/'cache/node.json',nodecfg)
    cfg['model_cache'].update(on_miss='fail',
        cpu_memory_budget=dict(schema_version=1,kv_banks_per_node={a.node:[]}),
        hicache_reserve_bytes_per_node={a.node:0},node_specs={a.node:desc(run/'cache/node.json')},
        usage_logs={a.node:[u for r in cfg['layouts'] for u in r['usage_logs']]})
    # Preserve existing frozen engine and exact native. No shared source edits.
    engine=Path(cfg['engine_root']);commit=subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip()
    assert commit==cfg['engine_commit']
    assert not subprocess.check_output(['git','-C',str(engine),'status','--porcelain'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(engine),'rev-parse',cfg['engine_tag']+'^{commit}'],text=True).strip()==commit
    fixed=run/'code/pp_fixed_cohort.py'; text=fixed.read_text()
    text=text.replace('size not in (8, 16, 32)', 'size not in (4, 8, 16, 32, 64)')
    text=text.replace("markers = [r.origin_input_ids[0] for r in requests]", "replicas = int(os.environ.get('BENCH_REPLICAS', '1'))\n    markers = [5000 + (r.origin_input_ids[0] - 5000) // replicas for r in requests]")
    text=text.replace("r.origin_input_ids[0] % 2,r.origin_input_ids[0]", "((r.origin_input_ids[0]-5000)//replicas) % 2,(r.origin_input_ids[0]-5000)//replicas")
    fixed.write_text(text)
    # Local N4 on each PP2 replica has two B2 pipeline slots. Extend only
    # the admitted static-message shape; native schemas remain mandatory.
    gap=run/'code/gap_schedule.py'; text=gap.read_text()
    assert 'value.shape[0] not in (4, 8, 16)' in text
    assert 'len(batch.reqs) not in (4, 8, 16)' in text
    text=text.replace('value.shape[0] not in (4, 8, 16)', 'value.shape[0] not in (2, 4, 8, 16, 32)')
    text=text.replace('len(batch.reqs) not in (4, 8, 16)', 'len(batch.reqs) not in (2, 4, 8, 16, 32)')
    gap.write_text(text)
    (run/'code/runtime_adapter.py').write_text('''import os
def install():
    import h100_runtime_adapter, diagnostic_switch, fragmentation, ep_fragmentation, forward_observer
    h100_runtime_adapter.install()
    diagnostic_switch.install()
    fragmentation.install()
    ep_fragmentation.install()
    if os.environ.get("BENCH_PP2_LOOKAHEAD") == "1":
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
    forward_observer.install()
''')
    runner=run/'code/runner.py';text=runner.read_text()
    marker="if __name__=='__main__':"
    assert text.count(marker)==1
    text=text.replace(marker,"import verify_workload\nverify_workload.install(sys.modules[__name__])\n\n"+marker)
    # The original owned launcher remains intact; select PP2-only adapters per server.
    text=text.replace("row=layout(cfg,arm);alias='ep2' if row['ep']>1 else 'pp2'", "row=layout(cfg,arm);alias='ep2' if row['ep']>1 else 'pp2'\n    inherited=dict(inherited, BENCH_PP2_LOOKAHEAD='1' if row['pp']==2 else '0')")
    runner.write_text(text)
    reg=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/kv_checkpoints/registry.json')
    entries=read(reg).get('entries',{}); matches=[e for e in entries.values() if 'qwen' in json.dumps(e.get('compatibility',{})).lower()]
    save(run/'kv_checkpoint_discovery.json',dict(registry=desc(reg),queried_at=time.time(),qwen_entries=len(matches),
        mode='Native model-produced GPU-hot prefix. No checkpoint restore; no claim of live checkpoint reuse.'))
    save(run/'config.json',cfg)
    spec=read(OLD/'dashboard_spec.json')
    spec.update(id=cfg['id'],title='35B single-node Decode pool necessity: measured and capacity cells',port=a.dashboard_port,
                runtime_dir=str(run/'dashboard_guard'),model_cache=cfg['model_cache'])
    spec['tasks'][0].update(nodes=[a.node],root=str(run/'task'),controller_script=str(runner),
                            expected=cfg['expected_points'],label='Resolved GPU-hot performance / native-capacity cells')
    frozen=[]
    for d in read(LATEST/'dashboard_spec.json')['frozen_inputs']:
        p=Path(d['path'])
        if LATEST in p.parents:continue
        assert desc(p)['sha256']==d['sha256'],str(p)
        frozen.append(d)
    frozen.extend(desc(p) for p in [run/'config.json',run/'cache/node.json',run/'kv_checkpoint_discovery.json',*sorted((run/'code').glob('*.py'))])
    frozen.extend(desc(QUALIFIED_GUARD/name) for name in ('gpu_registry.py','model_cache_adapter.py','dashboard_guard.py'))
    frozen.append(desc(QUALIFIED_BUNDLE/'manifest.json'))
    frozen.extend(read(QUALIFIED_BUNDLE/'manifest.json')['frozen_inputs'])
    spec['frozen_inputs']=list({r['path']:r for r in frozen}.values())
    # Preserve the actual inference identity. The separate persistent CPU
    # daemon runs its own unchanged engine revision; Guard validates that
    # identity as well. Bind it explicitly instead of inheriting a source list.
    cache_engine=Path(nodecfg['engine_root'])
    cache_commit=subprocess.check_output(['git','-C',str(cache_engine),'rev-parse','HEAD'],text=True).strip()
    assert cache_commit==nodecfg['source_commit']
    assert not subprocess.check_output(['git','-C',str(cache_engine),'status','--porcelain'],text=True).strip()
    cache_tag='experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated'
    assert subprocess.check_output(['git','-C',str(cache_engine),'rev-parse',cache_tag+'^{commit}'],text=True).strip()==cache_commit
    cache_files=('cpu_canonical.py','cpu_canonical_loader.py','cpu_protocol.py')
    compatibility=[]
    for name in cache_files:
        relative=Path('python/sglang/srt/weight_cache')/name
        left=desc(engine/relative);right=desc(cache_engine/relative)
        assert left['sha256']==right['sha256']
        compatibility.append(dict(inference=left,cpu_daemon=right))
    spec['sources']=[dict(role='inference_engine',worktree=str(engine),commit=commit,tag=cfg['engine_tag']),
                     dict(role='persistent_cpu_weight_daemon',worktree=str(cache_engine),commit=cache_commit,tag=cache_tag)]
    save(run/'dashboard_spec.json',spec)
    save(run/'launch_manifest.json',dict(experiment_id=cfg['id'],model_path=cfg['model_path'],node=a.node,job=a.job,
        engine_root=str(engine),engine_commit=commit,engine_source_tag=cfg['engine_tag'],
        cpu_weight_daemon_source=spec['sources'][1],cache_loader_compatibility=compatibility,
        native=desc(Path(cfg['deepep_native'])),wrapper=cfg['deepep_wrapper'],protocol=cfg['protocol'],
        layouts=cfg['layouts'],workloads=cfg['workloads'],frozen_inputs=spec['frozen_inputs'],
        primary_hypothesis='After including single-GPU and EP2 replicas, PP remains on the best-layout envelope in some Decode regimes; another regime prefers EP strongly enough to motivate ratio control.',
        decision_rule='Search batch, context and input diversity using all screening cohorts. Compare to the best tested full-budget baseline, '
        'not only EP4. Separate Decode TPOT, finite-cohort throughput, memory feasibility and request E2E. '
        'A >=20 percent two-sided best-layout difference and >=10 percent optimistic adaptive-pool benefit are useful practical targets, not significance tests. '
        'Revalidate decisive cells and any capacity-only claim with appropriate memory sizing before recommending a mechanism.'))
    save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',timestamp=time.time(),stage='queued',valid_points=0,expected=cfg['expected_points']))
    for p in (run/'code').glob('*.py'): ast.parse(p.read_text(),filename=str(p))
    print(json.dumps(dict(run=str(run),node=a.node,job=a.job,points=cfg['expected_points'],frozen_files=len(spec['frozen_inputs']))))

if __name__=='__main__':main()
