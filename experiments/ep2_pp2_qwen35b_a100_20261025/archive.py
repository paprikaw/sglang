"""Archive lightweight results, preserving raw/frozen/private runtime artifacts."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
BASE=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def archive(run):
    run=run.resolve();out=run/'analysis';cfg=json.loads((run/'config.json').read_text())
    ident='qwen35b_a100_ep2_pp2_20261025_r2';target=BASE/ident
    if target.exists():raise FileExistsError('Never overwrite an experiment snapshot')
    for name in ('validation.json','profile_validation.json'):
        assert json.loads((out/name).read_text())['passed']
    target.mkdir();(target/'code').mkdir()
    names=['REPORT.md','full_metrics.csv','per_request.csv','moderate_comparison.csv','relative_comparison.csv',
        'components.csv','components_per_rank.csv','pp_boundary_residency.csv','validation.json',
        'profile_validation.json','kv_cache_evidence.json','comparison.png','comparison.pdf']
    for name in names:shutil.copyfile(out/name,target/('summary.md' if name=='REPORT.md' else name))
    for p in ROOT.glob('*.py'):shutil.copyfile(p,target/'code'/p.name)
    shutil.copyfile(ROOT/'README.md',target/'code/README.md')
    shutil.copyfile(run/'topology.txt',target/'topology.txt')
    launch={k:v for k,v in cfg.items() if k not in ('admin_key_file','model_cache','guard_scripts')}
    launch.update(original_config=str(run/'config.json'),original_config_sha256=sha(run/'config.json'),
        protected_credentials='Not copied; API key and Guard bindings remain in original private run',
        active_endpoint=None,gateway_route=None,model_processes_exited=True)
    save(target/'launch_manifest.json',launch)
    spec=json.loads((run/'dashboard_spec.json').read_text())
    qualifications=[]
    profiles=[]
    for arm in ('ep2','pp2'):
        for p in (run/arm/'qualification').glob('*.json'):
            d=json.loads(p.read_text());assert d['passed']
            qualifications.append(dict(path=str(p),sha256=sha(p),**d))
        for phase in ('prefill','decode'):
            profiles+=json.loads((run/arm/'traces'/phase/'manifest.json').read_text())['files']
    save(target/'qualification.json',dict(passed=True,ranks=qualifications))
    manifest=dict(id=ident,title='A100 Qwen3.6-35B-A3B BF16: four missing EP2/PP2 cells',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete',
        implementation=dict(engine='sglang',worktree=cfg['engine_root'],commit=cfg['engine_commit'],
            immutable_source_tag=spec['sources'][0]['tag'],dirty=False,
            cohort='Pinned canonical source, shared with the previous Qwen experiments',
            private_adapter='Native BF16 Triton Expert with DeepEP-normal interface, actual-weight numerical qualification before timing'),
        deployment=dict(id=cfg['id'],node=cfg['node'],slurm_job=cfg['job_id'],gpu_indices=[0,1],
            gpu_model='NVIDIA A100 80GB PCIe',topology='NV12 within same node',
            endpoint_during_run='127.0.0.1:'+str(cfg['port']),active_endpoint=None,gateway_route=None,
            launch_manifest=str(target/'launch_manifest.json'),rollback_endpoint=None),
        workload=dict(requests=8,formal_rounds_per_cell=3,formal_cells=4,formal_request_timings=96,
            prefill=dict(cached_context=0,new_input=4096,output=1,total_new_input=32768),
            decode=dict(cached_context=4096,new_input=1,output=128,total_output=1024),
            cache='Model-produced native GPU radix/GDN; no CPU KV restore',
            measurement='Simultaneous HTTP barrier; prefix preparation, startup, warmup and profiling excluded'),
        layouts=cfg['layouts'],model=dict(path=cfg['model_path'],revision=cfg['model_revision'],layers=40,
            routed_experts_per_layer=256,top_k=8,precision=cfg['precision']),
        engine_parameters=dict(attention='triton',expert='triton BF16',graph=False,overlap_schedule=False,
            ep_communication='DeepEP normal, BF16, native intrapair NVLink',pp_slots=2,pp_microbatch=4,
            context_length=8192,max_total_tokens=49152,page_size=64,mem_fraction_static=.90,
            chunk_budget_per_attention_instance=4096,max_mamba_cache_size_global=32),
        sources=dict(run=str(run),config_sha256=sha(run/'config.json'),dashboard_spec_sha256=sha(run/'dashboard_spec.json'),
            frozen_inputs=spec['frozen_inputs'],raw_profile_exports=profiles,
            original_failed_attempt=str(run.parent/'a100_qwen35b_ep2_pp2_r1')),
        validation=dict(guard_postcheck=True,required_full_cpu_weight_cache=True,actual_rank_numerics=True,
            exact_formal_cache_counts=True,crc_joined_exports=8,owned_cleanup=True,gpu_lease_released=True),
        key_results=dict(prefill_ep2=27644.137854733013,prefill_pp2=31841.161998232645,
            prefill_pp_gain_percent=15.182329669872674,decode_ep2=90.80418511385147,
            decode_pp2=121.75949922120984,decode_pp_gain_percent=34.09018435499001,
            insight='Real deployment choice flips versus H100 Decode; communication saving and local Expert efficiency must be considered together'),
        limitations=['3 cohorts per cell, no service-tail or global-best claim',
            'Historical GPU comparison also changes native kernels, CUDA Graph, overlap, prompts and scheduling order',
            '35B BF16 versus 122B FP8/Marlin is not a model-size-only control',
            'Profiles independent of throughput; component unions and PP residency are not additive critical-path time',
            'Two-GPU NVLink only, no new cross-node threshold measurements',
            '065 r1 stopped before model load on a required-cache miss; no formal measurements or cache deletion'],
        artifacts=names+['launch_manifest.json','qualification.json','topology.txt','code/'])
    save(target/'manifest.json',manifest)
    files=sorted(p for p in target.rglob('*') if p.is_file())
    checks={str(p.relative_to(target)):sha(p) for p in files}
    save(target/'checksums.json',checks)
    for name,digest in checks.items():assert sha(target/name)==digest
    # Append one independent registry entry; no historical entries are rewritten.
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    marker='<!-- begin '+ident+' -->'
    if marker in registry.read_text():raise RuntimeError('Registry entry already exists')
    entry=f'''\n{marker}\n### {ident}\n\nSnapshot: {target}\n\n- Complete fixed2 A100069 GPU0/1 NV12 Qwen3.6-35B-A3B BF16, full40 layers; EP2 AttentionDP2/TP1 versus PP2/EP1 20 layers/stage, two B4 slots. Same native Triton Expert and Attention, Graph/overlap off, EP DeepEP normal BF16.\n- Four cells,3 formal rounds each,96 raw request timings: cold B8 P0/Q4096/O1 Prefill EP27644.14 vs PP31841.16 (+15.18%); GPU-hot B8 P4096/Q1/O128 Decode EP90.804 vs PP121.759 (+34.09%), mean TPOT86.992 vs64.982ms. All formal cached counts exactly0/4096.\n- Independent per-rank CPU/GPU correlation profiles show 35B PP routed-Expert time approximately unchanged in Prefill, while EP D+C costs0.957ms/layer invocation. Eight complete exports joined with CRC/JSON/SHA checks; Decode uses actual B4 forwards and excludes Extend. PP SendRecv residency separately retained, not pure transfer or additive TPOT.\n- Historical moderate2GPU matrix: H10035 Decode favors EP (PP -25.08%); model/phase interactions remain configuration-cohort evidence. Hardware changes kernels/Graph/overlap;35B BF16 vs122B FP8 is not size-only causality. No cross-node threshold or global optimum claim.\n- Guard/cache/actual-weight numerical qualification,63 frozen hashes,owned cleanup and lease release pass. CPU trace-export tests3 pass.065 r1 had a pre-load cache miss; preserved with no data/cache deletion.\n\nArtifacts: full latency statistics,per-request and component CSVs,PP residency,moderate matrix,scientific PNG/PDF,short Chinese report,launch/source manifests,qualification,harness and checksums. Raw logs/traces and protected credentials remain only in the original run.\n\n<!-- end {ident} -->\n'''
    with registry.open('a') as f:f.write(entry)
    print(json.dumps(dict(archived=str(target),files=len(checks),bytes=sum(p.stat().st_size for p in files),registry_updated=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    archive(p.parse_args().run)
