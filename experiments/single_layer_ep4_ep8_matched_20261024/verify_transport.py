"""Independent CPU audit of raw rank samples, artifacts and owned cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run):
    cfg=json.loads((run/'config.json').read_text())
    comparison=json.loads((run/'comparison.json').read_text());assert comparison['passed']
    points=comparison['points'];rank_points=samples=ce_samples=stress=distinct=rank_files=0
    for phase in cfg['phases']:
        ranks=[]
        for rank in range(phase['world']):
            f=run/phase['id']/f'result_rank{rank}.json'
            row=json.loads(f.read_text());rank_files+=1
            assert row['passed'] and row['rank']==rank and row['phase']==phase['id']
            assert row['model_loaded'] is row['expert_compute'] is row['attention_compute'] is False
            assert row['buffer_explicitly_destroyed']
            assert row['route']['ce_enabled']==phase['ce_enabled']
            assert row['route']['pcie_threshold']==phase['pcie_threshold']
            assert row['route']['actual_nic_backend']['passed']
            assert row['route']['actual_nic_backend']['actual_backend']=='cpu_host_memory'
            assert len(row['points'])==len(phase['workloads'])
            ranks.append(row)
        for i,work in enumerate(phase['workloads']):
            group=[r['points'][i] for r in ranks]
            point=next(p for p in points if p['phase']==phase['id'] and p['cohort']==work['cohort']
                       and p['global_batch']==work['global_batch'])
            expected=work['tokens']*8//phase['world']
            for g in group:
                rank_points+=1
                assert g['correctness_before']['received_assignments']==expected
                assert g['correctness_after_changed_payload']['received_assignments']==expected
                assert g['source_tokens_per_rank']*phase['world']==work['tokens']
                assert g['shape']['owner_assignments']==[expected]*phase['world']
                if g['distinct_expert_validation']:
                    assert g['distinct_expert_validation']['passed'];distinct+=1
                for check in g['skewed_changed_payload_checks']:
                    assert check['passed'];stress+=1
                for s in g['samples']:
                    samples+=1
                    if phase['ce_enabled']:
                        ce_samples+=1;c=s['ce_counter_delta']
                        assert c[0]==2 and c[1]==4 and c[2]>0 and c[3]>0 and c[4:]==[1,1]
            for key in ('pair_cuda_ms','dispatch_cuda_ms','combine_cuda_ms','pair_cpu_wall_ms'):
                maxima=[max(g['samples'][j][key] for g in group) for j in range(3)]
                assert maxima==point['metrics'][key]['samples']
                assert abs(statistics.median(maxima)-point['metrics'][key]['median_ms'])<1e-10
    cleanup=json.loads((run/'task/cleanup.json').read_text());assert cleanup['passed']
    assert len(cleanup['evidence'])==16
    for row in cleanup['evidence']:
        assert row['returncode']==0 and row['worker_exited'] and row['owned_processes_exited'] and row['gpu_idle']
        assert row['gpu_query']['passed'] and row['gpu_query']['post_exit_only']
    nics=[]
    for node in cfg['nodes']:
        nic=json.loads((run/f'{node}_nic.json').read_text())
        assert nic['vendor']=='0x15b3' and nic['device']=='0x101d'
        nics.append(dict(node=node,model=nic['device_model'],device=nic['device']))
    frozen=json.loads((run/'frozen_inputs.json').read_text())['files']
    for row in frozen:assert digest(row['path'])==row['sha256'],row['path']
    lease=json.loads((run/'task/guard_authorization.json').read_text())['lease_id']
    registry=json.loads(Path('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry/registry.json').read_text())
    assert not any(row['lease_id']==lease and row['state']=='active' for row in registry['leases'])
    receipt=dict(passed=True,rank_files=rank_files,rank_points=rank_points,formal_rank_samples=samples,
                 ce_formal_rank_samples=ce_samples,skewed_payload_rank_checks=stress,distinct_expert_rank_checks=distinct,
                 real_points=len(points),aligned_samples=len(points)*3,source_hashes_unchanged=len(frozen),
                 cleanup_proofs=len(cleanup['evidence']),lease_released=True,nics=nics,
                 scope='Communication-only; no model, Attention or expert GEMM; statistics independently recomputed')
    assert (rank_files,rank_points,samples,ce_samples,stress,distinct)==(64,1280,3840,1920,1024,256)
    (run/'independent_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    verify(p.parse_args().run.resolve())
