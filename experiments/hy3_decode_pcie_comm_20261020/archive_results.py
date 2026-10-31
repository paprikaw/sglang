"""CPU-only independent result validation and archival; never launches workers."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

BASE = Path(__file__).resolve().parent
RUN = BASE / 'hy3_decode_pcie_20261020_r1'
DEST = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / RUN.name

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def main():
    cfg=json.loads((RUN/'config.json').read_text())
    data=json.loads((RUN/'comparison.json').read_text())
    cleanup=json.loads((RUN/'task/cleanup.json').read_text())
    posts=list((RUN/'dashboard_guard/checks').glob('*post.json'))
    assert len(posts)==1
    post=json.loads(posts[0].read_text())
    assert data['passed'] and cleanup['passed'] and post['passed']
    assert post['cleanup_verified'] and post['scientific_success']
    assert len(data['points'])==6
    count=0
    for phase, point in zip(cfg['phases'], data['points']):
        assert point['phase']==phase['id']
        records=[json.loads((RUN/phase['id']/f'result_rank{r}.json').read_text()) for r in range(phase['world'])]
        for record in records:
            assert record['passed'] and record['buffer_explicitly_destroyed']
            row=record['points'][0]
            assert row['route']['pcie_p2p_max_tokens']==phase['pcie_p2p_max_tokens']
            assert row['source_tokens_per_rank']==1 and not row['cuda_graph']
            assert row['correctness_before']['passed'] and row['correctness_after_changed_payload']['passed']
        for metric in ('pair_cuda_ms','pair_cpu_wall_ms','dispatch_cuda_ms','combine_cuda_ms'):
            maxima=[max(r['points'][0]['samples'][i][metric] for r in records) for i in range(cfg['iterations'])]
            reported=point['metrics'][metric]
            assert maxima==reported['per_iteration_rank_max_ms']
            assert abs(statistics.median(maxima)-reported['median_ms'])<1e-9
            assert abs(statistics.mean(maxima)-reported['mean_ms'])<1e-9
        count+=len(records)
    validation={'passed':True,'rank_results':count,'measurement_points':6,'samples_per_point':3,
                'independent_recomputation':True,'cleanup_verified':True,'scientific_success':True,
                'guard_post':str(posts[0]),'guard_post_sha256':digest(posts[0])}
    save(RUN/'validation.json',validation)
    lines=['# Hy3 / A100 Decode: local SM PCIe versus RDMA', '',
           'Pure communication on booked065/167, same accepted native c3405cad for both policies. No model, Attention, Expert GEMM or HiCache. BF16 H4096, 192 experts, top8, capacity128, one source row/rank, recv-hook API, eager/CUDA Graph off. Three warmups and three samples per case.', '',
           'Transport threshold0 uses NVLink inside pairs and RDMA outside pairs. Threshold1 enables mapped same-node cross-pair SM PCIe payload for these source1 calls; cross-node remains RDMA. Cross-pair control/notification can still use RDMA. Copy Engine is not configured. Policy and mapped-peer receipts are evidence of branch eligibility, not physical wire counters.', '',
           '| placement / policy | Dispatch median ms | Combine median ms | D+C median ms | D+C mean ms | D+C rank-max samples ms |',
           '|---|---:|---:|---:|---:|---|']
    for point in data['points']:
        m=point['metrics']; pair=m['pair_cuda_ms']
        lines.append(f"| {point['phase']} | {m['dispatch_cuda_ms']['median_ms']:.6f} | {m['combine_cuda_ms']['median_ms']:.6f} | {pair['median_ms']:.6f} | {pair['mean_ms']:.6f} | {', '.join(f'{v:.6f}' for v in pair['per_iteration_rank_max_ms'])} |")
    lines+=['', '## Limits', '',
            '- For each aligned sample take maximum across ranks, then descriptive median/mean. Dispatch, combine and pair maxima/medians are independent; do not sum the displayed segment medians.',
            '- Per-rank work is matched: EP4 global4 tokens, EP8 global8. Balanced synthetic expert routes are not recorded Hy3 routes. EP4/EP8 also differ in group width, ownership and QPs/rank (48/24).',
            '- CUDA-event API spans include native packing/control/reduction/wait and CPU enqueue gaps. They are not physical-link latency or full-model TPOT. Three samples cannot establish population tails or a stable small difference.',
            '- Exact IDs/counts/BF16 payload and identity combine checked before timing and after a changed payload. All32 rank results independently recomputed. Guard postcheck and owned cleanup pass.', '',
            f'Raw source run: {RUN}', '']
    (RUN/'summary.md').write_text('\n'.join(lines))
    if DEST.exists(): raise FileExistsError(DEST)
    DEST.mkdir(parents=True)
    for name in ('summary.md','comparison.json','full_metrics.csv','config.json','cpu_test_receipt.json','validation.json','frozen_inputs.json'):
        shutil.copy2(RUN/name,DEST/name)
    shutil.copytree(RUN/'frozen/harness',DEST/'tools',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for phase in cfg['phases']:
        for pattern in ('result_rank*.json','route_rank*.json','*_topology.json'):
            for path in (RUN/phase['id']).glob(pattern):
                target=DEST/'raw'/phase['id']/path.name
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    save(DEST/'manifest.json',{'id':RUN.name,'title':'Hy3 A100 Decode SM PCIe vs RDMA, EP4 and EP8',
          'status':'complete','engine':'deepep','run_dir':str(RUN),'config':cfg,'validation':validation,
          'metrics':'comparison.json','summary':'summary.md','old_baseline':'hy3_a100_ep4_ep8_comm_20261020_r1'})
    save(DEST/'checksums.json',{str(p.relative_to(DEST)):digest(p) for p in sorted(DEST.rglob('*')) if p.is_file()})
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    marker=f'<!-- begin {RUN.name} -->'
    if marker not in registry.read_text():
        with registry.open('a') as handle:
            handle.write(f'\n{marker}\n### {RUN.name}\n\nSnapshot: {DEST}\n\n- Complete Decode-only pure communication on065/167, same nativec340, H4096 BF16/E192/top8/capacity128/source1, recv-hook/eager. EP4 each node and cross-node EP8, threshold0 RDMA versus threshold1 SM PCIe, CEoff. Three samples; all32 ranks and Guard cleanup pass. Summary retains median/mean and raw samples, no full-model TPOT claim.\n<!-- end '+RUN.name+' -->\n')
    print(json.dumps({'saved':str(DEST),'validation':validation}))

if __name__=='__main__': main()
