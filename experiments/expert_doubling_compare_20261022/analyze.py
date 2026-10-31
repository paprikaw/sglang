"""Compare measured deployment kernels; never interpret them as matched precision."""
import csv,json,hashlib,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
TS=[2**p for p in range(3,14)];ES=[2**p for p in range(6)]
MODELS={'hy3':dict(label='Hy3 BF16 / Triton',resident=48,H=4096,I=1536,weight=36*2**20),
        'glm':dict(label='GLM-4.7 AWQ INT4 / Humming',resident=40,H=5120,I=1536,weight=12257280)}

def load():
    rows=[];validation={};sources={}
    for name,meta in MODELS.items():
        run=ROOT/name/'runs/r1';p=run/'profile/result.json'
        v=json.loads(p.read_text());spec=json.loads((run/'dashboard_spec.json').read_text())
        expected={(t,e,m)for t in TS for e in ES if e<=t for m in ('eager','cuda_graph')}
        assert v['passed'] and len(v['rows'])==126
        assert {(r['T'],r['E'],r['mode'])for r in v['rows']}==expected
        assert len(v['numerical'])==126 and all(x['passed']for x in v['numerical'])
        end=json.loads((run/'profile/node0_end.json').read_text())
        assert all(end[k] for k in ('worker_exited','owned_processes_exited','gpu_idle'))
        posts=list((run/'dashboard_guard/checks').glob('*-post.json'))
        assert len(posts)==1
        post=json.loads(posts[0].read_text())
        cleanup=json.loads((run/'profile/task/cleanup.json').read_text())
        assert post['passed'] and cleanup['passed']
        validation[name]=dict(numeric_checks=126,numeric_pass=True,rows=126,cleanup_verified=True,
                              gpu=v['gpu'],authorization=v['authorization'],guard_postcheck=post)
        sources[name]=dict(result=str(p),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                           spec=str(run/'dashboard_spec.json'),engine=spec['sources'][0],
                           guard=spec['sources'][1],backend=v['backend'])
        for r in v['rows']:
            stat=r['timing'];bs=r.get('block_median_ms',[])
            assert stat['n']==30 and stat['median_ms']>0
            samples=r.get('raw_ms',r.get('cuda_event_ms'))
            assert len(samples)==30 and len(bs)==3
            assert all(math.isfinite(x) and x>0 for x in samples)
            assert math.isclose(statistics.median(samples),stat['median_ms'])
            counts=r['counts'];active=[x for x in counts if x]
            assert sum(counts)==r['T'] and len(active)==r['E']
            assert max(active)-min(active)<=1
            rows.append(dict(model=name,model_label=meta['label'],T=r['T'],E=r['E'],
                rows_per_active_expert=r['T']/r['E'],resident_experts=meta['resident'],
                active_checkpoint_weight_MiB=meta['weight']*r['E']/2**20,
                useful_gemm_GFLOPs=6*meta['H']*meta['I']*r['T']/1e9,
                mode=r['mode'],samples=stat['n'],median_ms=stat['median_ms'],
                mean_ms=stat['mean_ms'],p99_ms=stat['p99_ms'],
                block_drift_percent=(max(bs)/min(bs)-1)*100 if bs else None,
                actual_padded_rows=r.get('actual_padded_rows'),
                padding_ratio=r.get('padding_ratio'),
                local_assignments_per_s=r['T']*1000/stat['median_ms'],
                source=str(p)))
    return rows,validation,sources

def main():
    rows,validation,sources=load();OUT.mkdir(exist_ok=True)
    with (OUT/'full_metrics.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    graph={(r['model'],r['T'],r['E']):r for r in rows if r['mode']=='cuda_graph'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(2,2,figsize=(13.4,11.5),layout='constrained')
    allms=[r['median_ms']for r in rows if r['mode']=='cuda_graph']
    for col,(name,meta)in enumerate(MODELS.items()):
        a=np.full((len(TS),len(ES)),np.nan)
        for i,t in enumerate(TS):
            for j,e in enumerate(ES):
                if e<=t:a[i,j]=graph[name,t,e]['median_ms']
        ratio=a/a[:,[0]]
        for row,data,title,norm in [(0,a,'Median latency (ms)',LogNorm(min(allms),max(allms))),
                                     (1,ratio,'Latency / same T with E=1',LogNorm(.5,16))]:
            ax=axes[row,col];im=ax.imshow(np.ma.masked_invalid(data),norm=norm,cmap='YlOrRd',aspect='auto')
            ax.set_xticks(range(len(ES)),ES);ax.set_yticks(range(len(TS)),TS)
            ax.set_xlabel('Active local experts E (doubling)');ax.set_ylabel('Local token-expert rows T (doubling)')
            ax.set_title(meta['label']+'\n'+title)
            for i in range(len(TS)):
                for j in range(len(ES)):
                    if math.isfinite(data[i,j]):
                        v=data[i,j];label=f'{v:.3f}'if row==0 else f'{v:.2f}x'
                        ax.text(j,i,label,ha='center',va='center',fontsize=8.5,
                                color='white'if norm(v)>.64 else'#111827')
            fig.colorbar(im,ax=ax,fraction=.035,pad=.03)
    fig.suptitle('Expert footprint surface on the same A100 GPU\nBalanced routing; warm CUDA Graph replay; 30 samples per cell; compute only',fontsize=15)
    for ext in ('png','pdf'):fig.savefig(OUT/f'expert_doubling_surface.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13.4,4.6),layout='constrained')
    for ax,(name,meta)in zip(axes,MODELS.items()):
        for e in (1,4,8,16,32):
            tx=[t for t in TS if t>=e]
            ax.plot(tx,[graph[name,t,e]['median_ms']for t in tx],marker='o',label=f'E={e}')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.grid(alpha=.2)
        ax.set_xlabel('Local token-expert rows T');ax.set_ylabel('Expert core median (ms)')
        ax.set_title(meta['label']);ax.legend(ncol=3,fontsize=9)
    fig.suptitle('Doubling token rows: where extra experts become expensive')
    for ext in ('png','pdf'):fig.savefig(OUT/f'expert_doubling_curves.{ext}',dpi=180)
    plt.close(fig)
    text=['# Expert T × E doubling comparison','',
          'One GPU (A100 GPU0, node065), one real expert layer per model. No Attention, shared experts, router, communication or model loading in timing.',
          'T means actual received local token-expert rows, not unique user tokens or request batch size. E means balanced active local experts. Resident expert counts remain fixed at Hy3=48, GLM=40.',
          'Tokens: '+', '.join(map(str,TS))+'. Active experts: '+', '.join(map(str,ES))+'. E>T cells are invalid and omitted.',
          'Both eager and CUDA Graph are measured separately, 30 CUDA-event samples/cell in 3 rotating/reversed blocks. Immediate warmups: warm-cache replay, not a full-model cold-HBM simulation.',
          '', '| Model | H | I | Parameters/expert | Checkpoint bytes/expert | Backend |', '|---|---:|---:|---:|---:|---|']
    for name,meta in MODELS.items():
        text.append(f"| {meta['label']} | {meta['H']} | {meta['I']} | {3*meta['H']*meta['I']:,} | {meta['weight']/2**20:.5f} MiB | {sources[name]['backend']} |")
    text+=['','GLM has 25% more parameters per expert; at equal BF16 it would be45 MiB vs Hy3 36 MiB. The tested GLM checkpoint is INT4 AWQ including scales/zeros, so Hy3 source weights per expert are about3.08x larger. Source bytes are a footprint proxy, not measured device/HBM traffic. Different precisions, kernels, layouts and engine revisions are intentional deployment-path comparisons; this is not a precision-controlled weight-bandwidth experiment.',
           '', '| T | E | GLM median ms (Graph) | Hy3 median ms (Graph) |', '|---:|---:|---:|---:|']
    for t in (8,32,128,512,2048,8192):
        for e in (2,8,32):
            if e<=t:text.append(f"| {t} | {e} | {graph['glm',t,e]['median_ms']:.4f} | {graph['hy3',t,e]['median_ms']:.4f} |")
    text+=['','At fixed T, ideal useful GEMM FLOPs are 6 H I T. Increasing E changes the active weight working set E*W, per-expert rows T/E, padding, scheduling and reuse; it does not multiply ideal useful FLOPs by E. No HBM/L2/occupancy counters were measured, so the E penalty cannot be assigned wholly to weight reads.',
           'Hy3 records the actual selected BLOCK_M and padded rows per cell in raw results and CSV. GLM uses the validated AWQ GROUPED_MASKED path, fixed capacity8192/expert and expected_m=ceil(T/40) within each E sweep.',
           'This local compute test alone cannot rank PP2×EP4 vs EP8 end to end. PP must also compensate for increased local expert cost, PP boundaries, critical-path scheduling and capacity-dependent batch changes with communication/capacity gains.',
           '', f"Maximum within-run block-median drift: GLM {max(r['block_drift_percent']for r in rows if r['model']=='glm'):.1f}%, Hy3 {max(r['block_drift_percent']for r in rows if r['model']=='hy3'):.1f}%. Three blocks are not three independent process launches.",
           '', 'Artifacts: full_metrics.csv, validation.json, sources.json, expert_doubling_surface.png/pdf, expert_doubling_curves.png/pdf. Raw result source paths and pinned commits are in sources.json.']
    (OUT/'summary.md').write_text('\n\n'.join(text)+'\n')
    manifest=dict(id='hy3_glm47_expert_doubling_20261022_r1',title='Hy3 vs GLM real expert layer doubling grids',status='completed',
                  sources=sources,validation=validation,workload=dict(tokens=TS,active_experts=ES,balanced=True,compute_only=True),
                  caveats=['Different actual quantization, layouts and kernels','Warm replay, no hardware memory counters','Single GPU local compute, not an end-to-end EP/PP trial'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),out=str(OUT),validation=validation),indent=2))

if __name__=='__main__':main()
