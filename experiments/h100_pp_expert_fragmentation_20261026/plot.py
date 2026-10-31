"""Generate standalone scientific figures from independently recomputed CSVs."""
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'deliverables'
OUT.mkdir(exist_ok=True)
rows = list(csv.DictReader((ROOT / 'analysis/metrics.csv').open()))
components = list(csv.DictReader((ROOT / 'analysis/components.csv').open()))
cases = [('r4_reference_fixed','ep2','EP2 native'), ('r4_reference_fixed','pp2','PP2 native'),
         ('r5_adaptive_confirm','pp2','PP2 adaptive')]
colors = ['#179a8a','#d39337','#4077c4']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42})


def metric(run, arm, phase, key):
    value = next(d for d in rows if (d['run'],d['arm'],d['phase']) == (run,arm,phase))
    assert value['guard_stage'] == 'complete'
    return float(value[key])


def bar(ax, phase, key, title, unit, scale=1):
    values = [metric(r,a,phase,key) / scale for r,a,_ in cases]
    errors = [metric(r,a,phase,'throughput_stdev') / scale for r,a,_ in cases] if key=='throughput_tok_s' else None
    ax.bar(range(3), values, color=colors, width=.62, yerr=errors, capsize=4)
    for i,v in enumerate(values):
        ax.text(i,v*1.025,f'{v:,.1f}',ha='center',va='bottom')
    ax.set_xticks(range(3),[c[2] for c in cases]); ax.set_title(title)
    ax.set_ylabel(unit); ax.set_ylim(0,max(values)*1.2); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)


fig, axes = plt.subplots(2,3,figsize=(18,10.2))
bar(axes[0,0],'prefill','throughput_tok_s','Prefill throughput','Thousand new input tokens/s',1000)
bar(axes[0,1],'prefill','request_e2e_ms','Prefill request latency','Mean end-to-end latency (ms)')
bar(axes[0,2],'decode','throughput_tok_s','Decode throughput','Output tokens/s')
bar(axes[1,0],'decode','request_e2e_ms','Decode request latency','Mean end-to-end latency (ms)')
bar(axes[1,2],'decode','tpot_ms','Decode token latency','Mean time per output token (ms)')
ax = axes[1,1]
groups = [('attention.core','Attention core'),('expert.routed_gemm','Expert GEMM'),('comm','Dispatch + Combine')]
for j,(key,label) in enumerate(groups):
    values=[]
    for run,arm,_ in cases:
        selected=[d for d in components if d['run']==run and d['arm']==arm and d['phase']=='decode']
        assert len(selected)==2
        values.append(1000*statistics.mean((float(d.get('communication.dispatch') or 0)+float(d.get('communication.combine') or 0))
                      if key=='comm' else float(d.get(key) or 0) for d in selected))
    x=np.arange(3)+(j-1)*.24
    bars=ax.bar(x,values,.22,label=label,color=['#8099ae','#c07156','#9f83bc'][j])
    ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=2)
ax.set_xticks(range(3),[c[2] for c in cases]);ax.set_title('Native CUDA Graph components')
ax.set_ylabel('Microseconds / GPU / actual layer call');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('H100: improve PP Expert execution while preserving deployment and model precision',fontsize=20,y=.985)
fig.text(.5,.018,'Full Qwen3.6-35B-A3B BF16; the same NVLink GPU pair; 8 requests. Prefill: cached0 + new4096 -> output1.\n'
         'Decode: GPU-cached4096 + new1 -> output128. Formal timing excludes warmup, profiling and validation. Local Decode batch = 4.\n'
         'Component bars are nonadditive; PP boundary waits are separate. Prefill means retain one slow adaptive cohort; whiskers = sample SD.\n'
         'Both paths use BF16; generated token sequences differ. Task quality has not been evaluated.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.12,1,.95))
for suffix in ['png','pdf']:
    fig.savefig(OUT/f'end_to_end_and_components.{suffix}',dpi=170)
plt.close(fig)

micro=list(csv.DictReader((ROOT/'analysis/microbench.csv').open()))
routing=list(csv.DictReader((ROOT/'analysis/routing.csv').open()))
fig,axes=plt.subplots(1,3,figsize=(18,5.8))
ax=axes[0]
for rank,color in [(0,'#179a8a'),(1,'#4077c4')]:
    sample=[d for d in micro if d['run']=='r4_reference_fixed' and int(d['pp_rank'])==rank and int(d['tokens'])==4]
    values=[next(float(d['graph_ms'])*1000 for d in sample if d['implementation']=='native_resident'+str(n)) for n in (32,64,128)]
    ax.plot([32,64,128],values,'o-',label=f'Real layer from PP stage {rank}',color=color)
    for x,y in zip([32,64,128],values):ax.annotate(f'{y:.1f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
ax.set_title('Inactive Expert dimension adds cost');ax.set_xlabel('Resident experts supplied to the same kernel')
ax.set_ylabel('Expert execution (microseconds)');ax.set_xticks([32,64,128]);ax.set_ylim(0,95);ax.legend(fontsize=9);ax.grid(alpha=.15)
ax=axes[1]
names=[('native','Masked DG'),('deepgemm_compact','Compact DG'),('route_gemv8','Route GEMV'),('triton','Small-tile'),('m16n64k128','Tuned tile')]
for rank,color in [(0,'#179a8a'),(1,'#4077c4')]:
    sample=[d for d in micro if d['run']=='r3_tiled' and int(d['pp_rank'])==rank and int(d['tokens'])==4]
    values=[next(float(d['graph_ms'])*1000 for d in sample if d['implementation']==name) for name,_ in names]
    bars=ax.bar(np.arange(len(names))+(rank-.5)*.36,values,.34,color=color,label=f'Stage {rank}')
    ax.bar_label(bars,fmt='%.0f',fontsize=9,padding=2)
ax.set_xticks(range(len(names)),[name for _,name in names],rotation=25,ha='right');ax.set_title('Actual-input candidate comparison')
ax.set_ylabel('Expert execution (microseconds)');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax=axes[2]
for phase,t,label,color in [('D',4,'Decode: 4 tokens','#4077c4'),('P',4096,'Prefill: 4096 tokens','#d39337')]:
    sample=[d for d in routing if d['run']=='r1_native' and d['phase']==phase and int(d['tokens'])==t]
    sample.sort(key=lambda d:int(d['layer_id']))
    ax.plot([int(d['layer_id']) for d in sample],[int(d['active_experts']) for d in sample],label=label,color=color)
ax.axhline(256,color='#777777',ls='--',lw=1,label='Resident experts: 256')
ax.set_title('Actual active experts vary by phase');ax.set_xlabel('Model layer');ax.set_ylabel('Experts / layer call');ax.set_ylim(0,275);ax.legend(fontsize=9);ax.grid(alpha=.15)
fig.suptitle('Diagnose the execution cost before changing the PP/EP placement',fontsize=19,y=.98)
fig.text(.5,.018,'Left: identical 4-token inputs, selected weights, routing and effective FLOPs; common weight-bank addresses; Expert-only CUDA Graph timing.\n'
         'Middle: actual inputs from one layer per PP stage (r3 diagnostics; later logprob validation failed). GEMV and compact layouts were slower.\n'
         'These kernel experiments explain a mechanism; end-to-end deployment results are shown in the companion figure.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.14,1,.92))
for suffix in ['png','pdf']:fig.savefig(OUT/f'fragmentation_mechanism.{suffix}',dpi=170)
plt.close(fig)
print('Figures saved to', OUT)
