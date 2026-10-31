"""Absolute numeric labels; formal latency and diagnostic costs stay separate."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis/fixed_groups'
metrics=list(csv.DictReader((OUT/'metrics.csv').open()))
profiles=list(csv.DictReader((OUT/'profiles.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(12.8,7.8),layout='constrained')
for column,model in enumerate(('35b','122b')):
    ax=axes[0,column]
    values=[float(next(r for r in metrics if (r['model'],r['arm'],r['mode'])==
        (model,arm,mode))['tpot_ms']) for arm,mode in
        [('ep2','baseline'),('pp2','baseline'),('pp2','early_activation')]]
    bars=ax.bar(['EP2','PP2 original','PP2 early send'],values,color=['#159888','#9a6bc5','#e2a04c'])
    ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.set_ylim(0,max(values)*1.18);ax.set_ylabel('TPOT (ms / output token)')
    ax.set_title(('35B BF16' if model=='35b' else '122B FP8')+' — formal Decode')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax=axes[1,column]
    categories=['First-stage Expert','Last-stage Expert','First-stage NCCL\noverlap with Expert']
    for offset,mode,label,color in [(-.18,'baseline','Original','#9a6bc5'),
                                     (.18,'early_activation','Early send','#e2a04c')]:
        def stage(rank):return next(r for r in profiles if r['model']==model and r['arm']=='pp2'
            and r['mode']==mode and f'PP-{rank}.' in r['rank_trace'])
        first,last=stage(0),stage(1)
        values=[float(first['routed_gemm_union_ms']),float(last['routed_gemm_union_ms']),
                float(first['routed_gemm_nccl_overlap_ms'])]
        bars=ax.bar(np.arange(3)+offset,values,width=.34,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set_xticks(np.arange(3),categories,fontsize=9);ax.set_ylim(0,3.35)
    ax.set_ylabel('ms / card / actual Decode Graph')
    ax.set_title('PP2 diagnostic: same requests and identical outputs')
    ax.legend();ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.supxlabel('Each PP slot has 4 requests; 8 requests total, cached4096 + new1 → output128.\n'
    'Top: unprofiled client latency. Bottom: independent GPU profile; overlap is not additive.',fontsize=10)
for suffix in ('png','svg'):
    fig.savefig(OUT/('scheduling_absolute.'+suffix),dpi=180)
print(OUT/'scheduling_absolute.png')
