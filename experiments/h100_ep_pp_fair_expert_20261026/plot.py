"""Scientific figure from the accepted matched EP/PP measurements."""
import csv
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables'


def main():
    OUT.mkdir(exist_ok=True)
    rows=list(csv.DictReader((ROOT/'analysis/metrics.csv').open()))
    components=list(csv.DictReader((ROOT/'analysis/components.csv').open()))
    configurations=[('native','ep2','EP2\nnative','#80c5be'),
                    ('triton_small','ep2','EP2\nsmall-batch Triton','#12998c'),
                    ('native','pp2','PP2\nnative','#e6be81'),
                    ('triton_small','pp2','PP2\nsmall-batch Triton','#cf8b31')]
    def metric(mode,arm,phase):
        values=[r for r in rows if r['mode']==mode and r['arm']==arm and r['phase']==phase]
        assert len(values)==1,(mode,arm,phase,values)
        return values[0]
    def component(mode,arm,key):
        selected=[r for r in components if r['mode']==mode and r['arm']==arm and r['phase']=='decode']
        assert len(selected)==2
        return 1000*st.mean(float(r.get(key) or 0) for r in selected)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(17,5.8),gridspec_kw={'width_ratios':[1,1,1.35]})
    x=np.arange(4);colors=[r[3] for r in configurations]
    for ax,phase,title,unit in [(axes[0],'prefill','Prefill throughput','New input token/s'),
                              (axes[1],'decode','Decode throughput','Output token/s')]:
        values=[float(metric(mode,arm,phase)['throughput_tok_s']) for mode,arm,_,_ in configurations]
        errors=[float(metric(mode,arm,phase)['throughput_stdev']) for mode,arm,_,_ in configurations]
        ax.bar(x,values,color=colors,yerr=errors,capsize=3,width=.7)
        for i,v in enumerate(values):ax.text(i,v+max(values)*.065,f'{v:,.0f}',ha='center',fontsize=11)
        ax.set_ylim(0,max(v+e for v,e in zip(values,errors))*1.2)
        ax.set_xticks(x,[r[2] for r in configurations],fontsize=9)
        ax.set_title(title,fontsize=14);ax.set_ylabel(unit+' (higher is better)')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[2]
    groups=[('attention.core','Attention\ncore'),('expert.routed_gemm','Expert\nGEMMs'),
            ('communication.dispatch','Dispatch'),('communication.combine','Combine'),
            ('expert.layout_adapter','EP index\nadapter')]
    loc=np.arange(len(groups));width=.19
    for i,(mode,arm,label,color) in enumerate(configurations):
        values=[component(mode,arm,key) for key,_ in groups]
        bars=ax.bar(loc+(i-1.5)*width,values,width,color=color,label=label.replace('\n',' '))
        for bar,value in zip(bars,values):
            if value:ax.text(bar.get_x()+width/2,value+.8,f'{value:.1f}',ha='center',fontsize=7,rotation=90)
    maximum=max(component(mode,arm,key) for mode,arm,_,_ in configurations for key,_ in groups)
    ax.set_ylim(0,maximum*1.33)
    ax.set_xticks(loc,[label for _,label in groups],fontsize=9)
    ax.set_ylabel('µs per GPU per actual layer call (lower is better)')
    ax.set_title('Native Decode CUDA Graph components',fontsize=14)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.legend(fontsize=8,loc='upper right',frameon=False)
    fig.suptitle('H100 EP2 vs PP2: small-batch Expert optimization',fontsize=18,y=.98)
    fig.text(.5,.065,'Qwen3.6-35B-A3B BF16 · same NVLink GPU0/1 pair · 8 requests · 5 timed cohorts/configuration/phase',ha='center')
    fig.text(.5,.034,'Prefill: cached 0 + new 4096 → output 1. Decode: real GPU-cached 4096 + new 1 → output 128. Error bars: cohort SD.',ha='center',fontsize=9)
    fig.text(.5,.009,'Same Triton GEMM family; EP layout-adapter cost included in end-to-end timing. Components overlap and are not additive. Generation quality not evaluated.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.055,right=.99,top=.83,bottom=.22,wspace=.3)
    for suffix in ['png','pdf']:fig.savefig(OUT/f'fair_ep_pp_comparison.{suffix}',dpi=170)
    print(OUT/'fair_ep_pp_comparison.png')


if __name__=='__main__':main()
