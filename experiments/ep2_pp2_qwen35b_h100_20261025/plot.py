"""Plot paired formal metrics only; no profiler time in performance bars."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(run):
    out = run / 'analysis'
    rows = list(csv.DictReader((out/'full_metrics.csv').open()))
    cases = ['moderate','near_capacity_b16','near_capacity_b32']
    fig, axes = plt.subplots(2,3,figsize=(12,7.3))
    for ax,(phase,field,title) in zip(axes.flat,[
        ('prefill','tokens_per_s','Prefill fresh-input throughput (token/s)'),
        ('prefill','ttft_s_mean','Prefill mean TTFT (s)'),
        ('prefill','e2e_s_mean','Prefill mean end-to-end (s)'),
        ('decode','tokens_per_s','Decode output throughput (token/s)'),
        ('decode','tpot_ms_mean','Decode mean TPOT (ms)'),
        ('decode','e2e_s_mean','Decode mean end-to-end (s)')]):
        for arm,offset,color in [('ep2',-.19,'#168b80'),('pp2',.19,'#8756cc')]:
            vals=[float(next(r for r in rows if (r['case'],r['phase'],r['arm'])==(c,phase,arm))[field]) for c in cases]
            bars=ax.bar([i+offset for i in range(3)],vals,width=.37,label=arm.upper(),color=color)
            ax.bar_label(bars,labels=[f'{v/1000:.1f}k' if v>=10000 else f'{v:.1f}' if v>=10 else f'{v:.2f}' for v in vals],fontsize=8,padding=2)
        labels=[]
        for case in cases:
            r=next(r for r in rows if r['case']==case and r['phase']==phase)
            labels.append(f"{r['requests']} requests\ncached {int(r['cached_context_per_request']):,} each\n"
                          f"new {r['fresh_input_per_request']} / out {r['output_per_request']} each")
        ax.set_title(title,fontsize=10)
        ax.set_xticks(range(3),labels,fontsize=7)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.margins(y=.16)
    fig.suptitle('Qwen3.6-35B-A3B BF16 | two H100 per configuration | 3 unprofiled rounds',fontsize=12)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(out/'performance.png',dpi=160)
    fig.savefig(out/'performance.svg')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    plot(p.parse_args().run.resolve())
