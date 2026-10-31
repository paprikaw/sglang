"""Standalone scientific figures; consumes validated CSV, never launches CUDA."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent / 'compute_v2'


def main():
    rows=list(csv.DictReader((BASE/'expert_comparison.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharex=True,layout='constrained')
    for i,model in enumerate(['glm','hy3']):
        selected=[r for r in rows if r['model']==model]
        x=[int(r['whole_batch_unique_tokens']) for r in selected]
        for j in range(2):
            ax=axes[i,j]
            ax.plot(x,[float(r['ep8_whole_ms']) for r in selected],'-o',label='EP8: whole batch',color='#168f86')
            col,label=('ep4_mean_half_ms','EP4: mean of two half batches') if j==0 else ('ep4_equal_call_ms','EP4: same unique tokens per call')
            ax.plot(x,[float(r[col]) for r in selected],'-s',label=label,color='#8b5fc6')
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.grid(True,alpha=.2)
            ax.set_ylabel('Expert core latency (ms)')
            ax.set_title(('GLM-4.7 AWQ / Humming' if model=='glm' else 'Hy3 BF16 / Triton')+'\n'+('Equal token-expert pairs per GPU' if j==0 else 'EP4 has twice the pairs per GPU'),fontsize=11)
            ax.legend(fontsize=8)
            if i==1:ax.set_xlabel('Unique tokens in whole batch')
    fig.suptitle('A100 GPU0: real single-layer Expert compute, warmed eager calls\nNo communication, Attention, or pipeline scheduling',fontsize=13)
    fig.savefig(BASE/'expert_curves.png',dpi=170)
    fig.savefig(BASE/'expert_curves.pdf')
    plt.close(fig)


if __name__=='__main__':main()
