"""Plot independently measured communication and Expert latencies together."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent/'component_comparison'
rows=list(csv.DictReader((BASE/'communication_expert_comparison.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
for i,model in enumerate(['glm','hy3']):
    for j,boundary in enumerate(['whole_vs_half','equal_call']):
        selected=[r for r in rows if r['model']==model and r['boundary']==boundary]
        x=[int(r['global_batch_tokens']) for r in selected];ax=axes[i,j]
        for ep,color in [('ep8','#168f86'),('ep4','#8b5fc6')]:
            ax.plot(x,[float(r[f'{ep}_observed_faster_comm_ms']) for r in selected],
                    '-o',color=color,label=ep.upper()+' communication, observed min')
            ax.plot(x,[float(r[f'{ep}_expert_ms']) for r in selected],
                    '--s',color=color,label=ep.upper()+' Expert compute')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.grid(True,alpha=.22)
        ax.set_xticks([8,32,128,512,2048,4096],labels=['8','32','128','512','2048','4096'])
        ax.set_title(('GLM-4.7 AWQ / Humming' if model=='glm' else 'Hy3 BF16 / default Triton')+'\n'+
                     ('EP8 whole / EP4 half: equal pairs per GPU' if j==0 else 'Same tokens per call: EP4 has 2x pairs per GPU'))
        ax.legend(fontsize=8)
        if j==0:ax.set_ylabel('Latency per call (ms)')
        if i==1:ax.set_xlabel('Unique tokens in whole batch' if j==0 else 'Unique tokens per call')
fig.suptitle('A100: communication and single-layer Expert measured separately\nWarmed eager; no Attention or pipeline scheduling',fontsize=14)
fig.savefig(BASE/'communication_expert_curves.png',dpi=170)
fig.savefig(BASE/'communication_expert_curves.pdf')
plt.close(fig)
