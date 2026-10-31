"""Static scientific plots from the completed transport comparison CSV."""
import argparse
import csv
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((run/'transport_comparison.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for mi,(model,label) in enumerate([('glm','GLM geometry'),('hy3','Hy3 geometry')]):
        for bi,(boundary,title) in enumerate([('whole_vs_half','EP8 whole batch / EP4 half batch'),('equal_call','Same unique tokens per call')]):
            ax=axes[mi,bi]
            sample=[r for r in rows if (r['model'],r['boundary'],r['ep4_node'])==(model,boundary,'065')]
            x=[int(r['global_batch_tokens']) for r in sample]
            for key,name,color,style in [('ep8_rdma_ms','EP8 RDMA','#2563eb','-'),
                                         ('ep8_ce_ms','EP8 CE','#2563eb','--'),
                                         ('ep4_rdma_ms','EP4 RDMA','#ea580c','-'),
                                         ('ep4_ce_ms','EP4 CE','#ea580c','--')]:
                ax.plot(x,[float(r[key]) for r in sample],label=name,color=color,linestyle=style,marker='o',markersize=3)
            ax.set_xscale('log',base=2);ax.set_yscale('log')
            ax.set_xticks([8,32,128,512,2048,4096],['8','32','128','512','2,048','4,096'])
            ax.set_title(label+'\n'+title);ax.set_ylabel('Dispatch + Combine median (ms)')
            ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
            ax.set_xlabel('Whole-batch tokens' if boundary=='whole_vs_half' else 'Unique tokens per call')
    fig.suptitle('A100: same-native CE / RDMA within each model; EP4 on 065, EP8 on 065 + 167\nEager; 3 aligned samples; synthetic communication only, no Expert GEMM',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ('png','svg','pdf'):fig.savefig(run/f'transport_curves.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
