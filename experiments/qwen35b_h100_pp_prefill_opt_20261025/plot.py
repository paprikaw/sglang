"""Standalone scientific plots from saved metrics and correlated trace spans."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(run):
    data=json.loads((run/'analysis/summary.json').read_text())['rows']
    budgets=(4096,2048,1024,8192)
    colors=('#557fa6','#e0a04b','#9975b5','#49a486')
    cases=('short_cached32','long_cached32','cold8')
    descriptions=('32 requests; cached 4,096 each; +512 input each',
                  '32 requests; cached 89,984 each; +512 input each',
                  '8 requests; no cached prefix; +4,096 input each')
    fig,axs=plt.subplots(3,3,figsize=(13.5,9))
    for i,(case,desc) in enumerate(zip(cases,descriptions)):
        rows={r['policy']:r for r in data if r['case']==case}
        for j,(field,label,scale) in enumerate((('throughput_new_tokens_s','New input tokens / s',1),
                ('ttft_ms_mean','Mean request TTFT (ms)',1),('mean_cohort_duration_ms','Whole cohort duration (ms)',1))):
            ax=axs[i,j];vals=[rows[b][field]*scale for b in budgets]
            ax.bar(range(4),vals,color=colors,width=.68)
            for x,v in enumerate(vals):ax.text(x,v*1.015,f'{v:,.0f}',ha='center',va='bottom',fontsize=8)
            if j==2:
                for x,b in enumerate(budgets):
                    values=rows[b]['trial_durations_ms']
                    ax.plot([x-.08,x+.08],values,'k.',ms=4)
            ax.set_xticks(range(4),[f'{x:,}' for x in budgets]);ax.set_xlabel('New-token budget per PP forward')
            ax.set_ylabel(label);ax.set_ylim(0,max(vals)*1.18);ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if j==0:ax.set_title(desc,fontsize=10,loc='left')
    fig.suptitle('H100 PP2 Prefill: same process, four native chunk budgets\n'
                 'Two formal samples per budget; cached prefixes already on GPU; one output token',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    out=run/'analysis';fig.savefig(out/'chunk_comparison.svg');fig.savefig(out/'chunk_comparison.png',dpi=170);plt.close(fig)
    p=out/'profile_inventory.json'
    if p.exists():
        traces=json.loads(p.read_text())['timelines'];policies=sorted({t['policy'] for t in traces})
        fig,axs=plt.subplots(len(policies),1,figsize=(11,2.6*len(policies)),squeeze=False)
        for ax,policy in zip(axs[:,0],policies):
            rows=[t for t in traces if t['policy']==policy]
            origin=min(s['cpu_begin_us'] for t in rows for s in t['steps'])
            for t in rows:
                y=1-t['stage']
                for s in t['steps']:
                    begin=(s['gpu_begin_us']-origin)/1000;dur=(s['gpu_end_us']-s['gpu_begin_us'])/1000
                    ax.broken_barh([(begin,dur)],(y-.3,.6),facecolors=colors[0 if t['stage']==0 else 3],edgecolors='white')
                    ax.text(begin+dur/2,y,str(s['new_tokens']),ha='center',va='center',fontsize=8)
            ax.set_yticks([1,0],['Stage 0','Stage 1']);ax.set_xlabel('Milliseconds from first scheduler forward (profiled only)')
            ax.set_title(f'Chunk budget {policy}: labels are new input tokens in this forward',fontsize=11,loc='left')
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('Actual GPU spans for PP Prefill forwards (spans may contain launch gaps)',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'pipeline_timeline.svg');fig.savefig(out/'pipeline_timeline.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(plots=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();draw(a.run.resolve())
