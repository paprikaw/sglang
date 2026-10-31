"""CPU-only, sanitized visualization of the qualified traces and production cohorts."""
import csv
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent

def grouped_cycles(ranks):
    out=[]
    for r in ranks:
        for g in r['same_request_group_cycles']:
            out.extend([g['mean']]*g['cycles'])
    return {k:st.fmean(x[k] for x in out) for k in out[0]}

def main():
    evidence=json.loads((ROOT/'causal_evidence.json').read_text())
    current=evidence['current']; traces=evidence['flow']
    colors={'ep8':'#128b80','a':'#7052bd','b':'#ba83dc','prefill':'#e5a13b','gap':'#d7dde5','draft':'#e5a13b'}
    fig,axes=plt.subplots(3,1,figsize=(11.5,10.2),gridspec_kw={'height_ratios':[1,1,1.05]},constrained_layout=True)
    ax=axes[0]
    series=[('EP8 (all 8 requests)','ep8',range(8),colors['ep8']),('PP: slot A (4 requests)','pp2ep4',range(4),colors['a']),('PP: slot B (4 requests)','pp2ep4',range(4,8),colors['b'])]
    # Token events use the common client submission origin within each cohort.
    for y,(label,arm,indices,color) in enumerate(series):
        source=Path(current['arms'][arm]['cohorts'][0]['source']['path'])
        rows=json.loads(source.read_text())['requests']; n=min(len(rows[i]['token_events']) for i in indices)
        event_times=[st.fmean(rows[i]['token_events'][j][1] for i in indices)*1000 for j in range(min(n,5))]
        ax.plot(event_times,[2-y]*len(event_times),'-o',color=color,label=label,lw=2,ms=6)
        ax.annotate(f'first gap {event_times[1]-event_times[0]:.1f} ms',((event_times[0]+event_times[1])/2,2-y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
    ax.set_yticks([2,1,0],[x[0] for x in series]); ax.set_xlim(350,1050);ax.set_ylim(-.5,2.5)
    ax.set_title('1. First emissions: PP slot A starts earlier, then waits longer',loc='left',weight='bold')
    ax.set_xlabel('Client elapsed time since cohort submission (ms); first production cohort')
    ax.grid(axis='x',alpha=.18)

    ax=axes[1];rank=traces['pp2ep4']['ranks'][0]
    rows=rank['initial_forward_order'][:12]
    for r in rows:
        group='a' if r['request_indices'][0]<4 else 'b'
        prefill=r['phase']=='target_prefill'; color=colors['prefill'] if prefill else colors[group]
        ax.barh(0,r['end_relative_ms']-r['start_relative_ms'],left=r['start_relative_ms'],height=.5,color=color,edgecolor='white')
        ax.text((r['end_relative_ms']+r['start_relative_ms'])/2,0,('P' if prefill else 'V')+group.upper(),va='center',ha='center',fontsize=8,color='black' if prefill else 'white')
    first_verify=next(r for r in rows if r['phase']=='target_verify')
    next_verify=next(r for r in rows if r['phase']=='target_verify' and r['index']>first_verify['index'])
    a,b=first_verify['end_relative_ms'],next_verify['start_relative_ms']
    ax.axvspan(a,b,color=colors['gap'],alpha=.7,zorder=-1)
    ax.annotate(f'{b-a:.1f} ms between forwards',((a+b)/2,.3),ha='center',fontsize=9)
    ax.set_yticks([0],['PP stage 0\nGPU rank 0']);ax.set_ylim(-.5,.7);ax.set_xlim(0,1120)
    ax.set_xlabel('Local GPU event time relative to first target forward (ms); independent diagnostic cohort')
    ax.set_title('2. Startup transition: a real exposed interval, not a steady-state 129 ms bubble',loc='left',weight='bold')
    ax.legend(handles=[Patch(color=colors['prefill'],label='P: 65-token/request residual Prefill'),Patch(color=colors['a'],label='VA: slot A Verify'),Patch(color=colors['b'],label='VB: slot B Verify')],loc='upper right',fontsize=8,ncol=3)

    ax=axes[2]
    summary={
      'EP8':grouped_cycles(traces['ep8']['ranks']),
      'PP stage 0':grouped_cycles(traces['pp2ep4']['ranks'][:4]),
      'PP stage 1':grouped_cycles(traces['pp2ep4']['ranks'][4:])}
    names=list(summary);bottom=[0]*3
    for key,label,color in [('target_graph_ms','Target Graph',colors['ep8']),('draft_graph_ms','Draft Graphs',colors['draft']),('uncovered_current_stream_ms','Outside these Graphs',colors['gap'])]:
        values=[summary[n][key] for n in names]
        ax.barh(names,values,left=bottom,color=color,label=label,height=.55)
        for i,v in enumerate(values):
            if v>3:ax.text(bottom[i]+v/2,i,f'{v:.1f}',ha='center',va='center',fontsize=9)
        bottom=[a+b for a,b in zip(bottom,values)]
    for i,n in enumerate(names):ax.text(summary[n]['cycle_ms']+1,i,f"{summary[n]['cycle_ms']:.1f} ms",va='center',fontsize=9)
    ax.invert_yaxis();ax.set_xlim(0,max(x['cycle_ms'] for x in summary.values())+18)
    ax.set_xlabel('Same-request-group Verify start to next Verify start (ms); PP includes both slots')
    ax.set_title('3. Diagnostic cycle means: PP exposes more time outside the model Graphs',loc='left',weight='bold')
    ax.legend(loc='lower right',ncol=3,fontsize=8);ax.grid(axis='x',alpha=.16)
    fig.suptitle('GLM-4.7-AWQ | 8xA100 | semantic P4096, B8, Q1 -> 64 | MTP Q3',fontsize=13,weight='bold')
    fig.text(.5, -.008, 'Diagnostic whole-cohort wall change: EP8 +2.34%, PP +0.23%. Outside Graphs is not whole-GPU idle. Per-rank clocks are not aligned.', ha='center', fontsize=8)
    fig.savefig(ROOT/'startup_causality.png',dpi=165,bbox_inches='tight')
    fig.savefig(ROOT/'startup_causality.pdf',bbox_inches='tight')
    (ROOT/'steady_cycles.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (ROOT/'production_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['configuration','cohort','throughput_out_tok_s','wall_s','ttft_ms','tpot_ms','e2e_ms','first_transition_ms','first_transition_tpot_contribution_ms','remaining_tpot_contribution_ms'])
        writer.writeheader()
        for arm,v in current['arms'].items():
            for i,c in enumerate(v['cohorts']):
                writer.writerow(dict(configuration=arm,cohort=i+1,throughput_out_tok_s=c['throughput_out_tok_s'],wall_s=c['wall_s'],**{k:c['mean'][k] for k in writer.fieldnames if k in c['mean']}))
    print(json.dumps({'figure':str(ROOT/'startup_causality.png'),'steady_cycles':summary}))

if __name__=='__main__': main()
