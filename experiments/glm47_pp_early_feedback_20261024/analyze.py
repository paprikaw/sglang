"""Compare the qualified same-PP baseline and an early-feedback screening run."""
import argparse,importlib.util,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
from artifact_visibility import wait_for_stable_json
spec=importlib.util.spec_from_file_location('causal_analysis',ROOT.parent/'glm47_pp_ep8_startup_causality_20261023/analyze.py')
causal=importlib.util.module_from_spec(spec);spec.loader.exec_module(causal)

def read(path):return wait_for_stable_json(path)[0]

def metrics(path):
    cohorts=[causal.cohort(path/name) for name in ('p4096_decode_results.json','p4096_decode_repeat_results.json')]
    return dict(cohorts=cohorts,mean={key:statistics.fmean(c['mean'][key] for c in cohorts) for key in cohorts[0]['mean']},
                throughput_out_tok_s=statistics.fmean(c['throughput_out_tok_s'] for c in cohorts),
                wall_s=statistics.fmean(c['wall_s'] for c in cohorts))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    cfg=read(run/'config.json');result=read(run/'task/results.json');cleanup=read(run/'task/cleanup.json')
    if not result.get('passed') or not cleanup.get('passed'):raise RuntimeError('Incomplete, invalid or unclean run')
    arm='pp2ep4_early';baseline=Path(cfg['source_layout_references'][arm]['production']['path']).parent
    rows={'baseline':metrics(baseline),'candidate':metrics(run/arm)}
    old,new=rows['baseline'],rows['candidate']
    change={key:100*(new['mean'][key]/old['mean'][key]-1) for key in ('ttft_ms','tpot_ms','e2e_ms','first_transition_ms','middle_emission_cycle_ms')}
    change['throughput_percent']=100*(new['throughput_out_tok_s']/old['throughput_out_tok_s']-1)
    parity=read(run/'same_layout_reference_output_parity.json')
    summary=dict(passed=True,rows=rows,relative_change_percent=change,exact_same_pp_output_parity=all(v['passed'] for v in parity.values()),
        causal_flow=causal.flow(run/arm),historical_baseline=True,fresh_paired_confirmation_required=True,
        scope='Same GPU nodes/jobs, semantic Bank, model/kernel/transport, B8/twoB4slots P4096 Q1->64; scheduling-only candidate',
        sources={'config':causal.descriptor(run/'config.json'),'result':causal.descriptor(run/'task/results.json'),
                 'cleanup':causal.descriptor(run/'task/cleanup.json')},artifact_visibility=causal.VISIBILITY)
    target=run/'screening_comparison.json';target.write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# PP early-feedback screening','','Historical same-node baseline; a small delta requires fresh paired confirmation.','',
           '| PP2 x EP4 | Output tok/s | TTFT ms | TPOT ms | E2E ms |','|---|---:|---:|---:|---:|']
    for name,row in rows.items():
        m=row['mean'];lines.append(f"| {name} | {row['throughput_out_tok_s']:.3f} | {m['ttft_ms']:.3f} | {m['tpot_ms']:.3f} | {m['e2e_ms']:.3f} |")
    lines+=['','Exact same-layout output parity passed. Native CPU4032/GPU0 at timer start; source import/warmup excluded; native H2D included.',
            'The separate flow diagnostic has no new per-step GPU fence; its timeline is not an unprofiled latency measurement.']
    (run/'SCREENING.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparison=str(target),relative_change_percent=change,exact_output_parity=summary['exact_same_pp_output_parity'])))
if __name__=='__main__':main()
