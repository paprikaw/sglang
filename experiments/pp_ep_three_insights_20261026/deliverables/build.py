"""Three insights, each paired with E2E and Expert/communication charts."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
import numpy as np

ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent
OLD=EXP/'pp_ep_steady_decode_20261026'
sys.path.insert(0,str(OLD))
from deck_engine import Deck, BLUE, TEAL, INK, MUTED, FONT_ROOT

OUT=ROOT/'deliverables'
DATA=ROOT/'data'
HIST=EXP/'h100_ep2_pp2_model_weight_analysis_20261026'
SOURCES=[HIST/'full_metrics.csv',HIST/'absolute_components.csv',
    HIST/'source_manifest.json',HIST/'validation.json',
    EXP/'experiment_summary/qwen_ep2_pp2_metrics.csv',DATA/'components.csv',DATA/'components_per_rank.csv',
    DATA/'extraction_validation.json']


def rows(path):return list(csv.DictReader(path.open()))
def f(row,key):return float(row[key])
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def write_csv(path,values):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in values for k in row)))
        writer.writeheader();writer.writerows(values)


def collect():
    assert json.loads((DATA/'extraction_validation.json').read_text())['passed']
    metrics={}
    for row in rows(HIST/'full_metrics.csv'):
        if row['case']!='moderate':continue
        model='35B'if'35B'in row['model']else'122B'
        metrics[('H100',model,row['phase'],row['arm'])]=dict(e2e_s=f(row,'e2e_s_mean'),
            tpot_ms=float(row['tpot_ms_mean']or 0),throughput=f(row,'tokens_per_s'),
            requests=int(row['requests']),cached=int(row['cached_context_per_request']),
            new=int(row['fresh_input_per_request']),output=int(row['output_per_request']))
    for row in rows(EXP/'experiment_summary/qwen_ep2_pp2_metrics.csv'):
        if row['workload']!='decode'or not(row['configuration']=='EP2'or row['pipeline_slots']=='2'):continue
        arm='ep2'if row['configuration']=='EP2'else'pp2'
        metrics[('A100','122B','decode',arm)]=dict(e2e_s=f(row,'e2e_s_mean'),
            tpot_ms=f(row,'tpot_ms_mean'),throughput=f(row,'throughput'),
            requests=int(row['requests_per_round']),cached=int(row['cached_tokens_per_request']),
            new=int(row['fresh_input_tokens_per_request']),output=int(row['output_tokens_per_request']))
    components={(r['hardware'],r['model'],r['phase'],r['arm']):dict(
        expert=f(r,'expert_gemm_ms_per_layer'),exchange=f(r,'expert_exchange_ms_per_layer'),
        pipeline=f(r,'pp_communication_ms_per_layer'))for r in rows(DATA/'components.csv')}
    model_components={}
    for model in ['35B','122B']:
        for arm in ['ep2','pp2']:
            selected=[r for r in rows(HIST/'absolute_components.csv')if model in r['model']and r['phase']=='prefill'and r['arm']==arm]
            assert len(selected)==(1 if arm=='ep2'else 2)
            model_components[('H100',model,'prefill',arm)]=dict(
                expert=statistics.mean(f(r,'routed_expert_ms')for r in selected),
                exchange=statistics.mean(f(r,'dispatch_ms')+f(r,'combine_ms')for r in selected),
                pipeline=statistics.mean(f(r,'nccl_residence_ms')for r in selected)if arm=='pp2'else 0)
    for key,row in metrics.items():
        assert row['requests']==8
        assert (row['cached'],row['new'],row['output'])==((0,4096,1)if key[2]=='prefill'else(4096,1,128))
    return metrics,components,model_components


def setup():
    for weight in ['Regular','Bold']:font_manager.fontManager.addfont(str(FONT_ROOT/f'NotoSansCJKsc-{weight}.otf'))
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':11,'axes.unicode_minus':False,
        'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':MUTED,
        'axes.edgecolor':'#CED7E5','figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white'})


def finish(fig,path):
    fig.savefig(path,dpi=250,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def style(ax):
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='y',color='#E8EDF5',lw=.7);ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0)


def latency_chart(path,pairs,metrics):
    fig,axes=plt.subplots(1,2,figsize=(4.4,4.0))
    for ax,(label,prefix)in zip(axes,pairs):
        values=[metrics[(*prefix,arm)]['e2e_s']for arm in ['ep2','pp2']]
        bars=ax.bar([0,1],values,color=[BLUE,TEAL],width=.62)
        for bar,value in zip(bars,values):
            ax.annotate(f'{value:.3f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,6),
                textcoords='offset points',ha='center',fontsize=12,fontweight='bold')
        ax.set_xticks([0,1],['EP2','PP2']);ax.set_ylim(0,max(values)*1.26)
        ax.set_title(label,fontsize=13,fontweight='bold',pad=14)
        style(ax)
    fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.85,wspace=.50)
    finish(fig,path)


def component_chart(path,pairs,components,scope):
    fig,axes=plt.subplots(1,2,figsize=(7.5,4.0))
    labels=['Expert\n'+scope,'专家交换\nD + C','流水通信\nNCCL 驻留']
    keys=['expert','exchange','pipeline']
    for index,(ax,(label,prefix))in enumerate(zip(axes,pairs)):
        top=0
        for offset,arm,color in [(-.18,'ep2',BLUE),(.18,'pp2',TEAL)]:
            values=[components[(*prefix,arm)][key]for key in keys]
            bars=ax.bar(np.arange(3)+offset,values,.32,color=color,label=arm.upper())
            top=max(top,max(values))
            for bar,value in zip(bars,values):
                text=f'{value:.1f}'if scope=='模块'else f'{value:.3f}'
                ax.annotate(text,(bar.get_x()+bar.get_width()/2,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9.5)
        ax.set_xticks(np.arange(3),labels,fontsize=10)
        ax.set_ylim(0,top*1.30);ax.set_title(label,fontsize=13,fontweight='bold',pad=14)
        style(ax)
        if index==1:ax.legend(frameon=False,ncol=2,fontsize=10,loc='upper right')
    fig.subplots_adjust(left=.07,right=.995,bottom=.18,top=.85,wspace=.34)
    finish(fig,path)


def main():
    setup();OUT.mkdir(exist_ok=True)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    metrics,components,model_components=collect()
    hardware=[('A100 · Marlin',('A100','122B','decode')),('H100 · DeepGEMM',('H100','122B','decode'))]
    models=[('35B · BF16',('H100','35B','prefill')),('122B · FP8',('H100','122B','prefill'))]
    phases=[('Prefill',('H100','122B','prefill')),('Decode',('H100','122B','decode'))]
    def reduction(prefix,winner):
        loser='pp2'if winner=='ep2'else'ep2'
        return 100*(1-metrics[(*prefix,winner)]['e2e_s']/metrics[(*prefix,loser)]['e2e_s'])
    small_extra=model_components[('H100','35B','prefill','pp2')]['expert']-model_components[('H100','35B','prefill','ep2')]['expert']
    big_extra=model_components[('H100','122B','prefill','pp2')]['expert']-model_components[('H100','122B','prefill','ep2')]['expert']
    slides=[dict(name='gpu',title='GPU 不同，EP / PP 的优劣会反转',
        subtitle='122B FP8 · 2 GPU · 8 请求 · 每条缓存 4096、新输入 1、生成 128 token',
        pairs=hardware,components=components,scope='GEMM',unit='毫秒 / 卡 / 一次实际层调用',
        takeaway=f'A100：PP 延迟降低 {reduction(hardware[0][1],"pp2"):.1f}%    ｜    H100：EP 延迟降低 {reduction(hardware[1][1],"ep2"):.1f}%',
        notes='Compare saved native deployment baselines: A100 Marlin/eager and H100 DeepGEMM/Graph. E2E is the mean complete-request latency, output128. Expert is routed GEMM only, matched using CPU module/kernel correlation on A100 and native Graph on H100. D+C uses DeepEP GPU kernels; pipeline is NCCL GPU residency including wait. Per actual layer-call normalization avoids comparing 48-layer EP against 24-layer PP totals. Not an additive E2E breakdown.'),
        dict(name='model',title='模型不同，PP 额外付出的 Expert 代价不同',
        subtitle='H100 · 2 GPU · 8 请求 · Prefill：每条无旧缓存、新输入 4096、生成 1 token',
        pairs=models,components=model_components,scope='模块',unit='毫秒 / 卡 / 整批 Prefill',
        takeaway=f'PP 多出的 Expert 执行时间：35B 为 {small_extra:.1f} ms/卡，122B 为 {big_extra:.1f} ms/卡',
        notes='35B BF16 and 122B FP8 are the named native checkpoints, not a size-only ablation. E2E is mean complete-request latency across3 formal cohorts. Right chart is routed Expert module GPU kernel activity for the whole B8 Prefill cohort, rank mean; excludes Shared Expert. EP D+C and PP NCCL retained separately. Historical 122B Prefill uses ready-send enabled;35B baseline disabled. Small-model EP Prefill varies across3 rounds. Do not subtract GPU activities as exposed E2E latency.'),
        dict(name='phase',title='阶段不同，EP / PP 的选择也会改变',
        subtitle='H100 · 122B FP8 · 2 GPU · 8 请求；Prefill：0+4096→1，Decode：缓存 4096+1→128',
        pairs=phases,components=components,scope='GEMM',unit='毫秒 / 卡 / 一次实际层调用',
        takeaway=f'Prefill：PP 延迟降低 {reduction(phases[0][1],"pp2"):.1f}%    ｜    Decode：EP 延迟降低 {reduction(phases[1][1],"ep2"):.1f}%',
        notes='Same122B H100 model; phase-specific native paths. Prefill mean E2E uses output1;Decode mean E2E includes TTFT plus128 outputs. GPU comparison selects Expert GEMM for both phases and DeepEP dispatch/combine only. Actual per-layer invocation units; Prefill has larger fresh-token work than Decode. PP communication is profiled NCCL residency including wait, not wire-only latency. Formal timing is independent of profiles.')]
    deck=Deck(OUT)
    figure_data=[];latency_data=[]
    for slide in slides:
        left=figures/(slide['name']+'_latency.png');right=figures/(slide['name']+'_components.png')
        latency_chart(left,slide['pairs'],metrics)
        component_chart(right,slide['pairs'],slide['components'],slide['scope'])
        deck.slide(slide['title'],slide['subtitle'],sources=SOURCES,notes=slide['notes'])
        deck.rect(.78,1.82,4.25,4.62);deck.rect(5.26,1.82,7.30,4.62)
        deck.text('平均端到端延迟（秒）↓',1.00,2.03,3.83,.40,17,INK,True)
        deck.text('Expert '+slide['scope']+' 与通信',5.51,2.03,6.80,.40,19,INK,True)
        deck.image(left,.91,2.53,4.0,3.51)
        deck.image(right,5.41,2.53,7.00,3.51)
        deck.text(slide['unit'],5.57,6.10,6.65,.25,10.5,MUTED)
        deck.rect(.78,6.62,11.78,.40,'#E4EDF9')
        deck.text(slide['takeaway'],1.00,6.69,11.3,.26,14,INK,True)
        deck.finish_slide()
        for label,prefix in slide['pairs']:
            for arm in ['ep2','pp2']:
                latency_data.append(dict(slide=slide['name'],group=label,hardware=prefix[0],model=prefix[1],phase=prefix[2],arm=arm,**metrics[(*prefix,arm)]))
                figure_data.append(dict(slide=slide['name'],group=label,hardware=prefix[0],model=prefix[1],phase=prefix[2],arm=arm,
                    scope='routed Expert '+slide['scope'],unit=slide['unit'],**slide['components'][(*prefix,arm)]))
    result=deck.save('PP_EP_three_insights_20261026')
    write_csv(OUT/'latency.csv',latency_data);write_csv(OUT/'expert_communication.csv',figure_data)
    dump(OUT/'sources.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in SOURCES])
    dump(OUT/'build_result.json',result)
    dump(OUT/'insights.json',[dict(title=s['title'],subtitle=s['subtitle'],takeaway=s['takeaway'])for s in slides])
    (OUT/'speaker_notes.md').write_text('\n\n'.join(f"## {i+1} {s['title']}\n\n{s['notes']}"for i,s in enumerate(slides))+'\n')
    for name in ['build.py','extract.py']:
        shutil.copy2(ROOT/name,OUT/name)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
