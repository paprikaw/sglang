"""Join saved T01 communication and E06 Expert results. CPU-only, no new run."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent
OUT=BASE/'component_comparison'
TOKENS=[2**n for n in range(3,13)]


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    sources={k:BASE/v for k,v in {
        'expert_comparison':'compute_v2/expert_comparison.csv',
        'expert_rows':'compute_v2/expert_all_rows.csv',
        'expert_validation':'compute_v2/validation.json',
        'communication_comparison':'ce_rdma_r1/transport_comparison.csv',
        'communication_phases':'ce_rdma_r1/phase_metrics.csv',
        'communication_manifest':'ce_rdma_r1/manifest.json',
    }.items()}
    assert json.loads(sources['expert_validation'].read_text())['passed'] is True
    experts=read_csv(sources['expert_comparison']); comm=read_csv(sources['communication_comparison'])
    raw_expert=read_csv(sources['expert_rows']);phases=read_csv(sources['communication_phases'])
    ec={(r['model'],int(r['whole_batch_unique_tokens'])):r for r in experts}
    er={(r['model'],r['run'],int(r['whole_batch_unique_tokens']),r['comparison_case']):r
        for r in raw_expert if r['timing_mode']=='eager'}
    combined=[];phase_rows=[]
    for c in comm:
        if c['ep4_node']!='065':continue
        model=c['model'];total=int(c['global_batch_tokens']);e=ec[model,total]
        boundary=c['boundary'];half=boundary=='whole_vs_half'
        expert8=float(e['ep8_whole_ms']);expert4=float(e['ep4_mean_half_ms'] if half else e['ep4_equal_call_ms'])
        assert int(c['ep8_assignments_per_gpu'])==total
        assert int(c['ep4_assignments_per_gpu'])==total*(1 if half else 2)
        r={k:c[k] for k in ['model','boundary','global_batch_tokens','ep8_call_tokens','ep4_call_tokens',
                           'ep8_assignments_per_gpu','ep4_assignments_per_gpu']}
        r.update(expert_measured_node='065',ep4_communication_node='065',ep8_communication_nodes='065/167',
                 expert_source_run=e['run'],ep8_expert_ms=expert8,ep4_expert_ms=expert4)
        for k in ['ep8_rdma_ms','ep8_ce_ms','ep4_rdma_ms','ep4_ce_ms']:r[k]=float(c[k])
        for ep in ['ep8','ep4']:
            best=min(['rdma','ce'],key=lambda t:r[f'{ep}_{t}_ms'])
            r[f'{ep}_observed_faster_transport']=best
            r[f'{ep}_observed_faster_comm_ms']=r[f'{ep}_{best}_ms']
            r[f'{ep}_separate_component_sum_estimate_ms']=r[f'{ep}_expert_ms']+r[f'{ep}_observed_faster_comm_ms']
        r['communication_saving_ms']=r['ep8_observed_faster_comm_ms']-r['ep4_observed_faster_comm_ms']
        r['expert_extra_ms']=expert4-expert8
        r['net_component_saving_estimate_ms']=r['communication_saving_ms']-r['expert_extra_ms']
        assert math.isclose(r['net_component_saving_estimate_ms'],r['ep8_separate_component_sum_estimate_ms']-r['ep4_separate_component_sum_estimate_ms'],abs_tol=1e-12)
        r['rdma_only_net_component_saving_estimate_ms']=r['ep8_rdma_ms']-r['ep4_rdma_ms']-r['expert_extra_ms']
        r['ce_only_net_component_saving_estimate_ms']=r['ep8_ce_ms']-r['ep4_ce_ms']-r['expert_extra_ms']
        combined.append(r)

    # All actually timed calls on065 plus cross8, without averaging away slots.
    for c in phases:
        if 'local167' in c['phase']:continue
        model=c['model'];total=int(c['global_batch']);e=ec[model,total]
        q=er[model,e['run'],total,c['cohort']]
        assert c['route_sha256']==q['route_sha256']
        assert int(c['assignments_per_gpu'])==int(q['actual_token_expert_pairs_on_this_gpu'])
        phase_rows.append(dict(model=model,global_batch_tokens=total,
            call_unique_tokens=int(c['call_unique_tokens']),cohort=c['cohort'],transport=c['transport'],
            communication_phase=c['phase'],communication_world_size=int(c['world']),
            communication_nodes=c['nodes'],actual_pairs_per_gpu=int(c['assignments_per_gpu']),
            dispatch_ms=float(c['dispatch_cuda_ms_median_ms']),combine_ms=float(c['combine_cuda_ms_median_ms']),
            paired_dispatch_combine_ms=float(c['pair_cuda_ms_median_ms']),
            expert_single_gpu_ms=float(q['median_ms']),expert_node='065',expert_gpu=0,
            expert_active_experts=int(q['active_experts']),expert_resident_experts=int(q['resident_experts']),
            route_sha256=c['route_sha256'],native_sha256=c['native_sha256'],expert_source=q['source']))
    assert len(combined)==40 and len(phase_rows)==160
    write_csv(OUT/'communication_expert_comparison.csv',combined)
    write_csv(OUT/'dispatch_combine_expert_phases.csv',phase_rows)
    text=['# T01＋E06：通信与单层 Expert 完整联表','',
          '这是已测结果的CPU离线汇总，没有启动新GPU实验。全部延迟单位为**毫秒，越低越好**。', '',
          '## 先看口径','',
          '- 通信：EP8跨065／167两个A100节点；EP4只取065节点，与Expert测量硬件对齐。两节点均为ConnectX-6 Dx。167的独立EP4通信控制保留在原通信报告，不把065的Expert冒充167实测。',
          '- RDMA策略：NVLink pair内仍走NVLink，pair外及跨节点走RDMA。CE策略：pair内走NVLink，同节点pair外用Copy Engine经PCIe，跨节点仍走RDMA。不是所有链路都走同一种传输。',
          '- Expert：065的同一张A100 GPU0，加载真实一层专家权重，模拟一个代表性目的rank；GLM AWQ/Humming，Hy3 BF16/Triton默认配置。没有完整模型、Attention或KV缓存。',
          '- 全部采用预热eager计时。通信为每个正式样本对齐rank取最大值，再取3次中位数；Expert为单GPU每形状30次的中位数。**不是同一条多GPU关键路径的联合测量。**',
          '- token数均指本轮送入专家的唯一新增token，不是已缓存context。普通Decode每请求处理1个token时才等于活跃请求数；MTP验证可能每请求有多个token。',
          '- 相同整批：EP8处理整批，EP4分别处理两个半批；EP4通信和Expert分别取前后半批中位数的平均值。相同单次输入：EP4每卡实际配对数是EP8的两倍。', '',
          '## 差额怎么计算','',
          '通信节省＝EP8较小通信耗时−EP4较小通信耗时；Expert增量＝EP4计算−EP8计算；估算剩余收益＝通信节省−Expert增量。正值支持局部EP4，负值支持局部EP8。', '',
          '**“较小”只是本轮RDMA／CE两份观测的较小值，不代表实现了自动切换。三次通信样本不能证明细微差异。剩余收益只是分离组件的估算，未计Attention、PP边界、反馈、调度及重叠，不能当作完整PP的实测加速或TPOT。**','']
    for boundary,title in [('whole_vs_half','相同整批：EP8整批，对比EP4每次半批'),('equal_call','相同单次输入：EP4每卡有两倍配对行')]:
        text += [f'## {title}','']
        for model,label in [('glm','GLM-4.7 AWQ'),('hy3','Hy3 BF16')]:
            text += [f'### {label}','',
                '| '+('整批token' if boundary=='whole_vs_half' else '每次输入token')+' | EP8通信 RDMA／CE | EP4通信 RDMA／CE | Expert EP8／EP4 | 通信节省 | Expert增量 | 估算剩余收益 |',
                '|---:|---:|---:|---:|---:|---:|---:|']
            for r in combined:
                if r['model']!=model or r['boundary']!=boundary:continue
                text.append(f"| {r['global_batch_tokens']} | {r['ep8_rdma_ms']:.4f}／{r['ep8_ce_ms']:.4f} | {r['ep4_rdma_ms']:.4f}／{r['ep4_ce_ms']:.4f} | {r['ep8_expert_ms']:.4f}／{r['ep4_expert_ms']:.4f} | {r['communication_saving_ms']:.4f} | {r['expert_extra_ms']:+.4f} | **{r['net_component_saving_estimate_ms']:+.4f}** |")
            text += ['']
    text += ['## 曲线','', '![通信与Expert，分口径对比](communication_expert_curves.png)','',
        '图中的通信线每个点都取两条实测策略的较小值；完整RDMA／CE数字保留在上方表格和CSV。', '',
        '## Dispatch与Combine分项','',
        '[160行逐调用分项](dispatch_combine_expert_phases.csv)保留两种模型、两种传输、EP8整批／EP4前半批／后半批／相同单次输入，且逐行验证路由哈希和每卡实际配对数一致。', '',
        '**Dispatch、Combine分别取rank最大值及样本中位数，不能把它们独立相加来替代成对计时。** 成对通信列来自原始Dispatch＋Combine共同测量，没有用两个分项中位数拼造。', '',
        '## 来源与下载','',
        '[40行完整联表CSV](communication_expert_comparison.csv) · [分项CSV](dispatch_combine_expert_phases.csv) · [来源与核验](validation.json)', '',
        f"[T01通信原报告（含167控制）]({BASE/'ce_rdma_r1/summary.md'}) · [E06 Expert原报告（含Graph与复测控制入口）]({BASE/'compute_v2/REPORT.md'})",'']
    (OUT/'REPORT.md').write_text('\n'.join(text))
    validation=dict(passed=True,analysis_only=True,new_gpu_experiments=False,
        comparison_rows=40,matched_call_phase_rows=160,route_hash_and_assignment_checks=160,
        sources={k:dict(path=str(v),sha256=sha(v)) for k,v in sources.items()},
        excluded_ep4_control_node='167; original communication data retained; no Expert measured there',
        limits=['independently timed components','observed-min transport, not adaptive runtime',
                'one representative compute GPU versus communication critical rank','not measured PP latency'])
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,comparison_rows=40,route_and_count_checks=160,report=str(OUT/'REPORT.md'))))


if __name__=='__main__':main()
