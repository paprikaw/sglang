"""Expose existing equal-budget EP4 comparisons; recheck means from every round."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
SUMMARY=ROOT.parent
LAYOUTS=['ep4','two_ep2','four_ep1','two_pp2']
NAMES={'ep4':'EP4','two_ep2':'2×EP2','four_ep1':'4×单卡','two_pp2':'2×优化PP2'}

def read(p):return list(csv.DictReader(p.open()))
def write(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def get(rows,**kw):
    rs=[r for r in rows if all(r.get(k)==v for k,v in kw.items())]
    assert len(rs)==1,(kw,len(rs))
    return rs[0]
def f(x,places=3):return f'{float(x):.{places}f}'

def main():
    files=[SUMMARY/d/f for d in ['decode_pool_necessity_20261030','pd_mix_pool_20261030'] for f in ['full_metrics.csv','formal_rounds.csv']]
    decode,dround,mix,mround=map(read,files)
    metrics=[];audit=[]
    for r in decode:
        if r['layout'] not in LAYOUTS or r['status']!='measured':continue
        if r['run'] not in ('011_screen_r3','011_ep2_r4','171_reverse_r3','171_long32_r4'):continue
        rounds=[x for x in dround if all(x[k]==r[k] for k in ['run','arm','case'])]
        assert len(rounds)==int(r['rounds'])
        for metric in ['mean_request_tpot_ms','mean_request_e2e_ms','mean_request_ttft_ms','cohort_makespan_ms']:
            assert abs(statistics.mean(float(x[metric]) for x in rounds)-float(r[metric]))<1e-8
        metrics.append(dict(phase='fixed_decode',run=r['run'],node=r['node'],case=r['case'],
            layout=r['layout'],P=int(r['P']),N=int(r['N']),Q='1',O='128',cache='GPU-hot; no CPU KV restore',
            rounds=len(rounds),pp_policy='9' if r['layout']=='two_pp2' else '',
            ttft_ms=float(r['mean_request_ttft_ms']),tpot_ms=float(r['mean_request_tpot_ms']),
            e2e_ms=float(r['mean_request_e2e_ms']),makespan_ms=float(r['cohort_makespan_ms']),source=r['point_path']))
    for r in mix:
        if r['run'] not in ('011_mixed_r1','171_mixed_r1') or r['layout'] not in LAYOUTS or r['route']!='least_outstanding':continue
        assert r['arrival_valid']=='True'
        rounds=[x for x in mround if all(x[k]==r[k] for k in ['run','layout','case','route'])]
        assert len(rounds)==3 and all(x['arrival_valid']=='True' for x in rounds)
        for metric in ['all_mean_ttft_ms','all_mean_e2e_ms','makespan_ms']:
            assert abs(statistics.mean(float(x[metric]) for x in rounds)-float(r[metric]))<1e-8
        cold=r['kind']=='cold_prefill'
        metrics.append(dict(phase='cold_prefill' if cold else 'mixed_arrivals',run=r['run'],node=r['node'],case=r['case'],
            layout=r['layout'],P=0,N=int(float(r['requests'])),Q=str(int(float(r['input_tokens'])/float(r['requests']))) if cold else 'Pclass8192;Dclass256',
            O='1' if cold else 'Pclass16;Dclass256',cache='cold; verified cached_tokens=0',
            rounds=3,pp_policy=r['pp_policy'] if r['layout']=='two_pp2' else '',
            ttft_ms=float(r['all_mean_ttft_ms']),tpot_ms=None if cold else float(r['all_mean_tpot_ms']),
            e2e_ms=float(r['all_mean_e2e_ms']),makespan_ms=float(r['makespan_ms']),source=r['source']))
    assert len(metrics)==143,len(metrics)
    write(ROOT/'metrics.csv',metrics)
    sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    manifest=dict(id='qwen35b_h100_ep4_fixed_phase_comparison_20261030',status='complete',new_gpu_measurements=False,
        scope='Recomputed arithmetic means of saved fixed-layout measurements; EP4 explicitly displayed alongside EP2, single and PP2',
        model='Qwen3.6-35B-A3B BF16',gpu='H100 80GB',gpus_per_configuration=4,
        nodes=['spartan-gpgpu011','spartan-gpgpu171'],cross_node_means_combined=False,
        sources=sources,measurement_records=len(metrics),round_records_rechecked=sum(r['rounds'] for r in metrics),
        source_manifests=[str(SUMMARY/d/'manifest.json') for d in ['decode_pool_necessity_20261030','pd_mix_pool_20261030']],
        true_pd_transfer_measured=False,dynamic_allocation_measured=False)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# 固定配置对比：EP4与单卡／PP在不同P、D工况是否互有胜负', '',
        '2026-10-30 · [统一索引]('+str(SUMMARY/'README.md')+')', '',
        '**有静态互补。** 35B、同一节点四张H100预算下，冷Prefill偏单卡／PP；固定Decode小并发偏单卡／PP，大并发偏EP4。Decode的同阶段反转已经在另一节点用五轮、反向布局顺序复核。它支持继续评估动态分配，但不是已测出的动态收益。', '',
        '本次把已有实测中的EP4完整列出，并从全部正式轮次重新核对均值，没有重复运行已覆盖的GPU实验。共引用143条配置×工况×运行记录；各节点、缓存协议、PP实现与指标分别保留。单卡与纯PP统一作为EP=1类别，PP不需要独自胜过单卡。', '',
        '## 1. Prefill：EP4已经有直接对照', '',
        '以下固定011节点，旧缓存P=0，每请求输出O=1，各点三轮请求算术平均。N为全组请求数，Q为每请求新输入。这里的PP2采用已通过当前协议资格的原生policy0，不冒用固定Decode的policy9。', '',
        '| N×Q＝全组新输入token | EP4平均TTFT ms | 2×EP2 | 4×单卡 | 2×PP2 |',
        '|---|---:|---:|---:|---:|']
    for case,label in [('cold_b32_q512','32×512＝16384'),('cold_b16_q8192','16×8192＝131072'),('cold_b8_q16384','8×16384＝131072')]:
        rs=[get(metrics,phase='cold_prefill',node='spartan-gpgpu011',case=case,layout=l) for l in LAYOUTS]
        lines.append('| '+label+' | '+' | '.join(f(r['ttft_ms'],2) for r in rs)+' |')
    lines+=['',
        '三档平均首token均为2×PP2最低，但整批完成均为4×单卡最低；二者都属于EP=1类别。以N16/Q8192为例，EP4、2×EP2、4×单卡、2×PP2的整批完成分别为1030.50、1034.23、706.53、721.86 ms。171独立主扫描方向一致。TTFT是请求体验指标，整批时间是完成全部工作的指标，不能混为一个排名。', '',
        '## 2. Decode：同一缓存长度，改变并发就能反转', '',
        '固定011节点。P为每请求已有GPU缓存，Q1/O128；没有CPU KV恢复。全部为三轮逐请求平均TPOT，单位ms/token；PP2为已核验固定批次输出一致性的policy9。', '',
        '| P | N | EP4 | 2×EP2 | 4×单卡 | 2×优化PP2 |',
        '|---:|---:|---:|---:|---:|---:|']
    cases=['p4096_n8_repeated','p4096_n16_repeated','p4096_n32_repeated','p4096_n64_repeated','p256_n128_repeated','p131072_n8_repeated','p131072_n32_repeated']
    for case in cases:
        g=[r for r in metrics if r['phase']=='fixed_decode' and r['node']=='spartan-gpgpu011' and r['case']==case]
        vals=[]
        for layout in LAYOUTS:
            rs=[r for r in g if r['layout']==layout]
            vals.append(f(rs[0]['tpot_ms']) if rs else '未通过运行资格' if 'n128' in case else '容量不足')
        lines.append(f'| {g[0]["P"]} | {g[0]["N"]} | '+' | '.join(vals)+' |')
    lines+=['',
        '**最清楚的一对是P4096、N8与N64。** N8时单卡／PP均快于EP4；N64时EP4均快于单卡／PP。这不是跨缓存协议或不同模型拼出的反转。包含首token后也保留方向：N8请求E2E，EP4 964.66、单卡745.08、PP2 792.57 ms；N64则为EP4 1365.02、单卡1450.14、PP2 1534.63 ms。', '',
        '独立171节点五轮反向复测：', '',
        '| P4096，全组N | EP4平均TPOT ms | 4×单卡 | 2×优化PP2 |',
        '|---:|---:|---:|---:|']
    for case,n in [('p4096_n8_repeated',8),('p4096_n64_repeated',64)]:
        rs=[get(metrics,phase='fixed_decode',run='171_reverse_r3',case=case,layout=l) for l in ['ep4','four_ep1','two_pp2']]
        lines.append('| '+str(n)+' | '+' | '.join(f(r['tpot_ms']) for r in rs)+' |')
    lines+=['',
        'N128单卡的GDN临时显存申请失败，不能用不存在的值夸大EP收益；P131072/N32单卡有独立容量约束。后一个工况PP2请求E2E2181.57 ms、EP4 2284.38 ms，PP虽有小幅整请求优势，TPOT仍略慢。缺测、容量和指标取舍均保留，详见原运行审计。', '',
        '## 3. 固定布局下的P/D混合流量：把EP4也并列展示', '',
        '以下为011主扫描，三轮平均请求E2E，单位ms。每轮64请求；P类为Q8192/O16、D类为Q256/O256，旧缓存P=0；同质布局按未完成请求数路由。每请求在一个实例完成P和D，没有真实P→D交接。', '',
        '| P类／D类请求数 | 到达速率 请求/秒 | EP4 | 2×EP2 | 4×单卡 | 2×PP2 |',
        '|---|---:|---:|---:|---:|---:|']
    for label,composition in [('pheavy','48／16'),('balanced','32／32'),('dheavy','16／48')]:
        for rate in (16,32):
            rs=[get(metrics,phase='mixed_arrivals',node='spartan-gpgpu011',case=f'{label}_rps{rate}',layout=l) for l in LAYOUTS]
            lines.append('| '+composition+' | '+str(rate)+' | '+' | '.join(f(r['e2e_ms'],2) for r in rs)+' |')
    lines+=['',
        '这六档混合流量中，EP4尚未赢得平均请求E2E；171方向一致。D类较多、32请求/秒时，EP4整轮完成4633.24 ms略低于单卡4736.90 ms，但平均E2E仍更高。因此应表述为“固定Decode有已复测的反转，当前混合到达网格尚无明显E2E反转”。', '',
        '表中PP为主扫描微批8、policy0；后续微批16＋提前发送在三档改善TTFT，但没有改变四单卡最低平均E2E的结论。不能把主表当作全部PP实现的性能上限。[后续调优与SLO对照]('+str(SUMMARY/'pd_mix_pool_20261030/REPORT.md')+')。', '',
        '## 4. 对动态分配的含义', '',
        '静态互有胜负这一必要线索已经满足，无需先实现完整P/D分离，也无需要求PP胜过单卡。此前固定Decode零切换成本的选型代理从8.052降到7.451 ms，来自EP4与单卡之间的选择，属于本方向；它不是在线混合池收益的上界。', '',
        '应先以固定P4096、低／高Decode并发两个端点为基准，记录实际活跃batch与队列。接下来如果增加新的固定流量工况，重点是让实际Decode并发进入EP4有利区间，而不只是提高名义请求率或增加P类占比。若之后测动态分配，必须把分流导致的batch变化、缓存位置和调整开销计入，并与同预算最佳固定布局比较。', '',
        '这批Prefill与Decode分别使用冷输入和GPU热缓存，能定位部署工作区间；不能把二者差值全部归因于“阶段”这一单一因素。大模型、A100、跨节点与真实P/D迁移均不在本表结论范围。', '',
        '## 数据与复核', '',
        '- [全部143条均值与来源](metrics.csv) · [逐轮重算脚本](build_report.py) · [源CSV哈希和协议清单](manifest.json)',
        '- [固定Decode完整实验与失败审计]('+str(SUMMARY/'decode_pool_necessity_20261030/REPORT.md')+')',
        '- [冷Prefill、混合到达完整实验]('+str(SUMMARY/'pd_mix_pool_20261030/REPORT.md')+')',
        '- [单卡／PP类别与真实P/D分离的范围说明]('+str(SUMMARY/'pd_disaggregation_scope_20261030/REPORT.md')+')', '',
        '历史测量保留原时间与来源。本次仅重排EP4对照并复核算术均值，没有新增GPU数据、动态配比或P/D分离结果。', '']
    report='\n'.join(lines)
    report=re.sub(r'\]\(([^)]+)\)',lambda x: ']('+(x[1] if x[1].startswith('/') else str(ROOT/x[1]))+')',report)
    (ROOT/'REPORT.md').write_text(report)
    print(json.dumps({'measurement_records':len(metrics),'round_records_rechecked':manifest['round_records_rechecked'],'new_gpu_measurements':False}))

if __name__=='__main__':main()
