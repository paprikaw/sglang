"""Render the short Chinese report from audited arithmetic means."""
import csv
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
ANALYSIS=ROOT/'analysis/final'
HUMAN=WS/'experiments/experiment_summary/pd_mix_pool_20261030'

def rows(name):return list(csv.DictReader((ANALYSIS/name).open()))
def select(data,**keys):
    found=[r for r in data if all(r[k]==str(v) for k,v in keys.items())]
    assert len(found)==1,(keys,len(found))
    return found[0]
def num(row,key,digits=2):return f'{float(row[key]):.{digits}f}'

def main():
    d=json.loads((ANALYSIS/'decision.json').read_text())
    assert d['measured_points']==135 and not d['e2e_pp_wins']
    m=rows('full_metrics.csv');s=rows('slo_sensitivity.csv')
    lines=['# 35B：Prefill／Decode 混合负载是否支持 EP／PP 动态配比', '',
      '2026-10-30 · [返回全部实验索引](../README.md) · [Expert／通信权衡专题](../ep_pp_tradeoff/REPORT.md)', '',
      '**结论：混入 Prefill 后，PP 的首 token 优势是真实的；当前证据仍不足以支撑“PP／EP 动态配比有独有净收益”。** 两台 H100 节点各自的六档混合负载中，四个单卡副本都取得最低平均整请求时间。严格延迟目标下曾出现的混合池优势，在关键工况加入单卡请求分流对照后消失。', '',
      '建议暂缓35B单节点的PP／EP动态配比控制器，把可检验的问题收窄为：如何避免长输入请求干扰长输出请求，同时让空闲容量能借给另一类请求。后者目前只测了固定分流，没有实现动态借用。', '',
      '## 1. 本次新增测了什么', '',
      '完整 Qwen3.6-35B-A3B，BF16权重／KV、FP32 GDN。每个候选固定同一节点四张H100 80GB；011与171分别比较，均值不跨节点合并。候选为EP4、2×EP2、4×单卡、2×PP2、EP2＋PP2。EP的Attention为对应DP、TP1；PP2按20＋20层划分。', '',
      '- 复用已有固定批次Decode、四卡缓存重放、Expert／通信组件实验；没有重跑历史网格。[已有证据与增量](EXISTING_EVIDENCE.md)',
      '- 新增三档冷Prefill校准，以及六档真实到达的混合流量；每点独立预热，三轮完整无profiler测量。',
      '- **P类请求**：新输入8192、输出16；**D类请求**：新输入256、输出256。两类请求各自都完整执行P和D。这不是P/D分离，也没有把某条请求的状态搬到另一个实例。',
      '- 每轮64请求，按16或32请求/秒均匀到达；P类占48、32或16个，其余为D类。旧缓存P=0，首输入token唯一，每轮flush；逐请求验证缓存命中为0。',
      '- 同质副本按当前未完成请求数路由；混合池另测P类送PP、P类送EP两种固定分流。补充对照在同质池内也按请求类别分流，仍保持整请求粘性。', '',
      '| 混合负载 | P类／D类请求数 | 全轮新增输入token | 全轮输出token | 到达速率 |',
      '|---|---:|---:|---:|---|',
      '| P较多 | 48／16 | 397312 | 4864 | 16、32请求/秒 |',
      '| 均衡 | 32／32 | 270336 | 8704 | 16、32请求/秒 |',
      '| D较多 | 16／48 | 143360 | 12544 | 16、32请求/秒 |', '',
      '**计时口径：** TTFT从客户端实际发出请求到首token；E2E到最后token；TPOT是每请求首token之后的生成时间除以O−1。下表均为三轮请求算术平均，绝对毫秒；完整流量时间从计划到达起点到最后请求完成。请求平均延迟和整轮吞吐是不同指标。', '',
      '## 2. PP 首 token 更快，但没有同时降低整请求时间', '',
      '以下固定011节点。PP采用补测中通过受控输出一致性核验的提前发送policy1、每槽微批上限16；单卡复用相同协议的主扫描基线，均按未完成请求数路由。', '',
      '| 工作负载 | 4×单卡平均TTFT ms | 2×PP2平均TTFT ms | 4×单卡平均E2E ms | 2×PP2平均E2E ms |',
      '|---|---:|---:|---:|---:|']
    labels={'pheavy_rps16':'P较多，16请求/秒','pheavy_rps32':'P较多，32请求/秒','dheavy_rps32':'D较多，32请求/秒'}
    for case,label in labels.items():
        single=select(m,run='011_mixed_r1',layout='four_ep1',case=case,route='least_outstanding')
        pp=select(m,run='011_controlled_r3',layout='two_pp2',case=case,route='least_outstanding')
        lines.append(f'| {label} | {num(single,"all_mean_ttft_ms")} | {num(pp,"all_mean_ttft_ms")} | {num(single,"all_mean_e2e_ms")} | {num(pp,"all_mean_e2e_ms")} |')
    lines+=['',
      'D较多、32请求/秒时，PP的D类平均TPOT为17.12 ms，单卡为12.63 ms；首token较早没有抵消随后256-token生成的成本。这里显示的是实测整体取舍，尚不能把差额全部归因于Expert或通信。', '',
      '主扫描的原生PP微批上限8使高并发排队更严重。增大至16并开启提前发送后，D较多这档PP的TTFT从1072.79降到110.70 ms，但E2E只从3756.68降到3685.15 ms；四单卡为2639.24 ms。两个改动一起补测，不能分别归因。P较多32请求/秒时，原生PP的E2E为1735.80 ms，反而低于新设置的2347.99 ms，因此不能把更快TTFT称为所有指标都优化了。', '',
      '作为独立复核，171节点主扫描也在全部六档负载上由四单卡取得最低平均E2E；例如P较多32请求/秒为1732.41 ms，2×PP2为1754.24 ms，差距很小但未反转。171没有重复微批16与请求隔离补测；不把011的补测当成171的测量。', '',
      '## 3. 严格 SLO 下的优势，是否必须靠 PP？', '',
      '关键工况为P较多、32请求/秒；每条请求同时满足TTFT≤200 ms和平均TPOT≤20 ms才算达标。达标吞吐定义为每轮达标请求数／整轮完成时间，再对三轮求平均。', '',
      '| 四卡配置与路由 | 平均达标请求数／64 | 达标请求/秒 |',
      '|---|---:|---:|']
    slos=[('011_mixed_r1','four_ep1','least_outstanding','4×单卡，按未完成请求数'),
          ('011_mixed_r1','mixed','p_to_ep','EP2＋PP2：P类送EP、D类送PP'),
          ('011_controlled_r3','two_pp2','phase_pools','2×PP2：两个实例分别服务P类／D类'),
          ('011_controlled_r3','two_ep2','phase_pools','2×EP2：两个实例分别服务P类／D类'),
          ('011_arrival_recheck_r4','four_ep1','phase_pools','4×单卡：两卡服务P类、两卡服务D类')]
    for run,layout,route,label in slos:
        row=select(s,run=run,layout=layout,route=route,case='pheavy_rps32',ttft_slo_ms=200,tpot_slo_ms=20)
        assert row['arrival_valid']=='True'
        lines.append(f'| {label} | {float(row["good_fraction"])*64:.2f} | {num(row,"goodput_req_s",3)} |')
    lines+=['',
      '**关键发现：分开服务两类请求，就能解释这档曾归于异构混合池的优势。** 单卡隔离与该EP＋PP分流均每轮有16个D类请求达标，48个P类没有达标；单卡隔离完成整轮更快。因此这不是整体服务能力提高了数倍，也不是P/D状态分离的验证。固定分流还会损失容量共享，不能直接当最终调度策略。', '',
      '预先列出的16组SLO门槛逐项保留：011的三档有隔离补测工况中，PP在48个门槛组合均没有超过最好的非PP候选；其余三档仅有一个约1%的PP优势点，尚未做隔离复核。171主扫描仍有4个门槛点偏PP，其中包括很低的非PP达标基线；该节点没有隔离补测，不能宣称全部消失，也不能把它们当成独有机制收益。[全部SLO比较](slo_comparison.csv)', '',
      '## 4. 冷 Prefill 单独看，也有指标取舍', '',
      '下表为011主扫描，P=0、O=1。PP是原生policy0，冷Prefill微批按B/4设置，不能套用混合流量微批上限8／16的解释。', '',
      '| B×每请求Q＝全组新输入token | 4×单卡平均TTFT ms | 2×PP2平均TTFT ms | 4×单卡整批完成 ms | 2×PP2整批完成 ms |',
      '|---|---:|---:|---:|---:|']
    for case,label in [('cold_b32_q512','32×512＝16384'),('cold_b16_q8192','16×8192＝131072'),('cold_b8_q16384','8×16384＝131072')]:
        single=select(m,run='011_mixed_r1',layout='four_ep1',case=case,route='least_outstanding')
        pp=select(m,run='011_mixed_r1',layout='two_pp2',case=case,route='least_outstanding')
        lines.append(f'| {label} | {num(single,"all_mean_ttft_ms")} | {num(pp,"all_mean_ttft_ms")} | {num(single,"makespan_ms")} | {num(pp,"makespan_ms")} |')
    lines+=['',
      'PP在这些冷Prefill点的平均首token更早，但整批完成略慢。因此“平均请求等待更短”与“全部工作更快做完”都要保留。历史CPU缓存恢复Prefill的PP优势属于另一协议，不能叠加进本轮冷请求结果。', '',
      '## 5. 实现、有效性与可复现范围', '',
      '- 主扫描两节点各57点，布局顺序不同；011补测21点，再独立重测1点。总记录136点／408轮／22032请求。正式选型使用**135点／405轮／21840请求**。暖机与数值资格不计入正式轮次。',
      '- 原单卡隔离点的一轮客户端发送延迟106.08 ms，超过预设100 ms门槛，因此整个三轮点被排除。独立同配置三轮复测全部低于0.5 ms，替代后进入结论；原记录、两轮正常样本也完整保留，没有挑选拼接。[有效性与复测](ARRIVAL_RECHECK.md)',
      '- PP初次在自由到达下的输出序列不完全相同，无法隔离批次组成影响，故保守使用policy0。后续控制初始接纳顺序和不等输入／输出长度，两个PP布局逐token一致后启用policy1；正式自由到达测量记录实际提前发送。未使用仅适用于稳定固定批次的policy9。',
      '- EP使用DeepEP AUTO：Prefill Normal、Decode LL；四卡均在同节点NVLink。Decode CUDA Graph、FA3 Attention、Triton GDN与相同的小token Expert回退。保存Python可见dispatch类型与forward计数；Graph回放不逐次进入Python，这些计数不是完整通信操作数或GPU耗时。',
      '- 所有候选共享显存预算规则，PP按层数换算；实际原生容量随配置保存。此次固定64请求流量，不是持续负载容量极限。没有真实生产流量分布、突发轨迹、SLO置信区间或所有PP优化上限的结论。',
      '- 尚未测在线实例增减、原地EP↔PP权重重分片、P→D交接、KV／GDN迁移。实验支持的是这一模型／硬件／冷混合协议内的机会判断，不排除更大模型、长缓存恢复或其他通信拓扑出现不同结果。', '',
      '推理引擎：`0f2f1990994774472b1089d2b0682f0bf926a6b4`，标签`experiment/sglang/qwen35b-decode-ep-overlap-20261027-r1-cpu`；独立CPU权重daemon为`870269702285d94bdd926a50cc29ad1a20bcf60c`。相关loader文件逐字节一致，CPU权重银行复用不代表复用KV。每运行保存冻结源码、请求规格、模型加载证据、Guard后检查、拥有进程清理与资源恢复证据。', '',
      '## 6. 数据与索引', '',
      '- [全部绝对均值表](TABLES.md) · [完整均值CSV](full_metrics.csv) · [全部正式轮次](formal_rounds.csv)',
      '- [同节点候选比较](winner_comparison.csv) · [SLO逐配置原表](slo_sensitivity.csv) · [决策与排除项](decision.json)',
      '- [冻结输入与数值审计](audit.json) · [原始运行清单](manifest.json) · [资源恢复证据](RESOURCE_RECOVERY.json)',
      '- [原始实验目录]('+str(ROOT)+') · [可复现分析代码](analysis_code/analyze.py) · [报告生成代码](analysis_code/write_report.py)',
      '- [规范快照](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pd_mix_20261030/REPORT.md)', '',
      '下一步应先确定主目标是TTFT、生成节奏，还是整请求完成；在固定预算下比较请求隔离与共享容量。若继续研究PP／EP配比，须先在加入单卡／同质隔离强基线后找到可重复的独有达标吞吐收益，再测迁移或扩缩容。当前不据此启动重配置实现。', '']
    HUMAN.mkdir(parents=True,exist_ok=True)
    for p in ANALYSIS.iterdir():
        if p.is_file():shutil.copyfile(p,HUMAN/p.name)
    report='\n'.join(lines)
    report=re.sub(r'\]\(([^)]+)\)',lambda match: ']('+(
        match[1] if match[1].startswith(('/', 'https://', 'http://', '#'))
        else str((HUMAN/match[1]).resolve()))+')',report)
    (HUMAN/'REPORT.md').write_text(report)
    print(json.dumps(dict(report=str(HUMAN/'REPORT.md'),accepted_points=d['measured_points']),ensure_ascii=False))

if __name__=='__main__':main()
