"""CPU-only extraction of historical communication evidence, kept by cohort."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
SAVED = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
sources = []


def source(path):
    path = path.resolve()
    sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return path


def read(path):
    return json.loads(source(path).read_text())


def csv_rows(path):
    return list(csv.DictReader(source(path).open()))


def target(rows):
    def avg(key):
        return st.fmean(r['target']['exclusive_ms'][key] for r in rows)
    d = dict(dispatch=avg('deepep_dispatch'), combine=avg('deepep_combine'),
             TP=avg('nccl_tp'))
    d['dispatch_combine'] = d['dispatch'] + d['combine']
    d['all_internal_communication'] = st.fmean(r['target']['communication_ms'] for r in rows)
    return d


def main():
    ep = read(EXPERIMENTS/'glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/runs/paired_r1/ep8_mtp_fixed/stage_activity_summary.json')
    pp = read(EXPERIMENTS/'glm47_pp_stage_balance_20261023/runs/balance_r1/pp2ep4_balanced/stage_activity_summary.json')
    assert ep['passed'] and pp['passed'] and all(r['passed'] for r in ep['ranks']+pp['ranks'])
    e = target(ep['ranks'])
    stages = [target([r for r in pp['ranks'] if r['rank']//4 == s]) for s in (0, 1)]
    p = {k: sum(s[k] for s in stages) for k in e}
    prefill_rows = csv_rows(EXPERIMENTS/'glm47_a100_ep8_pp_attribution_analysis_20261010/root_layer_comparison.csv')
    prefill = {}
    for arm in ('ep8', 'pp2ep4'):
        rows = [r for r in prefill_rows if r['arm'] == arm]
        assert len(rows) == 3 and all(int(r['local_new_tokens']) == 1024 for r in rows)
        n = sum(int(r['sampled_rank_layer_calls']) for r in rows)
        prefill[arm] = {k:sum(float(r[k])*int(r['sampled_rank_layer_calls']) for r in rows)/n
                        for k in ('deep_ep.dispatch', 'deep_ep.combine', 'layer_ms')}
    width = csv_rows(SAVED/'deepep_ce_ep8_admission_20261010_r1/summary.csv')
    ce = csv_rows(SAVED/'deepep_prefill_pcie_optimization_20261010/full_metrics.csv')
    ce = [r for r in ce if r['run'] == 'deepep_prefill_pcie_opt_20261010_r7']
    report = [
        '# EP8 与节点内 EP4／PP2×EP4：通信优势历史证据', '',
        '只读取已完成实验；没有申请资源或运行新 GPU 测试。各组保留独立 workload、native、拓扑和计时协议，不跨组相加。', '',
        '## 1. 当前 GLM MTP Decode 的目标 Verify 通信', '',
        '065／167，8×A10080GB、同型号 ConnectX-6 Dx；真实语义 B8／P4096，实际 CPU 前缀4032，Q1→64；MTP top1/steps2/Verify Q3，local Graph B4。EP8 为 Attention TP4×DP2；PP 为两个 B4 slot、Attention TP4／Expert EP4，50／42目标层。', '',
        '单位 ms／每次目标 Verify 的每卡 Graph。EP8取8卡均值；PP先取每 stage 4卡均值，再沿完整目标模型路径合计。这个合计用于核对相同逻辑层路径的组件工作量，不是一个全局同步通信窗口，不是 TPOT，也不包含 PP stage 边界激活和反馈。', '',
        '| 目标模型内部通信 | EP8 | PP stage0 | PP stage1 | PP路径合计 | 少多少 |',
        '|---|---:|---:|---:|---:|---:|']
    for key,label in [('dispatch','DeepEP Dispatch'),('combine','DeepEP Combine'),
                      ('dispatch_combine','DeepEP D+C'),('TP','NCCL TP'),
                      ('all_internal_communication','内部通信合计')]:
        report.append(f'| {label} | {e[key]:.4f} | {stages[0][key]:.4f} | {stages[1][key]:.4f} | {p[key]:.4f} | {e[key]-p[key]:.4f}（{100*(1-p[key]/e[key]):.1f}%） |')
    report += ['',
        '原生 DeepEP SHA9dc34887…；NVLink pair内走NVLink，pair外及跨节点走RDMA，PCIe threshold0、无CE。Graph分段有诊断同步，不能用这些数冒充未插桩 TPOT。EP8为此前同Bank参照，并非与50/42 PP重新配对运行；跨布局严格输出等价仍未通过。', '',
        '## 2. 真实 GLM Prefill：同本地 P／Q 的单个 MoE 层', '',
        '168／169，8×A100，C4／8192→1；实际本地 chunk1024，DeepEP source256/cap256，BF16 H5120、E160/top8。详细采样 P=1024／5120／7168、Q1024，按真实 rank-layer 调用数加权。native同为9dc34887…；pair外为RDMA，没有CE。', '',
        '| 每 MoE 层 ms | EP8 | PP内EP4 | 差值 |',
        '|---|---:|---:|---:|']
    for key,label in [('deep_ep.dispatch','Dispatch'),('deep_ep.combine','Combine'),('layer_ms','整层区间')]:
        x,y=prefill['ep8'][key],prefill['pp2ep4'][key]
        report.append(f'| {label} | {x:.4f} | {y:.4f} | {x-y:.4f} |')
    x=sum(prefill['ep8'][k] for k in ('deep_ep.dispatch','deep_ep.combine'))
    y=sum(prefill['pp2ep4'][k] for k in ('deep_ep.dispatch','deep_ep.combine'))
    report += [f'| D+C | {x:.4f} | {y:.4f} | {x-y:.4f}（{100*(1-y/x):.1f}%） |', '',
        '这些是带完整性校验的 CUDA-event 互斥流区间，包含暴露等待，不是纯wire延迟。独立GPU kernel关联得到PP／EP8整轮规范化D+C7.62895／13.21203ms。增量主要出现在接收完成阶段；不能继续把它全归给NIC或CPU proxy。实际两组expert assignments相同，节点内chunk已经对齐。', '',
        '## 3. 纯通信：固定全局 token 和路由，节点内 EP4 对跨节点 EP8', '',
        '168／169；BF16 H5120/E160/top8，native c3405cad…；NVLink pair + 本地CE PCIe + 远端RDMA。无Attention、Expert GEMM、模型或KV；eager，无CUDA Graph。指标是每次对齐rank-max D+C之后的三样本中位数。EP4的source/cap256与40QP，EP8的source/cap128与20QP并不相同，不能把差额只归因于一根链路。', '',
        '| 全局 tokens | 节点内 EP4 ms | 跨节点 EP8 ms | EP4少多少 | EP8／EP4 |',
        '|---:|---:|---:|---:|---:|']
    width_results = []
    for tokens in (8,1024):
        rows = {r['phase']:r for r in width if int(r['global_tokens']) == tokens}
        assert all(r['correctness_passed'] == 'True' and r['cuda_graph'] == 'False' for r in rows.values())
        assert len({r['global_route_sha256'] for r in rows.values()}) == 1
        x=float(rows['ep4_ce']['pair_cuda_ms_median_ms']);y=float(rows['ep8_ce']['pair_cuda_ms_median_ms'])
        width_results.append(dict(global_tokens=tokens,ep4_ms=x,ep8_ms=y))
        report.append(f'| {tokens} | {x:.4f} | {y:.4f} | {100*(1-x/y):.1f}% | {y/x:.2f}× |')
    report += ['', 'D+C独立测量，不能把Dispatch与Combine各自的rank-max中位数再相加。初始化和合成expert输出准备排除，普通打包、动态计数、CE提交、完成保护和weighted reduction计时。', '',
        '## 4. 纯通信：同一节点的 RDMA／SM PCIe／CE PCIe 实现', '',
        'node168、4×A100，H5120/E160/top8，cap256、40QP，3次warmup与3次正式样本，eager。以下全部来自同一个r7对照，不把前轮最佳值拼在一起。', '',
        '| source tokens／卡 | 全局 tokens | RDMA ms | coalesced SM PCIe ms | CE PCIe ms | RDMA／CE |',
        '|---:|---:|---:|---:|---:|---:|']
    ce_results=[]
    for tokens in (1,16,64,256):
        rows={r['phase']:r for r in ce if int(r['source_tokens']) == tokens}
        phases=('rdma_baseline','pcie_coalesced','pcie_ce_local_notify')
        values=[float(rows[phase]['pair_cuda_ms_median']) for phase in phases]
        x,old,y=values
        ce_results.append(dict(source_tokens=tokens,rdma_ms=x,sm_pcie_ms=old,ce_ms=y))
        report.append(f'| {tokens} | {tokens*4} | {x:.4f} | {old:.4f} | {y:.4f} | {x/y:.2f}× |')
    report += ['',
        '**CE大幅加速确实测到过，但不是当前MTP Decode Graph已经使用的收益。** source1下CE反而慢；动态CPU计数→CE提交的已验证接口不能直接捕获进现有Decode Graph。本组较大消息收益不能直接叠加到第一组的TPOT。', '',
        '## 如何读这些证据', '',
        '- 当前MTP模型内D+C确实少约8.78ms／完整目标Verify路径，约32.9%；包含TP后，内部通信约少9.37ms。',
        '- Prefill同本地Q的真实单层D+C少约42.2%，并得到独立GPU kernel关联支持。',
        '- 固定全局数据量的CE纯通信，EP4比EP8在两个测试点都快，但它是eager测试，不是当前Graph路径。',
        '- 以上不能直接推导PP的每输出token一定快同样毫秒：还要计PP边界／反馈／调度依赖，以及一次MTP验证推进的token数量。当前数字支持优化动机，不是通信已在端到端关键路径上全部兑现的证明。', '',
        '## 原始来源及SHA256', '']
    for s in sources:
        report.append(f"- `{s['path']}` — `{s['sha256']}`")
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
    (ROOT/'evidence.json').write_text(json.dumps(dict(
        new_GPU_experiments=False,mtp_verify=dict(ep8=e,pp_stages=stages,pp_path_sum=p),
        prefill=prefill,CE_width=width_results,CE_vs_RDMA=ce_results,sources=sources),indent=2)+'\n')
    print(json.dumps(dict(report=str(ROOT/'REPORT.md'),source_count=len(sources),
                          mtp_DC_saved_ms=e['dispatch_combine']-p['dispatch_combine']),indent=2))


if __name__ == '__main__':
    main()
