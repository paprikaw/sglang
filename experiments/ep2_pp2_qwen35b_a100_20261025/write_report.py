"""Write a short, source-linked interpretation of the measured four cells."""
import argparse
import csv
import json
import re
from pathlib import Path


def report(run):
    run=run.resolve();out=run/'analysis'
    def read(name):
        with (out/name).open() as f:return list(csv.DictReader(f))
    rows=read('full_metrics.csv');comp=read('relative_comparison.csv');parts=read('components.csv')
    txt=['# 35B A100 补测：EP／PP 决策的四个缺失点','',
        '**已完成：2 个布局 × Prefill／Decode，共 4 点、每点 3 轮、96 条正式请求计时。** '
        'Guard、完整 CPU 权重缓存复用、实际 rank 数值核验、清理检查均通过，GPU 租约已释放。','',
        '固定两张 A100 80GB PCIe（069 节点 GPU0/1，NV12），依次运行 EP2 和两槽 PP2。'
        '完整 Qwen3.6-35B-A3B BF16：40 层、每层 256 路由专家、top-8。'
        'EP 每卡 40 层／128 专家，Attention TP1×DP2；PP 每阶段 20 层／256 专家，Attention TP1，两个最多 4 请求的流水槽。','',
        'Prefill：8 请求，每条 P=0 已缓存、Q=4096 新输入、O=1 输出，整批新输入 32768。'
        'Decode：8 请求，每条真实 P=4096 前缀已在 GPU、Q=1 新输入、O=128，整批输出 1024。'
        '正式 Prefill 的 cached_tokens 全为 0，Decode 全为 4096；prompt_tokens 分别为 4096／4097。'
        '构建前缀、形状预热、加载和 profile 均在正式计时外，无 CPU KV 换入、合成 KV 或 MTP。','',
        '| 本轮 A100 35B | EP2 吞吐 | PP2 吞吐 | PP 变化 | EP／PP 平均 TTFT | EP／PP 平均 TPOT |',
        '|---|---:|---:|---:|---:|---:|']
    for phase in ('prefill','decode'):
        a=next(r for r in rows if r['arm']=='ep2' and r['phase']==phase);b=next(r for r in rows if r['arm']=='pp2' and r['phase']==phase)
        delta=(float(b['tokens_per_s'])/float(a['tokens_per_s'])-1)*100
        tpot=f"{float(a['tpot_ms_mean']):.2f}／{float(b['tpot_ms_mean']):.2f} ms" if phase=='decode' else '不适用'
        txt.append(f"| {phase.capitalize()}，{'输入' if phase=='prefill' else '输出'} token/s | {float(a['tokens_per_s']):.2f} | {float(b['tokens_per_s']):.2f} | {delta:+.2f}% | {float(a['ttft_s_mean']):.3f}／{float(b['ttft_s_mean']):.3f} s | {tpot} |")
    txt += ['', '吞吐先按每轮整批完成时间计算，再对三轮取均值。TTFT 从请求放行到首 token；TPOT 为首至末 token 的时间除以 127。'
        '完整均值／中位数／P99 与每轮范围见 [指标 CSV](full_metrics.csv)，[逐请求数据](per_request.csv) 可独立重算。'
        '每格 24 请求的 P99 是样本最大值，不代表稳定服务尾延迟。','',
        '## 与已有中等负载拼起来看','',
        '下表均为两张 GPU、8 请求、同样的 P/Q/O 形状和两槽 PP；数字是 **PP 相对 EP 的吞吐变化**。'
        'H100 是已有部署数据，A100 122B 的 EP／PP 来自原实验及两槽补测，非交错配对。','',
        '| 模型／硬件 | Prefill | Decode |','|---|---:|---:|']
    for model,gpu in [('35B BF16','A100'),('122B FP8','A100'),('35B BF16','H100'),('122B FP8','H100')]:
        d={r['phase']:float(r['pp_change_percent']) for r in comp if (r['model'],r['gpu'])==(model,gpu)}
        txt.append(f"| {model}／{gpu} | {d['prefill']:+.2f}% | {d['decode']:+.2f}% |")
    txt += ['', '![固定两卡的 EP／PP 吞吐变化](comparison.png)','',
        '## 哪些 insight 已经能看见','',
        '- **硬件与部署栈会改变选择。** 同为 35B、8 请求 Decode，A100 上 PP 高 34.09%，历史 H100 上 PP 低 25.08%；122B 也出现相同方向的反转。'
        '部署不能只按模型能否装下或 GPU 数量决定。','',
        '- **工作负载会改变选择。** H100 35B 的 Prefill 均值偏向 PP（+14.15%），Decode 明确偏向 EP（PP -25.08%）。'
        'H100 Prefill 的三轮波动较大，因此前者是均值趋势，不宜宣布稳定交叉点；A100 35B 这两个点均偏向 PP。','',
        '- **模型相关的 Expert 代价确实不同。** 本轮 35B 中，PP 的 routed Expert 单层时间没有明显上升；此前 A100 122B 两槽 PP 则从 EP 的 6.722 增到 7.637 ms（约 +13.6%），同时避免逐层专家交换。'
        '这为“通信节省要与本地 Expert 执行效率一起考察”提供证据。A100 Prefill 也从 35B 的 PP +15.18% 变成 122B 的 -5.60%，但后者差距小且 EP 有波动。','',
        '**不能把第三点写成模型大小的单变量因果证明。** 35B 是 BF16 Triton，122B 是 FP8 权重／W8A16 Marlin，层数和几何也不同。'
        '这些是模型与实现组合的观测，未隔离权重工作集或 HBM 带宽瓶颈。','',
        '## 分项证据','',
        '以下为独立 profile，单位 **ms／每卡／一次实际层调用**。Attention 包括模型注解范围内的投影，按实际 Full／Linear 层调用加权；'
        '共享专家另列。Decode 只选真正 bs=4 的 Decode，排除初始 Extend。','',
        '| 阶段／布局 | Attention | Routed Expert | Shared Expert | Dispatch | Combine |',
        '|---|---:|---:|---:|---:|---:|']
    for phase in ('prefill','decode'):
        for arm in ('ep2','pp2'):
            r=next(r for r in parts if r['phase']==phase and r['arm']==arm)
            txt.append('| '+phase+'／'+arm.upper()+' | '+' | '.join(f"{float(r[k]):.4f}" for k in ('attention.total','expert.routed','expert.shared','communication.dispatch','communication.combine'))+' |')
    txt += ['', '35B Prefill 中，EP 的 Dispatch＋Combine 约 **0.957 ms／层调用**；PP 的 Expert 为 **2.222 ms**，与 EP 的 **2.245 ms** 接近。'
        '因此该点的 PP 收益与节省逐层专家交换相符，没有出现明显 Expert 惩罚。Decode 中 EP 的 D+C 约 **0.0945 ms／层调用**，绝对值较小，仍须相对于同阶段计算判断。','',
        '**这些分项不能相加得到 TTFT／TPOT。** GPU 分项可重叠，CPU 发射与调度也有成本；PP 的阶段边界在层范围之外。'
        '已单列 [PP 边界 SendRecv 驻留](pp_boundary_residency.csv)：Prefill 两阶段为 452.67／50.82 ms／整段 profile，Decode 为 0.162／0.205 ms／整个五步窗口（包含初始 Extend）。'
        'NCCL 驻留包含等待并可与计算重叠，不等于纯传输耗时；两阶段不能相加冒充一条请求的通信时间。','',
        '8 份逐 rank trace 均完成 gzip CRC／JSON 与 SHA-256 核验。Prefill 每 rank 采到 4 次全模型 EP forward 或 8 次 PP stage forward；'
        'Decode 排除 Extend 后为 EP 每 rank 3 次、PP 每 rank 2 次。样本少、真实路由和局部微批次不同，不能解释成完全相同路由的隔离 GEMM 实验。','',
        '## 部署上的直接用法与边界','',
        '本轮可用的简单规则是：先排除装不下的布局，再用实际 GPU／后端和目标阶段上的一个代表性负载配对测 EP／PP；'
        '用 Expert 变化、Dispatch／Combine 及 PP 边界和调度解释胜负。这里已经有“同模型 Decode 的选择随部署栈反转”，没有必要为了展示 trade-off 扩大成完整大网格。','',
        '本次 A100 两组同为 Triton Attention／BF16 Triton Expert，关闭 Graph／overlap，EP 使用 DeepEP normal 和 BF16 传输。'
        '历史 H100 使用原生 DeepGEMM、FA3、Decode Graph，EP 有 low-latency／overlap，PP 的 overlap 受引擎限制；提示词内容和加载顺序也不同。'
        '**因此表中反转不是纯硬件因果结论，也不是所有参数调优后的全局最优。** '
        '本次只测 NVLink 对内两卡，未补测跨节点；跨通信域的阈值仍不能由这四点直接外推。','',
        '## 复查入口','',
        f'- [本轮冻结配置]({run}/config.json)，[拓扑]({run}/topology.txt)，[独立指标复算](validation.json)，[profile 核验](profile_validation.json)。',
        '- [分项均值](components.csv)／[逐 rank 分项](components_per_rank.csv)；[历史与本轮对齐表](moderate_comparison.csv) 含原路径和哈希。',
        f'- [原始运行目录]({run}) 保留正式测量、8 份 trace、数值资格、Guard 和清理证据。065 上的 r1 在权重缓存缺失时于模型加载前退出，无性能数据；未删除缓存或改动其原始失败记录。',
        '- [A100 122B 口径与历史 Guard 状态](../../../../experiment_summary/component_reports/Q01.md)；历史 H100 35B 原 Guard 的单卡 rank 标签问题保留原失败状态，已有独立结果与缓存复核，未写成 Guard 全通过。','']
    text='\n'.join(txt)
    text=re.sub(r'\]\(([^)]+)\)', lambda m: ']('+str((out/m[1]).resolve())+')' if not m[1].startswith(('/', 'http:', 'https:', '#')) else m[0], text)
    (out/'REPORT.md').write_text(text)
    print(out/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    report(p.parse_args().run)
