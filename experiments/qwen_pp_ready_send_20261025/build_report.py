"""Build a small, provenance-preserving report after GPU cleanup."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
INDEX = ROOT.parent / 'ep8_pp2ep4_workload_index_20261024'
RUN = ROOT / 'runs/two_slots_r3'
OUT = ROOT / 'results'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join('---' for _ in headers) + '|',
        *['| ' + ' | '.join(map(str,row)) + ' |' for row in rows]])


def link(label, path):
    return f'[{label}]({path})'


def main():
    OUT.mkdir(exist_ok=True)
    data = json.loads((RUN / 'analysis/validated.json').read_text())
    checks = {kind:json.loads((RUN / f'pp2/correctness.{kind}.json').read_text())
              for kind in ('warmup','prefill','decode')}
    assert json.loads((RUN / 'task/cleanup.json').read_text())['passed']
    accepted = data['prefill_accepted']
    rows = {(r['phase'],r['policy']):r for r in data['performance']}
    gain = (data['prefill_throughput_ratio']-1)*100
    ttft = (1-data['prefill_ttft_ratio'])*100
    profile_rows = sorted(data['profiles'],key=lambda r:(r['stage'],r['policy']))
    hashes = []
    for p in sorted(RUN.rglob('*')):
        if p.is_file() and (p.suffix in ('.json','.py','.gz') or p.name=='topology.txt'):
            # Never archive authentication or model data. Manifests list only
            # public result/config/source evidence, not private Guard bindings.
            rel = p.relative_to(RUN)
            if 'dashboard_guard' in rel.parts or p.name=='preparation.json':
                continue
            hashes.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest = dict(id='qwen122_a100_pp_ready_send_20261025',source_run=str(RUN),
        source_engine_commit='870269702285d94bdd926a50cc29ad1a20bcf60c',
        implementation='experiment-private overlay; shared native engine unchanged',
        prefill_accepted=accepted,decode_accepted=data['decode_accepted'],sources=hashes)
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def triple(r,key):
        if r.get(key+'_mean') is None:return '不适用'
        return ' / '.join(f'{r[key+"_"+s]:.4f}' for s in ('mean','median','p99'))
    performance = []
    for phase in ('prefill','decode'):
        for policy,label in [('baseline','原版'),('ready','发送就绪候选')]:
            r=rows[phase,policy]
            performance.append([phase,label,f'{r["tokens_per_s"]:.2f}',triple(r,'ttft_s'),
                                triple(r,'tpot_ms'),triple(r,'e2e_s'),
                                '输出一致' if r['valid'] else '仅诊断，输出等价未通过'])
    profiles = [[r['stage'],'原版' if r['policy']=='baseline' else '候选',
        f'{r["attention_ms_per_layer"]:.4f}',f'{r["projection_ms_per_layer"]:.4f}',
        f'{r["routed_expert_ms_per_layer"]:.4f}',f'{r["shared_expert_ms_per_layer"]:.4f}',
        f'{r["longest_nccl_ms"]:.2f}',f'{r["attention_nccl_overlap_ms"]:.2f}'] for r in profile_rows]
    status='Prefill 在本轮已预热的受控测试中输出等价通过。' if accepted else 'Prefill 输出等价尚未通过；以下性能只作诊断，不放行该候选。'
    text=f'''# Q02：PP2 发送就绪优化与正确性诊断

日期：2026-10-25。{status}

## 1. 先说明本轮测的是什么

| 术语 | 本轮准确含义 |
|---|---|
| Prefill | 同时放行 8 条请求；每条**没有已缓存上下文**，处理 4096 个新输入 token，生成 1 token；整批新输入 32768 token。每批清空原生 GPU 缓存。 |
| Decode 回归 | 8 条请求，每条先独立重建 4096 个真实模型产生、驻留 GPU 的旧缓存，然后新增输入 1 token、生成 128 token。前缀构建在计时外；每批都重新构建，不让上下文持续增长。没有 CPU HiCache 换入。 |
| PP2、两槽 | 同一节点两张 GPU，每张负责连续 24 层；每阶段 Attention TP1、Expert EP1。两槽是调度循环批次位置，不是两个模型副本。Prefill 每次每卡实际处理 1 条请求的 4096 个新 token。 |
| TTFT、TPOT、端到端 | 客户端请求放行至首输出；首输出至末输出除以输出数量减一；放行至末输出。单位分别是秒、毫秒、秒。Prefill 仅输出 1 token，TPOT 不适用。 |
| 吞吐 | 每一整批输入或输出 token 数除以整批最后完成时间，再对正式批次取均值。Prefill 单位 input token/s，Decode 单位 output token/s。 |
| 均值 / 中位数 / P99 | 延迟表每格按此顺序；Prefill 每策略 2 批共 16 请求，Decode 每策略 2 批共 16 请求。P99 为最近秩样本最大值，不是稳定服务尾延迟估计。 |
| GPU 分项 | 独立 profile，按每卡每次层调用统计 GPU 活动并集。Attention 包含投影；Routed 是被路由的专家，Shared 是共享专家。NCCL 驻留包含等待，不能当作纯 NVLink 传输时间。 |

设备固定为 spartan-gpgpu069 GPU0/1、2×A100 80GB、NV12 NVLink。Guard 登记整节点 4 GPU，实际只用两卡，不申请新节点。
完整 Qwen3.5-122B-A10B-FP8：48 层、256 路由专家、每 token top-8；36 层线性 Attention、12 层 Full Attention。
A100 运行 W8A16 Marlin、BF16 激活和 KV、Triton Attention；CUDA Graph 与 overlap schedule 均关闭。原版和候选复用同一个已经加载的完整 CPU 权重 bank 和同一个模型进程。

## 2. 改动及受控方式

原生发送函数先把张量元数据通知另一阶段；数据若还没算好，接收阶段的 NCCL kernel 可能提前启动并等待，占用计算资源。
候选在发送端当前流已经接入生产事件依赖的位置，等待 CUDA 事件完成，然后再调用**同一个原生发送函数**发布元数据。
只改变通知时机，不改模型、张量、精度、专家路由或计算内核。等待仍在请求计时内；没有把开销移到计时外。

两个策略各预热两批；正式 Prefill 顺序为**原版→候选→候选→原版**。之后分别独立采一批 profile，不计入性能。Decode 使用原版→候选→原版→候选，每批重新准备真实 GPU 前缀。记录完整生成 token IDs 和文本哈希。
代码是实验私有 overlay，**尚未默认写入共享引擎**；也没有声称能推广到其它槽数、PP 深度、模型或链路。

## 3. 请求性能

{table(['负载','策略','吞吐 ↑','TTFT 秒 ↓','TPOT 毫秒 ↓','端到端秒 ↓','正确性状态'],performance)}

同进程正式 Prefill，候选相对原版吞吐变化 **{gain:+.2f}%**，平均 TTFT 降低 **{ttft:.2f}%**。是否可使用该收益，须同时看第 5 节的正确性状态。

## 4. 时间线归因：不是只看最终吞吐

每卡每层单位为毫秒；最后两列为该 trace 的 kernel 最长驻留和 Attention 与 NCCL 的时间交集。
GPU 分项存在重叠，不能把这张表相加当成请求延迟。两阶段各 8 请求×24 层，共 192 次相同本地形状的层调用。

{table(['stage','策略','Attention 毫秒/层','其中 Marlin 投影毫秒/层','Routed Expert 毫秒/层','Shared Expert 毫秒/层','最长 NCCL 驻留毫秒','Attention/NCCL 时间交集毫秒'],profiles)}

本轮第一阶段的 Attention 从 9.2318→7.9523 毫秒／层，其中投影从 6.3800→4.9118（降低 23.01%）；Routed Expert 从 8.3115→6.9984（降低 15.80%）。第二阶段投影 4.8379→4.8810，基本不变。改动主要缓解原先第一阶段的争用，而不是把所有层的数学计算做得更少。

第一阶段 NCCL kernel 活动并集从 2399.68→1.85 毫秒、与 Attention 的交集从 1035.91→0.06 毫秒；原来最长约 301.51 毫秒的 NCCL 驻留降到 0.10 毫秒。**这些长时间主要是等待，不是物理搬运变快了几千倍。** 等待改在发送端数据就绪事件之前由 CPU 承担，避免对端过早启动等待 kernel；等待仍计入请求时间。这组对照连同 profiler 关闭时的 15.08% 吞吐改善，支持“提前发布导致的 NCCL 等待干扰计算”是可优化慢点。

此次仍有 PP 顺序依赖、边界传输、每层更多专家权重等成本；不能把全部 EP／PP 差距都归因于这个问题。此轮没有重新测 EP2，也没有测试四槽候选，不能直接宣称候选在新一轮配对中胜过 EP2 或适用于四槽。

## 5. 正确性、此前失败与验收范围

{table(['检查','原版重复稳定','候选重复稳定','跨策略完整 token 一致','判定'],[
 [name,str(c.get('original_AA_stable')),str(c.get('candidate_BB_stable')),str(c['passed']),c['verdict']]
 for name,c in checks.items()])}

- r1：正式 Prefill 吞吐 8149.01→9360.71 input token/s，均值 TTFT 2.4012→2.1422 秒；64 条正式 Prefill 的文本哈希一致，但后续 Decode 哈希不一致、未采到 trace，整轮 Guard 为失败。不能拿这个中间结果当全部验收通过。
- r2：原版／候选第一次预热 Prefill 输出不一致，立即停止且 GPU 清理通过；未产出正式性能或 trace，不补写成功记录。
- r3：先保存预热、正式输出与 trace，再分别判断原版重复稳定性、候选重复稳定性和跨策略一致性。`inconclusive_original_unstable` 表示原版自身重复已变化，不能据此直接指控候选，也不能据此宣告候选正确。
- r3 的 Prefill 正式 32 条请求 token IDs 完全一致，预热 32 条也一致，独立 profile 输出亦与正式参考一致。Decode 原版 A/A 与候选 B/B 均不稳定，精确原因未定位；因此 Decode 指标只留作诊断，不当成有效提速或退化结论。
- 5 项候选 CPU 单元测试通过。r3 的 Guard、CPU 权重复用审计、实际 GPU 清理和注册释放通过。
- **验收范围仅为本轮已经预热的两槽 PP2 Prefill 性能实验，不是完整生成质量或通用调度正确性资格。** r2 预热输出差异仍保留为待查；不默认启用候选到 Decode 或其它工作负载。
- 原始失败证据不修改；数据完整性、Guard 操作状态、输出等价是三个独立判断。

## 6. 可追溯入口

- {link('本轮重新计算的指标',RUN/'analysis/metrics.csv')}；{link('逐卡 profile 表',RUN/'analysis/profile.csv')}；{link('校验 JSON',RUN/'analysis/validated.json')}。
- {link('冻结配置',RUN/'config.json')}；{link('候选实现',RUN/'code/pp_ready_send.py')}；{link('冻结 benchmark',RUN/'code/benchmark.py')}。
- {link('正式 Prefill 正确性',RUN/'pp2/correctness.prefill.json')}；{link('Decode 正确性',RUN/'pp2/correctness.decode.json')}；{link('逐文件来源与哈希',OUT/'provenance.json')}；{link('实际 GPU 清理证据',RUN/'task/cleanup.json')}。
- {link('此前未修改的 Q01',INDEX/'component_reports/Q01.md')}；{link('优化前 Prefill 诊断',INDEX/'qwen_prefill_diagnosis/REPORT.md')}。
'''
    (OUT/'REPORT.md').write_text(text)
    target=INDEX/'component_reports/Q02.md'
    target.write_text(text)
    index=INDEX/'README.md'
    value=index.read_text()
    row=f"| [Q02：PP2 发送就绪优化]({target}) | 与 Q01 同节点同一对 A100；两槽 PP，8 请求，无旧缓存、每条新输入 4096、输出 1；另有真实 GPU 缓存 Decode 回归。 | 同进程 A/B/B/A Prefill：{rows['prefill','baseline']['tokens_per_s']:.2f}→{rows['prefill','ready']['tokens_per_s']:.2f} input token/s（{gain:+.2f}%）。{'本轮正式 Prefill token 一致。' if accepted else '输出等价未通过，性能仅诊断。'} Decode 独立判定见报告。 | 同时保存每卡 Attention／投影、Routed／Shared Expert、NCCL 重叠的独立 GPU trace；原失败证据保留，未默认修改共享引擎。 |"
    import re
    value=re.sub(r'\n\| \[Q02：[^\n]*','',value)
    marker='<!-- END Q01 QWEN EP2 PP2 -->'
    assert value.count(marker)==1
    prefix,suffix=value.split(marker)
    value=prefix.rstrip()+'\n'+row+'\n\n'+marker+suffix
    index.write_text(value)
    print(json.dumps(dict(report=str(target),prefill_accepted=accepted,decode_accepted=data['decode_accepted'],gain_percent=gain)))


if __name__=='__main__':main()
