from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
ARMS=['ep4','two_ep2','four_ep1','two_pp2','pp4']
LABELS={'ep4':'EP4','two_ep2':'2×EP2','four_ep1':'4×EP1','two_pp2':'2×PP2','pp4':'PP4'}
def main():
    s=json.loads((ROOT/'summary.json').read_text());assert s['status']=='complete'
    assert {m['node'] for m in s['metrics']}=={'spartan-gpgpu178'}
    def metric(arm,P,phase,cache='natural_replay'):
        return next(m for m in s['metrics'] if m['layout']==arm and m['P']==P and m['phase']==phase and m['cache']==cache)
    rows=['# 35B 五种四卡部署：简单报告','',
    '五种布局均在同一台 4×H100 80GB NVLink 节点测量；Qwen3.6-35B-A3B BF16，全部开启 HiCache。每批64请求，正式点各3轮。',
    '', 'P 为每请求缓存的前缀长度。Prefill 每请求新增512 tokens、输出1 token；Decode 每请求新增1 token、输出128 tokens。下表单位均为 tokens/s，Prefill 仅计算新增输入，吞吐越高越好。','',
    '| 部署 | P=8,256 Prefill | P=8,256 Decode | P=130,304 Prefill | P=130,304 Decode | 长前缀 Decode 的 CPU 命中比例 |',
    '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        values=[metric(arm,P,phase)['throughput'] for P in [8256,130304] for phase in ['prefill','decode']]
        cpu=metric(arm,130304,'decode')['cpu_prefix_fraction']
        rows.append('| '+LABELS[arm]+' | '+' | '.join(f'{x:,.0f}' for x in values)+f' | {100*cpu:.1f}% |')
    ratio=metric('ep4',130304,'decode')['throughput']/metric('four_ep1',130304,'decode')['throughput']
    rows += ['',f'长前缀合计缓存8,339,456 tokens。开启 HiCache 后，EP4 对 4×EP1 的本轮长前缀 Decode 优势为 {ratio:.2f}×；4×EP1 主要从 CPU 命中，原先未开 HiCache 的24.1×不适用于本表。', '']
    capacities={c['layout']:c['conservative_gpu_token_pool'] for c in s['capacities']}
    rows.append('GPU KV token 池：'+'；'.join(f'{LABELS[a]} {capacities[a]/1e6:.3f}M' for a in ARMS)+'。长前缀占 EP4 GPU 池'+f'{100*8339456/capacities["ep4"]:.1f}%'+f'、PP4 GPU 池{100*8339456/capacities["pp4"]:.1f}%。')
    rows += ['', 'CPU 命中比例是实际原生计数的 CPU 前缀 tokens / 全部前缀 tokens，属于请求层面的命中，不是迁移字节比例。各部署每卡 CPU 池固定64 decimal GB KV + 8 decimal GB GDN。']
    rows += ['', '| 受控 CPU 重放，P=130,304 | Prefill tokens/s | Decode tokens/s | CPU 前缀 token 命中比例（Decode） |', '|---|---:|---:|---:|']
    for arm in ARMS:
        a=metric(arm,130304,'prefill','cpu_replay');b=metric(arm,130304,'decode','cpu_replay')
        rows.append(f"| {LABELS[arm]} | {a['throughput']:,.0f} | {b['throughput']:,.0f} | {100*b['cpu_prefix_fraction']:.1f}% |")
    rows += ['', 'CPU→GPU 恢复包含在请求耗时中；批前写入 fence 和主动移出 GPU 缓存位于计时外。这两种重放有不同缓存历史和生成分支，不能用两表差值单独归因 CPU 传输成本。这是合成的重复前缀负载，尚未测生产 trace 的命中率，也未搜索 HiCache 总容量上限。', '',
    'PP 按阶段、前缀长度、缓存模式选择本轮最快的有效候选。PP2/PP4 最终使用原生或 early-send 与 microbatch 调优；PP2 lookahead 实验路径未通过输出资格，tail-local 反馈在长上下文出现 slot 复用错误，均排除在正式比较外。这是最新 PP benchmark 仍需补齐的正确性资格。', '',
    '所有正式数据已从逐请求样本独立重算；冻结输入、缓存来源、Guard 后检、自有模型清理和租约释放已核验。', '',
    f'[完整结果与配置]({ROOT}/REPORT.md) · [逐轮数据]({ROOT}/full_metrics.csv) · [核验记录]({ROOT}/validation.json)']
    (ROOT/'SUMMARY.md').write_text('\n'.join(rows)+'\n')
    print('Wrote',ROOT/'SUMMARY.md')
if __name__=='__main__':main()
