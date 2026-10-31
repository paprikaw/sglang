"""CPU-only report derived from completed same-native transport measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(run):
    source=run/'comparison.json';data=json.loads(source.read_text());assert data['passed']
    cleanup=json.loads((run/'task/cleanup.json').read_text());assert cleanup['passed']
    independent=json.loads((run/'independent_validation.json').read_text());assert independent['passed']
    points=data['points'];assert len(points)==280
    tokens=(8,16,32,64,128,256,512,1024,2048,4096)
    def point(model,node,mode,cohort,n):
        location='cross8' if cohort=='whole' else 'local'+node
        return next(p for p in points if p['phase']==f'{model}_{location}_{mode}'
                    and p['cohort']==cohort and p['global_batch']==n)
    views=[];phase_rows=[]
    for p in points:
        phase_rows.append({k:v for k,v in p.items() if k!='metrics'}|{
            f'{key}_{stat}':val[stat] for key,val in p['metrics'].items() for stat in ('median_ms','mean_ms')})
    for model in ('glm','hy3'):
        for node in ('065','167'):
            for boundary in ('whole_vs_half','equal_call'):
                for n in tokens:
                    row=dict(model=model,ep4_node=node,boundary=boundary,global_batch_tokens=n,
                             ep8_call_tokens=n,ep4_call_tokens=n//2 if boundary=='whole_vs_half' else n,
                             ep8_assignments_per_gpu=n,ep4_assignments_per_gpu=n if boundary=='whole_vs_half' else 2*n)
                    for mode in ('rdma','ce'):
                        ep8=point(model,node,mode,'whole',n)
                        ep4s=[point(model,node,mode,c,n) for c in
                              (('half0','half1') if boundary=='whole_vs_half' else ('equal_call',))]
                        row[f'ep8_{mode}_ms']=ep8['metrics']['pair_cuda_ms']['median_ms']
                        row[f'ep4_{mode}_ms']=statistics.mean(p['metrics']['pair_cuda_ms']['median_ms'] for p in ep4s)
                    for width in ('ep4','ep8'):
                        row[f'{width}_ce_latency_change_pct']=(row[f'{width}_ce_ms']/row[f'{width}_rdma_ms']-1)*100
                        row[f'{width}_best_transport']='ce' if row[f'{width}_ce_ms']<row[f'{width}_rdma_ms'] else 'rdma'
                    views.append(row)
    write_csv(run/'phase_metrics.csv',phase_rows);write_csv(run/'transport_comparison.csv',views)
    lines=['# A100 单层专家通信：CE 与 RDMA 的 token 扫描','',
           '状态：完成。只测 DeepEP 通信，不加载模型或专家权重，不运行 Attention／Expert GEMM／KV Cache。',
           '', '## 测的是什么','',
           '| 项目 | 本轮设置 |','|---|---|',
           '| 硬件 | 已预约的 065／167，两节点共 8 张 A100；两个节点均核验 ConnectX-6 Dx |',
           '| GLM 通信形状 | BF16 向量宽度 5120；160 专家；每 token 路由到 8 个专家 |',
           '| Hy3 通信形状 | BF16 向量宽度 4096；192 专家；每 token 路由到 8 个专家 |',
           '| token 数 | 8、16、32、64、128、256、512、1024、2048、4096；都是本轮新送入专家的唯一 token，不是已缓存 context |',
           '| 数据／路由 | 确定性合成向量、均衡 top-8；活跃专家数自然增长；不另扫专家数量 |',
           '| RDMA 路径 | NVLink pair 内仍用 NVLink；同节点 pair 外及跨节点用 RDMA |',
           '| CE 路径 | pair 内 NVLink；同节点 pair 外 CE 经 PCIe；跨节点仍为 RDMA |',
           '| 执行 | eager；关闭 CUDA Graph；每点预热 3 次、正式 3 次 |',
           '| 计时 | Dispatch＋接收 hook＋Combine＋接收 hook；每次取参与 rank 最大值，再取中位数；含原生打包／还原、同步和 CPU 提交间隙 |',
           '| 排除 | 初始化、正确性验证、数据构造；没有模型计算、HiCache 或完整 PP 调度 |','',
           'GLM 的两条路径均使用 c3405cad…。Hy3 的两条路径使用同一个新尺寸适配二进制：只改 CE 内五处 5120→4096 常量，保留 host、其他 CUDA 对象、SM 配置和通知协议。完整 SHA 与构建命令见 `manifest.json`、`candidate/build_result.json`。',
           '', 'CE 的 dedup、匹配 peer 顺序、本地通知均开启；只有全本地 EP4 跳过本地 RDMA drain，EP8 保留远端完成顺序。容量固定为每 rank 1024 个源 token，足以覆盖最大一档；不是按每档重新调 buffer。',
           '', '## 怎么读两种比较方式','',
           '1. **相同整批、EP4 每次半批**：例如整批 128 个唯一 token，EP8 一次处理 128，EP4 两个槽各处理 64；两种布局每卡各执行 128 个 token–专家配对。EP4 表值是两个半批各自计时中位数的算术平均，属于每次调用成本，不是完整 PP 的吞吐。',
           '2. **单次调用 token 相同**：例如两者单次都处理 128 个唯一 token；EP8 每卡执行 128 个配对，EP4 每卡执行 256 个配对。这个对照保留 EP4 的更多局部工作，不能称为每卡等工作量。',
           '', '下表越低越好；单位毫秒。065、167 的 EP4 分别列出，不将不同节点的测量拼成一个八卡同步样本。','',
           '## 本轮观察','',
           '- **节点内 EP4：小消息 RDMA 更快，大消息 CE 更快。** 两种模型、两个节点的趋势一致。在“EP8 整批、EP4 半批”的比较中，整批 32 token 时 RDMA 仍较快，64 token 时 CE 已较快；对应 EP4 单次处理 16／32 个唯一 token。这里只定位到相邻实测档位，不声称精确切换阈值。',
           '- **节点内大消息收益明显。** 整批 4096 token、EP4 每次处理 2048 token 时，065 上 GLM 的 RDMA／CE 为 17.7161／2.4376 ms，Hy3 为 14.3401／2.0006 ms，CE 分别降低 86.2%／86.0% 的通信耗时。167 的完整表保留在下文。',
           '- **跨节点 EP8 不能照搬 EP4 的选择。** GLM 的 CE 没有显示实质优势：整批 256 token 基本持平，其余实测档位 CE 更慢；整批 4096 token 为 RDMA 26.6117、CE 31.0398 ms。Hy3 在 256–2048 token 的实测点上 CE 低 4.4%–18.8%，但 4096 token 时又变成 RDMA 21.3309、CE 26.4391 ms。',
           '- **不能只按 Prefill／Decode 名称选后端。** 这次隔离实验没有 Attention，也没有请求语义或 Prefill／Decode 内核切换。它支持“极小消息倾向 RDMA、节点内大消息受益于 CE”的有限结论；跨节点需同时考虑通信尺寸、模型形状与完整完成路径。',
           '- **机制解释与已证事实分开。** CE 路径的计时包含 CPU 读计数、提交拷贝和完成同步；EP8 还保留远端 RDMA 与 drain，源码及计数能确认这些路径存在。但本轮没有进一步拆出它们各占多少，不能把 EP8 的大消息退化直接归因于某一个环节。',
           '', '## 曲线','',
           '![两种模型通信形状、两种批次比较方式下的 CE 与 RDMA 曲线](transport_curves.png)','',
           '图中 EP4 使用 065；167 的独立对照见完整表。左列是相同整批、EP4 每次半批；右列是单次 token 相同。两轴均为对数刻度。','']
    for node in ('065','167'):
        lines += [f'## EP4 在节点 {node}；EP8 跨 065／167','']
        for model,label in [('glm','GLM'),('hy3','Hy3')]:
            for boundary,title in [('whole_vs_half','相同整批，EP4 每次处理半批'),('equal_call','两者单次处理相同 token 数')]:
                lines += [f'### {label}：{title}','',
                    '| 整批／单次 token 数 | EP8 RDMA | EP8 CE | EP4 RDMA | EP4 CE |',
                    '|---:|---:|---:|---:|---:|']
                for row in views:
                    if (row['model'],row['ep4_node'],row['boundary'])!=(model,node,boundary):continue
                    lines.append(f"| {row['global_batch_tokens']} | {row['ep8_rdma_ms']:.4f} | {row['ep8_ce_ms']:.4f} | {row['ep4_rdma_ms']:.4f} | {row['ep4_ce_ms']:.4f} |")
                lines.append('')
    lines += ['## 可直接复用的边界与限制','',
              '- CE／RDMA 对照逐点核验同二进制、同源 token 数、同路由哈希、同接收配对数、同容量和同 Graph 模式。',
              '- 验证专家计数、源 token 身份、全部 BF16 通道、Combine 返回值；最小／最大整批点补不等权重、不同专家输出、四次换数据与错峰到达检查；CE 正式样本有实际调用／拷贝／字节计数。',
              '- 单独的 Dispatch／Combine 数字在 `phase_metrics.csv`；二者各自的关键 rank 可不同，不应把两列中位数相加替代已测的总时间。',
              '- 三次采样不支持服务尾延迟结论；本轮没有请求级 TTFT／TPOT。大小消息的最佳路径以本轮实测点为准，不将 Prefill／Decode 名称当作固定的切换阈值。',
              '- 不含真实专家 GEMM，不把通信差值直接解释为 PP2×EP4 的完整模型加速；后续可按相同输入／路由协议接入单层计算和 PP 边界。','',
              '## 数据与复现入口','',
              '- `transport_comparison.csv`：80 行两种比较方式，含 CE 相对 RDMA 的变化率。',
              '- `phase_metrics.csv`：280 个真实通信点，Dispatch、Combine、完整通信和 CPU wall 的中位数／均值。',
              '- `full_metrics.csv`：840 个对齐的正式样本；各 phase 的 `result_rank*.json` 保留逐 rank 结果与正确性证据。',
              '- `independent_validation.json`：独立复算 64 份逐卡文件、1280 个逐卡测量点与 3840 个逐卡正式样本；校验 65 份冻结输入、16 份退出清理证据及租约释放。',
              '- `transport_curves.png`／`.svg`／`.pdf`：可直接查看或导出的科学曲线。',
              '- `config.json`、`frozen_inputs.json`、`frozen/harness`：本轮配置和实际运行的冻结代码。',
              '- `task/cleanup.json`：所有参与节点的退出和 GPU 清理证据；Guard 审计不与科学计时混合。','']
    (run/'summary.md').write_text('\n'.join(lines))
    native=json.loads((run/'candidate/build_result.json').read_text())
    manifest=dict(id='single_layer_ep4_ep8_ce_rdma_20261024_r1',guard_experiment_id=run.name,
          status='complete',kind='single_layer_shape_communication_only',
          source_run=str(run),models=['GLM-4.7-AWQ communication geometry','Hy3 BF16 communication geometry'],
          actual_model_weights_loaded=False,attention_compute=False,expert_compute=False,
          config=str(run/'config.json'),config_sha256=sha(run/'config.json'),
          comparison_sha256=sha(source),native_glm=next(p['native_sha256'] for p in points if p['model']=='glm'),
          native_hy3=native['native_sha256'],hy3_patch_sha256=native['cuda_source_sha256'],
          points=len(points),aligned_samples=len(points)*3,cleanup_passed=True,
          independent_validation_passed=independent['passed'],
          artifacts=['summary.md','transport_comparison.csv','phase_metrics.csv','full_metrics.csv','comparison.json',
                     'independent_validation.json','config.json','frozen_inputs.json',
                     'transport_curves.png','transport_curves.svg','transport_curves.pdf'],
          source_boundary='CE/RDMA same native within each model; model shapes use different natives; no GEMM')
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(report=str(run/'summary.md'),points=len(points),comparisons=len(views))))
    return views


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    report(p.parse_args().run.resolve())
