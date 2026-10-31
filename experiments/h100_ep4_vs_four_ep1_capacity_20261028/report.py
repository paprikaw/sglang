"""Verify saved measurement boundaries, then archive a small report snapshot."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,math,re,statistics

def load(p):return json.loads(p.read_text())
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def avg(xs):return statistics.mean(xs)

def report(run):
    cfg=load(run/'config.json');posts=[load(p) for p in (run/'dashboard_guard/checks').glob('*-post.json')]
    assert len(posts)==1
    post=posts[0]
    assert post['valid']==post['expected']==11 and post['model_cache']['passed'] and post['cleanup_verified']
    assert post['exit_code']==1 and post['passed'] is False
    worker_log=(run/'controller.log').read_text()
    assert 'SystemExit: 0' in worker_log and 'RuntimeError:' not in worker_log
    assert load(run/'analysis/entrypoint_fix_validation.json')['fixed_success_exit_codes']==[0,0,0]
    for node in post['model_cache']['nodes']:
        for use in node['uses']:
            raw=Path(use['path']).read_bytes()
            assert len(raw)==use['delta_bytes'] and hashlib.sha256(raw).hexdigest()==use['delta_sha256']
            assert use['observed_ranks']==use['expected_ranks']
    assert load(run/'task/cleanup.json')['passed']
    assert '"gpu_registry_state": "released"' in (run/'supervisor.log').read_text()
    spec=load(run/'dashboard_spec.json')
    for d in spec['frozen_inputs']:assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
    plan=load(run/'ep4_maximum_plan.json');maximum=plan['result_case'];cases=['four_ep1_common','ep4_common',maximum,'four_ep1_at_ep4max']
    cap1=load(run/'evidence/four_ep1_common/capacity.json');cap4=load(run/'evidence/ep4/capacity.json');ratio=cap4['aggregate_conservative_tokens']/cap1['aggregate_conservative_tokens']
    rows=[];metrics={};checks=[]
    for case in cases:
        protocol=load(run/'results'/case/'workload.json')
        for path in sorted((run/'results'/case).glob('metrics.*.json')):
            phase=path.name.split('.')[1];j=load(path);metrics[(case,phase)]=j
            assert j['valid'] and j['profiler_enabled'] is False
            flat=[z for rr in j['rounds'] for z in rr['requests']]
            assert math.isclose(j['output_tokens_per_s'],avg([rr['output_tokens_per_s'] for rr in j['rounds']]),rel_tol=1e-10)
            assert math.isclose(j['ttft_s']['mean'],avg([z['ttft_s'] for z in flat]),rel_tol=1e-10)
            if phase=='decode':assert math.isclose(j['tpot_ms']['mean'],avg([z['tpot_ms'] for z in flat]),rel_tol=1e-10)
            expected_output=128 if phase=='decode' else 1
            for idx,round_ in enumerate(j['rounds']):
                rr=round_['requests'];assert len(rr)==64 and all(r['output_tokens']==expected_output for r in rr)
                duration=max(r['finished'] for r in rr)-min(r['started'] for r in rr)
                assert math.isclose(duration,round_['makespan_s'],rel_tol=1e-10)
                assert math.isclose(sum(r['output_tokens'] for r in rr)/duration,round_['output_tokens_per_s'],rel_tol=1e-10)
                cached=[r['cached_tokens'] or 0 for r in rr]
                if phase=='cold_prefill':assert max(cached)==0
                if phase!='cold_prefill' and case!='four_ep1_at_ep4max':assert min(cached)>=protocol['P']-64
                P=0 if phase=='cold_prefill' else protocol['P'];Q=protocol['P'] if phase=='cold_prefill' else 512 if phase=='prefill' else 1
                tpots=[r['tpot_ms'] for r in rr if r['tpot_ms'] is not None]
                rows.append(dict(case=case,phase=phase,round=idx,B=64,P_cached_target=P,Q_new=Q,output_tokens_per_request=expected_output,
                    makespan_s=duration,output_tokens_per_s=round_['output_tokens_per_s'],new_input_tokens_per_s=64*Q/duration,
                    ttft_mean_s=avg([r['ttft_s'] for r in rr]),tpot_mean_ms=avg(tpots) if tpots else None,
                    cache_min=min(cached),cache_max=max(cached),prefix_hit_requests=sum(v>=protocol['P']-64 for v in cached)))
            checks.append(dict(case=case,phase=phase,cohorts=len(j['rounds']),requests_per_cohort=64,passed=True))
    assert len(metrics)==11, len(metrics)
    common1=load(run/'results/four_ep1_common/workload.json');common4=load(run/'results/ep4_common/workload.json')
    assert common1['prompt_hashes']==common4['prompt_hashes'];assert len(set(common1['prompt_hashes']))==64
    for case in ['four_ep1_common','ep4_common',maximum]:
        primes=[load(p) for p in sorted((run/'results'/case).glob('prime.*.json'))]
        assert len(primes)>=2 and all(p['all_prefixes_hit'] for p in primes[-2:])
    graph=load(run/'analysis/decode_graph.json');assert len(graph['rows'])==12
    assert all(r['graph_replays']>0 and r['actual_batch_sizes']==[16] for r in graph['rows'])
    for p in run.glob('evidence/*/cleanup.json'):assert load(p)['cleanup_verified']
    qualifications=[load(p) for p in run.glob('evidence/*/ep2/ep_kernel_qualification/*.json')];assert len(qualifications)==4 and all(q['passed'] for q in qualifications)
    validation=dict(passed=True,guard_postcheck_passed=False,guard_model_cache_passed=True,guard_cleanup_verified=True,wrapper_failure="Main catches its own SystemExit(0), exits1 after all11 measurements; actual parsed boundary reproduced and corrected for reuse",frozen_hashes_verified=True,common_prompts_identical=True,distinct_prefixes=64,formal_checks=checks,
        native_ep_small_kernel_qualification_ranks=4,profiler_correlated_rank_traces=12,
        raw_cold_P_field='Raw workload P is prefix-build target; this report uses cached P=0 and new Q=target for cold Prefill.',
        earlier_runs='r1 and r2 remain failed wrapper runs. Final metrics below use r3 only.')
    write(run/'analysis/protocol_validation.json',json.dumps(validation,indent=2)+'\n')
    out=run.parent.parent
    with (out/'full_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hot1=metrics[('four_ep1_common','prefill')];hot4=metrics[('ep4_common','prefill')];dec1=metrics[('four_ep1_common','decode')];dec4=metrics[('ep4_common','decode')]
    long4=metrics[(maximum,'decode')];long1=metrics[('four_ep1_at_ep4max','decode')]
    summary=dict(capacity_ratio=ratio,ep4_allocator_tokens=cap4['aggregate_conservative_tokens'],four_ep1_allocator_tokens=cap1['aggregate_conservative_tokens'],
        common_P=common1['P'],ep4_hot_prefill_ratio=hot4['input_tokens_per_s']/hot1['input_tokens_per_s'],ep4_hot_decode_ratio=dec4['output_tokens_per_s']/dec1['output_tokens_per_s'],
        ep4_maximum=plan,ep4_max_decode_tps=long4['output_tokens_per_s'],four_ep1_same_max_decode_tps=long1['output_tokens_per_s'],max_capacity_service_ratio=long4['output_tokens_per_s']/long1['output_tokens_per_s'])
    def n(v):return '—' if v is None else f'{v:,.3f}'
    lines=['# 35B：EP4 与四个 EP1 的容量、Prefill、Decode 对照','',
        'Qwen3.6-35B-A3B；同一个 H100 四卡节点 spartan-gpgpu178，四卡两两 NVLink。使用完整 BF16 权重、激活和 KV，GDN 状态 FP32。结果来自 2026-10-28 最终 r3，全部正式样本关闭 profiler。r3 末尾正常退出被包装脚本误判为失败；Guard 原始总状态保留 failed，缓存/清理后检通过，正式数据另经独立核验。','',
        '每个配置均为四卡总资源。EP4 的 Expert 跨四卡，Attention 为 TP1×DP4；四个 EP1 为四个独立、各持完整模型的单卡副本。PP 均为 1。两边每卡最多 16 个活动请求、84 个 GDN 状态槽，页长64、静态显存比例0.94、单卡 Prefill chunk4096、Decode Graph 开。引擎实际并发上限16已从启动日志验证。','',
        'B 是全局请求数，本轮固定64；P 是每请求计时前已驻留的 GPU context tokens；Q 是每请求本次新增输入。TTFT 从发送请求到首 token，TPOT 为首末输出 token 间时间除以127；吞吐按整个64请求 cohort 的完成跨度计算。高吞吐、低延迟较好。','',
        '## 主要结果','',f'- EP4 的 KV token 池为 {cap4["aggregate_conservative_tokens"]:,}，四个 EP1 合计 {cap1["aggregate_conservative_tokens"]:,}，容量比 **{ratio:.2f}×**。这是本轮固定状态槽/Graph/显存设置下的原生分配池，不是最大 QPS 或最大并发扫描。',
        f'- 相同 B64、每请求缓存 {common1["P"]:,} tokens：EP4 新增512-token Prefill 吞吐为四个 EP1 的 {summary["ep4_hot_prefill_ratio"]:.3f}×，Decode 输出吞吐为 {summary["ep4_hot_decode_ratio"]:.3f}×。',
        f'- EP4 通过驻留检查的最大测试点为每请求 {plan["P"]:,} tokens、总计 {plan["context_tokens"]:,} tokens，约占 KV token 池 {100*plan["allocator_fraction"]:.2f}%。全部64请求在正式热缓存样本中通过命中检查。','',
        '## 同时放得下的对照','',
        f'缓存 context 每请求 P={common1["P"]:,}，全局 B64。冷 Prefill 为 P=0、Q={common1["P"]:,}、生成1 token；热 Prefill Q512→1；Decode Q1→128。冷 Prefill 各1个 cohort，热 Prefill/Decode 各3个正式 cohort，另有未计时预热。','',
        '| 阶段 | 四个 EP1 总吞吐 tokens/s | EP4 总吞吐 tokens/s | 四个 EP1 平均 TTFT s | EP4 平均 TTFT s | 四个 EP1 TPOT ms | EP4 TPOT ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for phase,label in [('cold_prefill','冷 Prefill：全 context 新输入'),('prefill','热 Prefill：新增512'),('decode','Decode：输出128')]:
        a=metrics[('four_ep1_common',phase)];b=metrics[('ep4_common',phase)];key='output_tokens_per_s' if phase=='decode' else 'input_tokens_per_s'
        lines.append(f'| {label} | {n(a[key])} | {n(b[key])} | {n(a["ttft_s"]["mean"])} | {n(b["ttft_s"]["mean"])} | {n(a["tpot_ms"]["mean"] if a["tpot_ms"] else None)} | {n(b["tpot_ms"]["mean"] if b["tpot_ms"] else None)} |')
    lines+=['','## EP4 接近容量上限，交给两种布局相同输入','',f'两边同样 B64、每请求 context {plan["P"]:,} tokens。EP4 的热 Prefill/Decode 都复用真实 GPU 前缀。四个 EP1 的单卡 KV token 池只有 {cap1["per_attention_instance_min_tokens"]:,}，每卡最多完整驻留1条这样的请求；其相同负载包含排队与前缀重算。因此后两行是容量压力下的服务表现，不能当作纯热缓存 Expert/通信比值。','',
        '| 配置 / 阶段 | 吞吐 tokens/s | TTFT s | TPOT ms | 正式 cohort 数 |', '|---|---:|---:|---:|---:|']
    for case,phase,label in [(maximum,'cold_prefill','EP4 冷 Prefill'),('four_ep1_at_ep4max','cold_prefill','四个 EP1 冷 Prefill'),(maximum,'prefill','EP4 热 Prefill Q512'),(maximum,'decode','EP4 热 Decode'),('four_ep1_at_ep4max','decode','四个 EP1 同负载 Decode，含重算/排队')]:
        j=metrics[(case,phase)];key='output_tokens_per_s' if phase=='decode' else 'input_tokens_per_s'
        lines.append(f'| {label} | {n(j[key])} | {n(j["ttft_s"]["mean"])} | {n(j["tpot_ms"]["mean"] if j["tpot_ms"] else None)} | {len(j["rounds"])} |')
    lines+=['','## Expert 收益和通信开销','',
        '下表为独立 profiler 中每卡每次真实 B16 Decode Graph replay 的平均 GPU kernel 时间，单位 ms。GEMM 只含 routed Expert 两次矩阵乘；DeepEP kernel 含通信等待。不同流可重叠，kernel 合计不能相加当作独占占比，也不能直接解释正式 TPOT。','',
        '| 负载 / 配置 | Graph GPU 跨度 | Routed Expert GEMM 合计 | DeepEP kernel 合计 | rank traces × replay 范围 |','|---|---:|---:|---:|---|']
    for case in ['four_ep1_common','ep4_common',maximum]:
        rr=[r for r in graph['rows'] if r['case']==case]
        lines.append(f'| {case} | {n(avg([r["graph_gpu_span_ms"] for r in rr]))} | {n(avg([r["routed_gemm_sum_ms"] for r in rr]))} | {n(avg([r["deepep_sum_ms"] for r in rr]))} | {len(rr)} × {min(r["graph_replays"] for r in rr)}–{max(r["graph_replays"] for r in rr)} |')
    gr1=[r for r in graph['rows'] if r['case']=='four_ep1_common'];gr4=[r for r in graph['rows'] if r['case']=='ep4_common'];grm=[r for r in graph['rows'] if r['case']==maximum]
    gemm1=avg([r['routed_gemm_sum_ms'] for r in gr1]);gemm4=avg([r['routed_gemm_sum_ms'] for r in gr4])
    lines+=['',f'同负载下，EP4 每步 Expert GEMM 合计由 {gemm1:.3f} ms 降至 {gemm4:.3f} ms，减少 {100*(1-gemm4/gemm1):.1f}%；另有约 {avg([r["deepep_sum_ms"] for r in gr4]):.3f} ms 的 DeepEP kernel。Graph GPU 跨度由 {avg([r["graph_gpu_span_ms"] for r in gr1]):.3f} ms 变为 {avg([r["graph_gpu_span_ms"] for r in gr4]):.3f} ms，表明本点 Expert GEMM 的节省没有转化为更短的完整 GPU Graph。正式 TPOT 还包含 Graph 间调度、服务等开销，不能用上述 kernel 时间相减替代。',
        f'EP4 在长 context 下，完整 Attention kernel 合计从约 {avg([r["attention_full_sum_ms"] for r in gr4]):.3f} ms 增至 {avg([r["attention_full_sum_ms"] for r in grm]):.3f} ms；长 context 的 Decode 成本明显由读取 KV 的 Attention 推高。',
        '','## 测量边界与证据','',
        '- 前缀由模型真实计算并保留 GPU KV/GDN；没有匹配的 Qwen hybrid CPU checkpoint，本轮没有伪造或恢复 CPU KV。正式热缓存测量前，必须连续两轮64请求全部命中；准备和 profiler 不进入正式指标。所有初始请求使用不同首 token，避免跨请求共享同一个前缀。正文重复构造，属于合成负载，未测生产 trace 或任务质量。',
        '- 两边共享已有的小 batch Triton routed Expert GEMM 优化，局部输入≤16时启用；大 Prefill 仍为 DeepGEMM。EP4 使用原生 DeepEP normal Prefill / low-latency Decode，BF16 dispatch，原生 shared Expert overlap 开；TBO 关。H100 全 NVLink，DEEPEP_LL_NVLINK_PAIR_ONLY=0；本次未设置 CE 实验参数；所用 native12 二进制中检出的 DEEPEP 环境开关为 pair/PCIe threshold。原生路径和哈希见配置、冻结 manifest。',
        '- 早期 r1 因状态槽不足被引擎将每卡并发从16降到10，r2 的接近容量点首轮预热只有63/64命中，均保留为失败 wrapper 轮次。最终表格只用完成11项测量、缓存与清理后检通过的 r3。容量候选与驻留过程留存，不能将97%的配置目标称为已证明的硬件绝对上限。',
        '- 冷 Prefill 和四个 EP1 的容量压力负载只有1个 cohort；热缓存各3个，不能把请求级小样本 p99 当作稳定服务尾延迟。正式端到端结果与独立 profile 的时间边界不同。',
        f'- 引擎 `{cfg["engine_commit"]}`，固定源码标签 `{spec["sources"][0]["tag"]}`；模型 revision `{cfg["model_revision"]}`。',
        f'- [原始运行目录]({run}) · [配置]({run}/config.json) · [冻结输入]({run}/dashboard_spec.json) · [逐轮 CSV]({out}/full_metrics.csv) · [协议核验]({run}/analysis/protocol_validation.json) · [GPU Graph 分析]({run}/analysis/decode_graph.json)。',
        f'- [实际拓扑]({run}/topology.txt) · [容量候选]({run}/ep4_capacity_candidates.json) · [进程与 GPU 清理]({run}/task/cleanup.json)。Guard 的模型缓存与清理检查通过；总状态因退出处理错误为 failed。三组自有模型进程均已退出，租约已释放；没有新建 Slurm 预订。','']
    lines+=['## 收尾状态说明','',
        'r3 的 worker 已完成全部11项测量并成功清理三组服务，但主函数将自己的 sys.exit(0) 放在 except BaseException 内，捕获了正常 SystemExit:0，最终返回1。Guard 因此保留 scientific_success=false、passed=false，不能把它标成成功轮次。后检独立确认模型缓存9份启动日志/12个rank证明与清理全部通过；本报告重新核验了冻结哈希、全部请求时间与输出数、前缀命中和实际Graph批量。',
        f'用于以后复跑的修正版在 [reusable_code]({out}/reusable_code)，原始冻结运行没有改动。[补丁]({out}/entrypoint_fix.patch) 与 [实际主函数退出路径回归验证]({run}/analysis/entrypoint_fix_validation.json) 保留。没有为修复退出标记重新占用GPU。','']
    report_path=out/'REPORT.md';write(report_path,'\n'.join(lines));write(out/'summary.json',json.dumps(summary,indent=2)+'\n')
    eid='qwen35b_h100_ep4_vs_four_ep1_capacity_20261028';snap=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/eid;snap.mkdir(parents=True,exist_ok=True)
    import shutil
    for src,dst in [(report_path,snap/'summary.md'),(out/'full_metrics.csv',snap/'full_metrics.csv'),(out/'summary.json',snap/'summary.json'),(run/'analysis/protocol_validation.json',snap/'protocol_validation.json'),(run/'analysis/decode_graph.csv',snap/'stage_metrics.csv')]:shutil.copy2(src,dst)
    manifest=dict(id=eid,title='Qwen35B H100 EP4 versus four EP1: measured KV capacity and phase performance',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='measurements_complete_wrapper_failed',
        implementation=dict(engine='sglang',worktree=cfg['engine_root'],commit=cfg['engine_commit'],source_tag=spec['sources'][0]['tag'],dirty=False,private_adapters='all exact sources SHA256 frozen in dashboard_spec.json'),
        deployment=dict(id=cfg['id'],node=cfg['node'],job_id=cfg['job_id'],launch_manifest=str(run/'launch_manifest.json'),ep4_and_four_ep1_instances=9,state='retired',gateway_route=None),
        model=dict(path=cfg['model_path'],revision=cfg['model_revision'],precision=cfg['precision']),workload=dict(B=64,common_P=common1['P'],max_plan=plan,hot_prefill_Q=512,decode_Q=1,decode_output=128,hot_repeats=3,cold_repeats=1),
        artifacts=dict(original_run=str(run),report=str(report_path),snapshot_files=[p.name for p in snap.iterdir()],profiles=str(run/'results'),frozen_inputs=str(run/'dashboard_spec.json')),key_results=summary,validation=validation)
    write(snap/'manifest.json',json.dumps(manifest,indent=2)+'\n')
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    block=f'<!-- begin {eid} -->\n### {eid}\n\nSnapshot: {snap}\n\n- Four full-NVLink H100; full BF16 Qwen3.6-35B-A3B; EP4 Attention TP1/DP4 versus four independent EP1; B64.\n- KV allocator {cap4["aggregate_conservative_tokens"]:,} versus {cap1["aggregate_conservative_tokens"]:,} tokens ({ratio:.2f}x). Common P{common1["P"]}: EP4 hot Prefill ratio {summary["ep4_hot_prefill_ratio"]:.3f}, hot Decode throughput ratio {summary["ep4_hot_decode_ratio"]:.3f}.\n- Stable tested EP4 P{plan["P"]}, {100*plan["allocator_fraction"]:.2f}% token-pool occupancy. Same maximum workload on four EP1 includes queue/prefix recomputation. Formal hot3/cold1 cohorts, separate correlated Graph profiles, protocol hashes, canonical cache and cleanup pass. Guard overall failed solely because main catches SystemExit(0); independently validated raw measurements retained.\n\nReport: [{report_path.name}]({report_path})\n\n<!-- end {eid} -->'
    text=registry.read_text();pat=re.compile(r'<!-- begin '+re.escape(eid)+r' -->.*?<!-- end '+re.escape(eid)+r' -->',re.S)
    write(registry,pat.sub(lambda m:block,text) if pat.search(text) else text.rstrip()+'\n\n'+block+'\n')
    index=out.parent/'experiment_summary/README.md';text=index.read_text();entry=f'<!-- begin {eid} -->\n\n## 35B 四卡容量补充：EP4 与四个 EP1（2026-10-28）\n\n[简短报告]({report_path}) · [逐轮数据]({out}/full_metrics.csv)。同一 H100 四卡、Attention TP1、BF16，B64；KV token 池 EP4 {ratio:.2f}×。相同 P{common1["P"]}：EP4 热 Prefill {summary["ep4_hot_prefill_ratio"]:.3f}×、Decode 吞吐 {summary["ep4_hot_decode_ratio"]:.3f}×。EP4 稳定测试点 P{plan["P"]}；四个 EP1 同负载另记排队/重算，不能混入热缓存比较。此 B64 长 context 容量实验作为额外对照，未并入132格标准 Benchmark 计数。\n\n<!-- end {eid} -->'
    write(index,pat.sub(lambda m:entry,text) if pat.search(text) else text.rstrip()+'\n\n'+entry+'\n')
    print(json.dumps(dict(report=str(report_path),snapshot=str(snap),key_results=summary,validation_passed=True),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run)
