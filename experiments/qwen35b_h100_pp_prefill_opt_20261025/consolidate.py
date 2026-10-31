"""Recompute all saved samples and build a bounded, traceable result report."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
PERF=('h100_pp_prefill_coalesce_r1','h100_pp_prefill_chunk_r3','h100_pp_prefill_ready_r4')
PROFILE='h100_pp_prefill_profile_r5'


def digest(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return json.loads(p.read_text())
def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def build():
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    combined=[];request_rows=[];sources=[];validations=[];stage_rows=[]
    for name in PERF:
        run=ROOT/'runs'/name;cfg=load(run/'config.json');done=load(run/'completed.json')
        assert done['passed'] and done['points']==cfg['expected_points']
        check=next(run.glob('dashboard_guard/checks/*-post.json'));post=load(check)
        assert post['passed'] and post['scientific_success'] and post['cleanup_verified']
        assert load(run/'worker_cleanup.json')['cleanup_verified']
        raw=list(sorted(run.glob('*/metrics.*.json')));assert len(raw)==cfg['expected_points']
        for path in raw:
            d=load(path);case=d['case'];result=d['result'];reqs=result['requests']
            assert len(reqs)==case['requests']
            expected_rate=case['requests']*case['new_tokens']/result['makespan_s']
            assert math.isclose(expected_rate,d['new_token_throughput'],rel_tol=1e-12)
            for i,r in enumerate(reqs):
                assert r['cached_tokens']==case['cached_tokens']
                assert r['prompt_tokens']==case['cached_tokens']+case['new_tokens']
                assert r['output_tokens']==1 and r['output_ids'] is not None
                assert r['ttft_s']==r['e2e_s'] and r['tpot_ms'] is None
                request_rows.append(dict(run=name,case=case['id'],configuration=d['arm'],
                    policy=d.get('chunk_tokens',d.get('coalescing',d.get('ready_send'))),trial=d['trial'],request_index=i,
                    cached_tokens=r['cached_tokens'],new_prefill_tokens=case['new_tokens'],
                    output_tokens=1,ttft_ms=r['ttft_s']*1000,end_to_end_ms=r['e2e_s']*1000,
                    source=str(path)))
            sources.append(digest(path))
        summary=load(run/'analysis/summary.json')
        for row in summary['rows']:combined.append(dict(run=name,**row))
        agreement=run/'pp2'/('chunk_output_agreement.json' if name.endswith('r3') else 'output_agreement.json')
        assert load(agreement)['passed'];sources.extend((digest(agreement),digest(check),digest(run/'worker_cleanup.json')))
        validations.append(dict(run=name,formal_cohorts=len(raw),guard_postcheck_passed=True,
            cleanup_verified=True,model_cache_verified=post['model_cache']['passed'],postcheck=digest(check)))
    r3=ROOT/'runs'/PERF[1];cold=load(r3/'analysis/cold_output_agreement.json');assert cold['passed']
    # Re-derive the headline output agreement from the formal samples, not only a stored pass flag.
    cold_rows=[load(p) for p in sorted(r3.glob('pp2/metrics.cold8.*.json'))]
    outputs=[[r['output_ids'] for r in p['result']['requests']] for p in cold_rows]
    assert len(outputs)==8 and all(x==outputs[0] for x in outputs)
    r4=ROOT/'runs'/PERF[2];ready_rows=[load(p) for p in sorted(r4.glob('pp2/metrics.*.json'))]
    outputs=[[r['output_ids'] for r in p['result']['requests']] for p in ready_rows]
    assert len(outputs)==4 and all(x==outputs[0] for x in outputs)
    diagnostic=ROOT/'runs'/PROFILE
    assert load(diagnostic/'completed.json')['passed']
    assert load(diagnostic/'worker_cleanup.json')['cleanup_verified']
    check=next(diagnostic.glob('dashboard_guard/checks/*-post.json'));profile_post=load(check)
    assert profile_post['exit_code']==0 and profile_post['cleanup_verified'] and profile_post['model_cache']['passed']
    assert profile_post['valid']==0 and profile_post['expected']==2 and not profile_post['passed']
    for directory in diagnostic.glob('pp2/traces/*'):
        manifest=load(directory/'manifest.json');assert manifest['passed'] and len(manifest['files'])==2
        for exported in manifest['files']:assert digest(Path(exported['path']))['sha256']==exported['sha256']
    for run in (ROOT/'runs'/PERF[0],r3,diagnostic):
        with (run/'analysis/prefill_components.csv').open() as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==4
        stage_rows.extend(dict(run=run.name,**r) for r in rows)
        sources.extend((digest(run/'analysis/prefill_components.csv'),digest(run/'analysis/profile_inventory.json')))
    validations.append(dict(run=PROFILE,profile_only=True,guard_postcheck_passed=False,cleanup_verified=True,
        model_cache_verified=True,formal_metrics_expected=False,profile_completed_points=2,
        guard_reported_valid=0,guard_reported_expected=2,
        limitation='Controller final progress counter counts metrics rather than profile manifests; original failed postcheck is preserved.',
        two_complete_gpu_exports_per_policy=True,postcheck=digest(check)))
    write_csv(out/'full_metrics.csv',combined);write_csv(out/'request_metrics.csv',request_rows);write_csv(out/'stage_metrics.csv',stage_rows)
    evidence=dict(passed=True,created_at=datetime.now(timezone.utc).isoformat(),formal_cohorts=sum(x.get('formal_cohorts',0) for x in validations),
        grouped_metric_rows=len(combined),request_rows=len(request_rows),stage_rows=len(stage_rows),
        validation=validations,sources=sources,formal_cold_outputs_identical=True,formal_long_ready_outputs_identical=True,
        profile_warning='R4 final candidate stage1 export was not joined and is partial. R1 and R3 have both stage traces. R5 independently re-exports both ready-send policies completely; formal performance is never rerun or replaced.')
    (out/'validation.json').write_text(json.dumps(evidence,indent=2)+'\n')
    def row(run,case,policy,arm='pp2'):
        return next(r for r in combined if r['run']==run and r['case']==case and r['policy']==policy and r['configuration']==arm)
    def gain(a,b):return (b['throughput_new_tokens_s']/a['throughput_new_tokens_s']-1)*100
    ca=row(PERF[1],'cold8',4096);cb=row(PERF[1],'cold8',8192)
    la=row(PERF[2],'long_cached32',False);lb=row(PERF[2],'long_cached32',True)
    report=['# H100 PP2 Prefill 优化：结果与分阶段证据','',
      '2026-10-25；Qwen3.6-35B-A3B BF16；H100 节点 178。这里只报告新输入处理，输出固定为 1 token；不是 Decode／TPOT 优化结果。','',
      '## 1. 两项有输出一致性证据的改善','',
      '| 明确的请求负载 | 原版 → 候选 | 新输入 token/s ↑ | 整批完成，毫秒 ↓ | 平均 TTFT，毫秒 ↓ |',
      '|---|---|---:|---:|---:|',
      f'| 8 请求；每条没有旧缓存，新输入 4096 token | 每次 forward 最多 4096 → 8192 新 token | {ca["throughput_new_tokens_s"]:.1f} → **{cb["throughput_new_tokens_s"]:.1f}（+{gain(ca,cb):.2f}%）** | {ca["mean_cohort_duration_ms"]:.2f} → {cb["mean_cohort_duration_ms"]:.2f} | {ca["ttft_ms_mean"]:.2f} → {cb["ttft_ms_mean"]:.2f} |',
      f'| 32 请求；每条 GPU 已缓存 89,984 token，再输入 512 token | 保持每次最多 4096 新 token；数据就绪后再发送 | {la["throughput_new_tokens_s"]:.1f} → **{lb["throughput_new_tokens_s"]:.1f}（+{gain(la,lb):.2f}%）** | {la["mean_cohort_duration_ms"]:.2f} → {lb["mean_cohort_duration_ms"]:.2f} | {la["ttft_ms_mean"]:.2f} → {lb["ttft_ms_mean"]:.2f} |','',
      '第一项是改变原生 Prefill 每次执行的新输入预算，两个 PP stage 同步修改 `chunked_prefill_size` 和 `max_prefill_tokens`；没有减少请求或输入。第二项复用 A100 的发送就绪适配：等待生产端 CUDA 事件完成，再交给原生发送函数发布元数据，避免接收端过早占用 NCCL 执行资源等数据。等待仍包含在测量内。','',
      '**适用范围不同：** 长缓存的 8192-token chunk 比原 4096 更慢，不能统一放大；短缓存并发 32 的 8192 也没有改善。没有把以上两项收益相乘，或声称已经得到同配置下的 EP2 最优对照。','',
      '## 2. 测量协议和名词','',
      '- **已缓存历史 token：** 请求之前的真实模型 KV 与线性 Attention/GDN 状态，正式计时前已经在 GPU。不是再次做相同长度的 Prefill，也不是从 CPU HiCache 换入。',
      '- **本轮新输入：** 正式计时中真正处理的 Prefill token；吞吐只计这些新 token，不把旧前缀重复计入。',
      '- **每次 forward 的新 token 预算：** 调度器一次送入模型的所有请求新输入之和，不是请求数；受实际到达、请求粒度及缓存状态影响，并不保证每次恰好满。',
      '- **TTFT：** 单个请求从发送至收到首个生成 token 的客户端耗时，包含调度、必要同步和通信；整批完成时间从 cohort 提交计时到全部完成。',
      '- **端到端延迟：** 本轮都只生成 1 token，因此等于 TTFT，完整 CSV 同时保留 mean／median／P99；没有可报告的 TPOT。',
      '- **PP2：** GPU2／3，各 20 层，Attention TP1，Expert EP1，原生两个循环槽位；Prefill 仍按预算分块，不是固定将 32 请求分成两个 16 请求一起计算。',
      '- **EP2 参考：** 仅第一轮另用 GPU0／1，Attention TP1×DP2、Expert EP2、各卡完整 40 层；每点两次，波动明显，不作为稳定最优排名。','',
      '三个负载为：32 请求各缓存 4096、新输入512；32 请求各缓存89984、新输入512；8请求无缓存、新输入4096。均保留所有 40 层，真实 BF16 DeepGEMM、FA3／Triton；Decode Graph 开、Prefill Graph 关。复用完整 CPU 权重 Bank，但没有把权重 Bank 称为 KV Bank。每次缓存试验独立重建同初始 GPU 前缀，预热与前缀准备不计时，正式样本与 profiler 不重叠。','',
      '合批和发送就绪使用同进程原版／候选／候选／原版，每种两次。chunk 扫描使用 4096／2048／1024／8192／8192／1024／2048／4096。正式冷 Prefill 8 批 64 个输出逐请求一致；长缓存发送就绪 4 批128个输出逐请求一致；另保存短缓存所有 chunk 的同输入输出检查。只能支持已测的单 token 输出一致性，不代替长生成质量验证。','',
      '## 3. 全部正式指标','',
      '数值为每策略两批的均值；TTFT median／P99 来自该策略两批全部请求。小样本 P99 只是这批请求的高位值，不是稳定服务尾延迟。不同轮次各自有基线，不拼接成连续加速比例。','']
    labels={PERF[0]:'第一轮：有上限的输入合批',PERF[1]:'第二轮：四种原生 chunk 预算',PERF[2]:'第三轮：长缓存发送就绪'}
    for name in PERF:
        report+=['### '+labels[name],'',
          '| 请求数 | 每条已缓存 token | 每条新输入 token | 配置／策略 | 新 token/s ↑ | 整批 ms ↓ | TTFT mean / median / P99，ms ↓ |',
          '|---:|---:|---:|---|---:|---:|---|']
        for r in combined:
            if r['run']!=name:continue
            mode=(str(r['policy'])+' 新 token／forward' if name==PERF[1] else
                 ('数据就绪后发送' if name==PERF[2] else '最多5ms合批') if r['policy'] else '原版')
            report.append(f'| {r["requests_per_sample"]} | {r["cached_tokens_per_request"]:,} | {r["new_prefill_tokens_per_request"]:,} | {r["configuration"].upper()}，{mode} | {r["throughput_new_tokens_s"]:.1f} | {r["mean_cohort_duration_ms"]:.2f} | {r["ttft_ms_mean"]:.2f} / {r["ttft_ms_median"]:.2f} / {r["ttft_ms_p99"]:.2f} |')
        report.append('')
    report+=['## 4. 分阶段归因','',
      '以下全部是独立采集的长缓存 32 请求 Prefill，每条 GPU 已缓存89,984、新输入512、输出1。单位为**毫秒／每卡／整批**，不是毫秒／层，也不是请求 TTFT。Attention 含模块内投影；路由和共享 Expert 分开。内核累计可重叠，NCCL 驻留还包括等待，不能相加作为端到端时间。','',
      '| 独立采样 | 策略 | stage | forward 次数 | Attention | 路由 Expert | 共享 Expert | NCCL 驻留 |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in stage_rows:
        sample='输入合批' if r['run']==PERF[0] else 'chunk 对照' if r['run']==PERF[1] else '发送就绪补采'
        report.append(f'| {sample} | {r["policy"]} | {r["stage"]} | {r["actual_model_forwards"]} | {float(r["attention_ms"]):.2f} | {float(r["routed_expert_ms"]):.2f} | {float(r["shared_expert_ms"]):.2f} | {float(r["nccl_residence_ms"]):.2f} |')
    report+=['',
      '缩小 chunk 的直接证据：4096 时共5次 forward，新增输入组成为512＋4096＋4096＋4096＋3584；2048时变成9次。Attention 总工作近似不变，路由 Expert 累计耗时上升，尽管 NCCL 驻留下降，实际整批仍更慢。该 profile 支持“更细分块有额外计算／执行开销”，没有测 HBM 读字节，不能据此断言精确的权重重复读取量。','',
      '发送就绪补采中，第1阶段NCCL驻留146.57→27.96ms，第2阶段20.88→1.45ms，与减少过早发布发送通知后的GPU等待相符。它不是NVLink载荷传输时间，也不能把减少的累计驻留直接当作端到端收益：正式整批只降低约22.55ms。补采原版首块为1024、新版首块为512，虽然总新token及forward次数相同，仍不能把全部差额严格归为单一NCCL内核因素。','',
      '有上限合批只把首块从512改为1024，总forward仍为5，三个负载都未获益，因此不采用。长缓存发送就绪的完整两阶段表来自单独补采 R5；R4最后一份末阶段 trace 没有等待写盘就清理，只保留为不完整诊断，不用它冒充完整因果表。正式计时来自原始 R1／R3／R4，未被 profiler 数字替换。','',
      '## 5. 图、源码、验证和限制','',
      f'![四种 chunk 的吞吐和延迟]({r3}/analysis/chunk_comparison.png)','',
      f'![实际分块执行时间线；包含发射间隔]({r3}/analysis/pipeline_timeline.png)','',
      f'[全部请求指标 CSV]({out}/full_metrics.csv) · [逐请求时序 CSV]({out}/request_metrics.csv) · [逐卡分阶段 CSV]({out}/stage_metrics.csv) · [原始路径、哈希与验证]({out}/validation.json)','',
      f'[发送就绪完整诊断]({diagnostic}/analysis/PROFILE.md) · [chunk 诊断]({r3}/analysis/PROFILE.md) · [适配与测试源码]({ROOT})','',
      '三轮46个正式cohort、23组指标均通过缓存命中、输出长度与原始 Guard postcheck；补采另有2组完整profile；其控制器错误地按性能文件计为0／2，原始Guard postcheck因此失败，未改写为通过。两阶段导出、缓存及清理证据单独校验通过，不能把该补采称为原始Guard全通过。所有 GPU 进程按拥有者范围清理、租约释放；保留 CPU 权重 Bank。11 项 CPU 测试通过。改动保持在冻结的实验私有适配，不修改共享引擎默认值，也不影响正在使用的服务。','',
      '本轮只优化两个已测 Prefill 区间，不代表所有负载最优。特别是长缓存8192策略两批波动较大；没有扫描所有层分配、内核或请求到达模式。没有运行 Decode，不能外推 TPOT；也没有充分重测 EP2 最优参数，因此未声称已消除此前 PP 相对 EP 的全部差距。','']
    (ROOT/'RESULTS.md').write_text('\n'.join(report))
    (out/'headlines.json').write_text(json.dumps(dict(cold_chunk_gain_percent=gain(ca,cb),long_ready_gain_percent=gain(la,lb),cold_baseline=ca,cold_candidate=cb,long_baseline=la,long_candidate=lb),indent=2)+'\n')
    print(json.dumps(dict(report=str(ROOT/'RESULTS.md'),metrics=len(combined),requests=len(request_rows),stages=len(stage_rows))))


if __name__=='__main__':build()
