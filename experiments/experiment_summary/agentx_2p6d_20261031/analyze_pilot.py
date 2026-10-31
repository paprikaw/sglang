"""Summarize only completed, independently marked pilot episodes; read-only GPU side."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
RUN=WORK/'experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot'
LEGACY=RUN
RUNS=[RUN,WORK/'experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_3ep2_e8_v4_gdn256_20261031_control',
      WORK/'experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control']
sys.path.insert(0,str(HERE.parent/'agentx_workload_profile_20261031'))
from analyze import percentile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(name,rows):
    if not rows:
        return
    with (HERE/name).open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def collect(RUN):
    cfg=json.loads((RUN/'config.json').read_text())
    manifest=json.loads((RUN/'launch_manifest.json').read_text())
    progress=json.loads((RUN/'task/progress.json').read_text())
    metrics,instances,evidence,forward_rows=[],[],[],[]
    for p in sorted(RUN.glob('results/*/*/*/formal.r*.json')):
        marker=p.with_name('completed.'+p.stem.split('.')[-1]+'.json')
        if not marker.exists():
            continue
        completed=json.loads(marker.read_text())
        assert completed['passed']
        raw=json.loads(p.read_text());rows=raw['requests']
        case=p.parent.parent.name;arm=p.parent.parent.parent.name
        expected=171 if case=='agentx_c8' else 2736
        assert len(rows)==expected==raw['completed_requests']
        total_input=sum(r['input_tokens'] for r in rows)
        total_output=sum(r['output_tokens'] for r in rows)
        cached=sum(r['prefill_meta']['cached_tokens'] for r in rows)
        device=host=0
        for r in rows:
            m=r['prefill_meta'];d=m.get('cached_tokens_details') or dict(device=0,host=0)
            assert d['device']+d['host']==m['cached_tokens']
            device+=d['device'];host+=d['host']
        mean=lambda k:statistics.fmean(r[k] for r in rows)
        result=dict(arm=arm,case=case,source_run=RUN.name,
            placement_cohort='instance_equal_diagnostic' if RUN==LEGACY else 'owner_balanced',
            repeats=1,requests=expected,wall_s=raw['wall_s'],
            ttft_ms=mean('ttft_ms'),tpot_ms=mean('tpot_ms'),e2e_ms=mean('e2e_ms'),
            ready_ttft_ms=statistics.fmean(r['ttft_ms']+r['client_dispatch_lag_ms'] for r in rows),
            ready_e2e_ms=statistics.fmean(r['e2e_ms']+r['client_dispatch_lag_ms'] for r in rows),
            ttft_p50_ms=percentile([r['ttft_ms'] for r in rows],.5),
            ttft_p90_ms=percentile([r['ttft_ms'] for r in rows],.9),
            ttft_p99_ms=percentile([r['ttft_ms'] for r in rows],.99),
            tpot_p50_ms=percentile([r['tpot_ms'] for r in rows],.5),
            tpot_p90_ms=percentile([r['tpot_ms'] for r in rows],.9),
            tpot_p99_ms=percentile([r['tpot_ms'] for r in rows],.99),
            mean_new_prefill_tokens=(total_input-cached)/len(rows),
            mean_decode_tokens=total_output/len(rows),
            p_prefix_hit_pct=100*cached/total_input,p_device_hit_pct=100*device/total_input,
            p_host_hit_pct=100*host/total_input,new_p_to_d_token_ratio=(total_input-cached)/total_output,
            mean_p_reported_queue_ms=1000*statistics.fmean(r['prefill_meta']['queue_time'] for r in rows),
            decode_batch_step_weighted_mean=None)
        forwards=p.with_name('forwards.'+p.stem.split('.')[-1]+'.json')
        if forwards.exists():
            delta=json.loads(forwards.read_text())['delta'];numerator=denominator=0
            for rank,d in delta.items():
                for mode,count in d['counts'].items():
                    match=re.fullmatch(r'(DECODE|EXTEND):b(\d+)',mode)
                    if match:
                        phase,size=match[1],int(match[2])
                        forward_rows.append(dict(arm=arm,case=case,rank=rank,phase=phase,batch_size=size,forward_count=count))
                        if phase=='DECODE':numerator+=size*count;denominator+=count
            if denominator:
                result['decode_batch_step_weighted_mean']=numerator/denominator
        metrics.append(result)
        evidence.append(dict(path=str(p),sha256=sha(p),marker=str(marker),marker_sha256=sha(marker)))
        for inst in sorted({r['decode_instance'] for r in rows}):
            group=[r for r in rows if r['decode_instance']==inst]
            inp=sum(r['input_tokens'] for r in group)
            instances.append(dict(arm=arm,case=case,decode_instance=inst,
                handoff='same_node' if all(r['prefill_instance'].split('.')[0]==inst.split('.')[0] for r in group) else 'cross_node',
                requests=len(group),output_tokens=sum(r['output_tokens'] for r in group),
                tpot_ms=statistics.fmean(r['tpot_ms'] for r in group),
                ttft_ms=statistics.fmean(r['ttft_ms'] for r in group),
                d_reported_prefix_hit_pct=100*sum(r['meta_info']['cached_tokens'] for r in group)/inp))
    allocations,markers=[],[]
    marker_names=['MAMBA_CACHE_CHECKPOINT_DEFERRED','HICACHE_LOAD_BACK_DEFERRED',
                  'HICACHE_LOAD_BACK_RETRY_SUCCEEDED','layers_current_pp_stage is out of range',
                  'out of mamba cache slots','AssertionError','CUDA out of memory',
                  'Prefill bootstrap failed','Decode transfer failed',
                  'timed out after 120.0s','timed out after 300.0s']
    for node,local in cfg['node_configs'].items():
        for layout in local['layouts']:
            for i,inst in enumerate(layout['instances']):
                p=Path(layout['usage_logs'][i]['path']);text=p.read_text()
                states=[int(s) for s in re.findall(r'Mamba Cache is allocated\. max_mamba_cache_size: (\d+)',text)]
                kv=[int(s) for s in re.findall(r'KV Cache is allocated[^\n]*#tokens: (\d+)',text)]
                for rank,(state,tokens) in enumerate(zip(states,kv)):
                    assert state==256
                    allocations.append(dict(node=node,arm=layout['id'],replica=i,role=inst['role'],
                        ep=inst['ep'],pp=inst['pp'],log_rank_order=rank,gdn_slots=state,kv_tokens=tokens,
                        log=str(p),evidence_scope='startup allocation; PP stages share token indices'))
                for name in marker_names:
                    markers.append(dict(node=node,arm=layout['id'],replica=i,role=inst['role'],
                        marker=name,count=text.count(name),scope='entire arm including qualification/warmup;0 in empty log is not evidence'))
    cleanup_path=RUN/'task/cleanup.json'
    cleanup=json.loads(cleanup_path.read_text()) if cleanup_path.exists() else None
    partials=[]
    for p in sorted(RUN.glob('results/*/*/*/partial.r*.json')):
        raw=json.loads(p.read_text())
        partials.append(dict(path=str(p),sha256=sha(p),arm=p.parent.parent.parent.name,
            case=p.parent.parent.name,placement_cohort='instance_equal_diagnostic' if RUN==LEGACY else 'owner_balanced',
            collected_requests=raw['collected_requests'],
            error_type=raw['error_type'],error=raw['error'],complete=False))
    return dict(run=str(RUN),manifest=manifest,progress=progress,metrics=metrics,instances=instances,
        evidence=evidence,forward_rows=forward_rows,allocations=allocations,markers=markers,
        partials=partials,cleanup=cleanup)


def main():
    sources=[collect(run) for run in RUNS if (run/'config.json').exists()]
    flat=lambda key:[row for source in sources for row in source[key]]
    metrics,instances,evidence=flat('metrics'),flat('instances'),flat('evidence')
    forward_rows,allocations,markers=flat('forward_rows'),flat('allocations'),flat('markers')
    partials=flat('partials')
    assert len({(m['source_run'],m['arm'],m['case']) for m in metrics})==len(metrics), 'Do not double-count repeat1 episodes'
    primary=[m for m in metrics if m['placement_cohort']=='owner_balanced']
    diagnostic=[m for m in metrics if m['placement_cohort']=='instance_equal_diagnostic']
    for name,rows in [('full_metrics.csv',metrics),('instance_metrics.csv',instances),
        ('slot_allocations.csv',allocations),('runtime_markers.csv',markers),('forward_batches.csv',forward_rows)]:
        write_csv(name,rows)
    manifest=sources[0]['manifest']
    assert len({s['manifest']['engine_commit'] for s in sources})==1
    active=[s for s in sources if s['progress']['stage'] not in ('queued','failed','complete')]
    progress=(active[-1] if active else sources[-1])['progress']
    cleanup=dict(passed=all(s['cleanup'] and s['cleanup']['passed'] for s in sources),
        scope='Aggregation of individual native task cleanup evidence; see source_runs',
        source_runs=[dict(run=s['run'],cleanup=s['cleanup']) for s in sources])
    analysis=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=str(RUN),
        engine_commit=manifest['engine_commit'],progress=progress,accepted_episodes=len(metrics),
        expected_episodes=6,expected_primary_episodes=4,accepted_primary_episodes=len(primary),
        accepted_diagnostic_episodes=len(diagnostic),evidence=evidence,cleanup=cleanup,partials=partials,
        source_runs=[dict(run=s['run'],progress=s['progress'],cleanup=s['cleanup']) for s in sources],
        analyzer_sha256=sha(Path(__file__)))
    (HERE/'pilot_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    labels={'pd_2p6d_pp2_ep4ep2':'EP4＋EP2 D','pd_2p6d_pp2_3ep2':'3×EP2 D'}
    lines=['# 2P6D 运行记录','',f'更新时间：{analysis["updated_at"]}；状态：**{progress["stage"]}**；owner均衡对照完整 episode：**{len(primary)}/4**；另保留旧分流诊断 {len(diagnostic)} 轮。','',
        f'当前引擎 E8 `{manifest["engine_commit"]}`，包含 E7 状态恢复修复和本轮 Hybrid MHA PP 层映射修复。', '',
        '## 已验证性能（每点一轮，不能估计稳定误差）','',
        '| D 配置 | C | 请求数 | episode s | 平均 TTFT ms | 平均 TPOT ms | 平均 E2E ms | P 前缀命中 | 新增 P token／请求 | 实际 D batch均值 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in primary:
        batch='—' if m['decode_batch_step_weighted_mean'] is None else f'{m["decode_batch_step_weighted_mean"]:.2f}'
        lines.append(f'| {labels[m["arm"]]} | {m["case"][8:]} | {m["requests"]} | {m["wall_s"]:.2f} | {m["ttft_ms"]:.2f} | {m["tpot_ms"]:.2f} | {m["e2e_ms"]:.2f} | {m["p_prefix_hit_pct"]:.2f}% | {m["mean_new_prefill_tokens"]:,.1f} | {batch} |')
    if not primary:
        lines += ['', '尚无通过 completed marker 的完整性能结果；加载、资格检查和部分响应均不作为正式 benchmark。']
    if partials:
        lines += ['', '未完成的 episode 不计入上表：']
        for row in partials:
            lines += [f'- {labels[row["arm"]]} · {row["case"]} · {row["placement_cohort"]}：仅收集 {row["collected_requests"]} 条，`{row["error_type"]}`；[部分响应与错误]({row["path"]})。不能用已完成子集估计整轮性能。']
    if diagnostic:
        lines += ['', '## 旧分流诊断，不能充当均衡布局对照', '',
            '初次EP4＋EP2按实例等概率，C128初始根会话分配为EP2每owner32、EP4每owner16。以下完整结果有效，但不与均衡主表混排；修正后仍是静态路由。', '',
            '| D配置 | C | 平均TTFT ms | 平均TPOT ms | 平均E2E ms |',
            '|---|---:|---:|---:|---:|']
        for m in diagnostic:
            lines += [f'| {labels[m["arm"]]} · 实例等分 | {m["case"][8:]} | {m["ttft_ms"]:.2f} | {m["tpot_ms"]:.2f} | {m["e2e_ms"]:.2f} |']
    lines += ['', '均值为请求等权；D batch 来自正式 episode 的 CPU forward 计数，按执行步加权，不能当作时间加权并发。TTFT／E2E 主表以 HTTP 起点计算，ready 起点在 CSV 中另列。', '',
        '## 启动与资格证据', '',
        '- [当前运行目录]('+str(RUN)+')；[冻结配置]('+str(RUN/'config.json')+')；[命令与状态预算]('+str(RUN/'preflight.json')+')。',
        '- [原始结果和完成 marker]('+str(RUN/'results')+')；资格结果位于运行目录 `evidence/<arm>/serial_qualification.json`，必须通过才进入性能阶段。',
        '- [GDN／KV 实际分配](slot_allocations.csv)；[运行标记](runtime_markers.csv)；[本地／跨节点 D 实例分项](instance_metrics.csv)；[完整性能与分位数](full_metrics.csv)；[正式 forward batch分布](forward_batches.csv)。未产生的文件暂不可用。',
        '- [当前分析与原始哈希](pilot_analysis.json)；[复算脚本](analyze_pilot.py)。', '',
        '- [C128 即时诊断样本](diagnostic_snapshot.json)：EP2 曾出现 KV 约98%、GDN 约21%，同时 P 排队；包含接收预分配，仅为日志样本，不是 episode 均值或因果证明。', '',
        '## 保留的失败', '',
        '- 最初默认 Guard 不接受 Qwen profile，在 GPU claim 前停止，无 GPU 数据。',
        '- E7/V4 r2：准备脚本把 ASCII控制值初始化为 binary0；首个资格请求触发 capture控制错误。两节点原生 worker清理通过；中断的 supervisor没有成功postcheck。',
        '- E7/V4 r3：原生 PP2→EP2 第一条512-token资格请求失败，日志为 `layers_current_pp_stage=5, len(dst_k_ptrs)=0`。8个rank均成功分配256GDN槽，错误来自 MHA 层索引映射，正式性能结果0。原生 cleanup通过，Guard确认释放；科学postcheck失败按原样保留。',
        '- E8：修复 Hybrid MHA KV全局层映射，28项CPU回归通过；实际GPU正确性和正式episode以本页当前证据为准。',
        '- E8 首个布局 C128：native bootstrap 120s／接收等待300s超时；[首次错误与队列样本](failure_diagnostics.json)。旧实例等分会使EP2每owner负载加倍，不能仅归因于GDN或KV预算。3×EP2均匀池保持原路由，EP4＋EP2另补owner均衡对照。', '',
        '这些失败不计入平均值。旧 E5/V3、E7/V3 和新 E8/V4 的性能分别归档。', '',
        '## 逐次运行与清理状态', '']
    for source in sources:
        r=Path(source['run'])
        lines += [f'- [{r.name}]({r})：`{source["progress"]["stage"]}`；[配置]({r/"config.json"})；'+
            (f'[native清理证据]({r/"task/cleanup.json"})' if source['cleanup'] else '清理待完成')+'。']
    lines += ['',
        '[返回简明方案](REPORT.md)；[已有C8／C128 workload与缓存分布](../agentx_workload_profile_20261031/REPORT.md)。','']
    # Only emit links to generated metric files that actually exist.
    text='\n'.join(lines)
    for name in ('slot_allocations.csv','runtime_markers.csv','instance_metrics.csv','full_metrics.csv','forward_batches.csv'):
        if not (HERE/name).exists():
            text=re.sub(r'\[([^\]]+)\]\('+re.escape(name)+r'\)',r'\1（待生成）',text)
    if not (RUN/'results').exists():
        text=text.replace('[原始结果和完成 marker]('+str(RUN/'results')+')','原始结果和完成 marker（待生成）')
    (HERE/'STATUS.md').write_text(text)
    print(json.dumps(dict(progress=progress,accepted=len(metrics),allocations=len(allocations),cleanup=cleanup)))


if __name__=='__main__':
    main()
