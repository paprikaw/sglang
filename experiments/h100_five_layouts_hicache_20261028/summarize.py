"""Independently validate and summarize the frozen five-layout HiCache runs."""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics,sys,time
sys.path.insert(0,'/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
from artifact_visibility import wait_for_stable_json
ROOT=Path(__file__).resolve().parent
LABELS={'ep4':'EP4','two_ep2':'2×EP2','four_ep1':'4×EP1','two_pp2':'2×PP2','pp4':'PP4','ep4_control':'EP4（171校准）'}
receipts=[]
def load(p):
    d,r=wait_for_stable_json(p);receipts.append(r);return d
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def csvfile(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(x):return f'{100*x:.1f}%'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ep',default='ep_178_r4');ap.add_argument('--pp',default='ep_178_pp2_r1');ap.add_argument('--extra',action='append',default=[]);ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    metrics=[];roundrows=[];capacities=[];selections=[];runchecks=[];promptsets={}
    names=[args.ep,args.pp,*args.extra]
    for name in names:
        run=ROOT/'runs'/name;cfg=load(run/'config.json')
        progress=load(run/'task/progress.json')
        for p in sorted(run.glob('results/*/metrics.*.json')):
            m=load(p);w=m['workload'];phase=w['phase'];cache=w.get('cache_mode','cold');length=w['P'];arm=w['arm']
            assert m['valid'] is True and m['profiler_enabled'] is False
            assert w['B']==64 and w['hicache'] is True
            assert len(m['rounds'])==(1 if phase=='cold_prefill' else cfg['repeats'])
            prompts=tuple(w['prompt_hashes']);assert len(prompts)==64
            if length in promptsets:assert promptsets[length]==prompts
            else:promptsets[length]=prompts
            rates=[];cachefractions=[];cpu=[];gpu=[];cpu_requests=[]
            for i,cohort in enumerate(m['rounds']):
                requests=cohort['requests'];assert len(requests)==64
                duration=max(r['finished'] for r in requests)-min(r['started'] for r in requests)
                close(duration,cohort['makespan_s'])
                expected_output=128 if phase=='decode' else 1
                assert all(r['output_tokens']==expected_output for r in requests)
                expected_prompt=length+(0 if phase=='cold_prefill' else w['Q'])
                assert all(r['prompt_tokens']==expected_prompt for r in requests)
                inp=64*w['Q']/duration;out=sum(r['output_tokens'] for r in requests)/duration
                close(inp,cohort['input_tokens_per_s']);close(out,cohort['output_tokens_per_s'])
                detail=[r.get('cached_tokens_details') for r in requests]
                if phase!='cold_prefill':assert all(isinstance(x,dict) for x in detail)
                detail=[x or {} for x in detail]
                tiers={key:sum(d.get(key,0) for d in detail) for key in ['device','host','storage']}
                assert tiers['storage']==0
                prefix=sum(min(r.get('cached_tokens') or 0,length) for r in requests)/(64*length)
                host=tiers['host']/(64*length);device=tiers['device']/(64*length)
                if cache=='cpu_replay':assert tiers['host']>0
                if phase!='cold_prefill':
                    raw=load(p.parent/f'{cache}.{phase}.r{i}.json');assert raw['performance_sample'] is True and raw['result']==cohort
                    audit=load(p.parent/f'{cache}.{phase}.r{i}.before.json')
                    states=[s for server in audit for s in server['internal_states']]
                    assert states and all(s['enable_hierarchical_cache'] is True for s in states)
                    aa=[s['benchmark_hicache_audit'] for s in states]
                    assert all(a['ongoing_writes']==0 and a['ongoing_loads']==0 for a in aa)
                    if cache=='cpu_replay':assert all(a['command']==2 and a['after']=={'full':0,'mamba':0} for a in aa)
                rate=inp if phase!='decode' else out;rates.append(rate);cachefractions.append(prefix);cpu.append(host);gpu.append(device)
                cpu_requests.append(sum(d.get('host',0)>0 for d in detail))
                roundrows.append(dict(run=name,node=cfg['node'],layout=arm,P=length,B=64,Q=w['Q'],phase=phase,cache=cache,repeat=i,makespan_s=duration,input_tokens_per_s=inp,output_tokens_per_s=out,prefix_hit_fraction=prefix,cpu_prefix_fraction=host,gpu_prefix_fraction=device,cpu_hit_requests=cpu_requests[-1],source=str(p)))
            expected_rates=[r['input_tokens_per_s'] for r in m['rounds']];close(statistics.mean(expected_rates),m['input_tokens_per_s'])
            close(statistics.mean(r['output_tokens_per_s'] for r in m['rounds']),m['output_tokens_per_s'])
            for key in ['ttft_s','e2e_s','tpot_ms']:
                values=[r[key] for c in m['rounds'] for r in c['requests'] if r[key] is not None]
                if values:close(statistics.mean(values),m[key]['mean'])
            metrics.append(dict(run=name,node=cfg['node'],layout=arm,label=LABELS[arm],P=length,B=64,Q=w['Q'],phase=phase,cache=cache,throughput=statistics.mean(rates),throughput_min=min(rates),throughput_max=max(rates),ttft_s=m['ttft_s']['mean'],e2e_s=m['e2e_s']['mean'],tpot_ms=(m['tpot_ms'] or {}).get('mean'),prefix_hit_fraction=statistics.mean(cachefractions),cpu_prefix_fraction=statistics.mean(cpu),gpu_prefix_fraction=statistics.mean(gpu),cpu_hit_requests=statistics.mean(cpu_requests),selected_optimization=m.get('selected_optimization'),source=str(p)))
        for row in cfg['layouts']:
            p=run/'evidence'/row['id']/'capacity.json'
            if not p.exists():continue
            c=load(p);stage_caps=[];qualifications=[]
            import re
            for usage in row['usage_logs']:
                text=Path(usage['path']).read_text()
                found=[int(x) for x in re.findall(r'max_total_num_tokens=(\d+)',text)]
                if found:stage_caps.extend(found)
            for p in (run/'evidence').glob(row['id']+'.r*/**/qualification/*.json'):
                q=load(p);assert q.get('passed') is True;qualifications.append(str(p))
            expected_ranks=sum(len(g) for g in row['gpu_groups'])
            assert len(qualifications)==expected_ranks
            # EP startup summary is emitted once per server; PP emits each
            # stage. DP capacities are also exported in native server_info.
            assert len(stage_caps)>=row['replicas']*(row['pp'] if row['pp']>1 else 1)
            assert min(stage_caps)==c['per_attention_instance_min_tokens']
            conservative=min(stage_caps)*row['replicas']*row['dp']
            capacities.append(dict(run=name,layout=row['id'],node=cfg['node'],ep=row['ep'],pp=row['pp'],attention_tp=1,attention_dp=row['dp'],replicas=row['replicas'],gpu_groups=row['gpu_groups'],native=c,conservative_gpu_token_pool=conservative,all_logged_rank_capacity_tokens=stage_caps,full_model_rank_qualifications=qualifications,mem_fraction_static=.92 if row['pp']>1 else cfg['mem_fraction_static'],mamba_full_model_equivalent_slots_per_gpu=cfg['local_mamba_slots']))
        for p in run.glob('results/*/selection.*.json'):
            d=load(p);assert d['selection_samples_excluded_from_formal_metrics'] is True
            selected=d['selected'];assert selected==max(d['candidates'],key=lambda x:x['mean_score'])
            _,cache,phase,_=p.name.split('.')
            for candidate in d['candidates']:
                rates=[];valid=True
                for repeat in range(2):
                    pilot=load(p.parent/f"pilot.{cache}.{phase}.p{candidate['policy']}.mb{candidate['microbatch']}.r{repeat}.json")
                    assert pilot['performance_sample'] is False and pilot['optimization_selection'] is True
                    assert len(pilot['result']['requests'])==64
                    assert all(r['output_tokens']==(128 if phase=='decode' else 1) for r in pilot['result']['requests'])
                    if phase=='decode':
                        valid=valid and pilot['output_equal_to_control'] is True
                        assert all(r['output_tokens']==128 for r in pilot['result']['requests'])
                    rates.append(pilot['result']['input_tokens_per_s' if phase=='prefill' else 'output_tokens_per_s'])
                close(statistics.mean(rates) if valid else -1,candidate['mean_score'])
            formal=p.parent/f'metrics.{cache}.{phase}.json'
            if formal.exists():assert load(formal)['selected_optimization']==selected
            selections.append(dict(run=name,workload=p.parent.name,selection=p.name,selected=selected,candidates=d['candidates'],source=str(p)))
        frozen=load(run/'dashboard_spec.json')['frozen_inputs']
        mismatches=[r['path'] for r in frozen if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']]
        assert not mismatches,mismatches
        check=dict(run=name,stage=progress['stage'],points=len([m for m in metrics if m['run']==name]),expected=cfg['expected_points'],frozen_sha256_passed=True)
        if not args.partial:
            assert check['points']==check['expected'] and check['stage']=='complete',check
            cleanup=load(run/'task/cleanup.json');assert cleanup['passed'] and cleanup['gpu_idle'] and cleanup['model_processes_exited'];check['cleanup']=cleanup
            posts=sorted((run/'dashboard_guard/checks').glob('*-post.json'))
            assert posts;post=load(posts[-1]);assert post['passed'] is True;check['guard_postcheck']=str(posts[-1])
            log=(run/'supervisor.log').read_text();assert '"gpu_registry_state": "released"' in log
            check['guard_lease_released']=True
        runchecks.append(check)
    summary=dict(created_at=time.time(),status='partial' if args.partial else 'complete',runs=runchecks,metrics=metrics,capacities=capacities,pp_selections=selections)
    target=ROOT/('partial_summary.json' if args.partial else 'summary.json');save(target,summary)
    csvfile(ROOT/('partial_metrics.csv' if args.partial else 'full_metrics.csv'),roundrows)
    save(ROOT/('partial_validation.json' if args.partial else 'validation.json'),dict(passed=True,partial=args.partial,checks=runchecks,stable_artifact_receipts=receipts,prompt_hashes_identical_across_layouts=True,metric_recomputation_passed=True,cache_tier_and_idle_audit_passed=True))
    for c in runchecks:print(c)
    for m in metrics:
        if m['phase']!='cold_prefill':print(m['label'],m['P'],m['cache'],m['phase'],round(m['throughput'],2),'CPU',pct(m['cpu_prefix_fraction']),'GPU',pct(m['gpu_prefix_fraction']))
    if args.partial:return
    doc=['# 35B：五种四卡部署，开启 HiCache','', 'Qwen3.6-35B-A3B；每种部署合计 4×H100 80GB、节点内 NVLink。全局 64 请求，缓存前缀 P=8,256 或 130,304；Prefill 每请求新增 Q=512、输出 1 token，Decode 新增 Q=1、输出 128 tokens。每项 3 轮正式样本；吞吐越高越好。','', '所有部署启用原生 GPU/CPU HiCache，write_through/kernel/page_first；每张物理 GPU 的 CPU 池为 64 decimal GB Attention KV + 8 decimal GB GDN。GPU 前缀主动移出只发生在受控 CPU 重放开始前，传回 GPU 的时间包含在吞吐与延迟内。CPU 写入 fence 与 GPU 主动移出位于 cohort 计时外，因此这里是每批请求吞吐，不是包含批间准备间隔的连续服务吞吐。没有使用磁盘 L3 或 CPU checkpoint replay。','']
    for cache,title in [('natural_replay','自然前缀重放'),('cpu_replay','受控 CPU 前缀重放')]:
        for length in [8256,130304]:
            doc += [f'## {title}，P={length:,}','','| 部署 | Prefill 新 tokens/s | Decode tokens/s | Decode TTFT 秒 | Decode TPOT ms | Prefill CPU 前缀比例 | Decode CPU 前缀比例 |','|---|---:|---:|---:|---:|---:|---:|']
            for arm in [a for a in LABELS if any(m['layout']==a for m in metrics)]:
                pair={m['phase']:m for m in metrics if m['layout']==arm and m['P']==length and m['cache']==cache}
                a=pair['prefill'];b=pair['decode']
                doc.append(f"| {LABELS[arm]} | {a['throughput']:,.0f} | {b['throughput']:,.0f} | {b['ttft_s']:.3f} | {b['tpot_ms']:.2f} | {pct(a['cpu_prefix_fraction'])} | {pct(b['cpu_prefix_fraction'])} |")
            doc.append('')
    doc += ['## 解释与范围','','CPU 前缀比例为实际 native cached_tokens_details.host / (B×P)，其余命中来自 GPU；总命中和每轮明细见 full_metrics.csv。它不是 CPU 请求命中率或逐 rank 传输字节比例，CPU 命中请求数另存于 CSV。不同缓存模式使用不同生成分支与缓存历史，两模式的吞吐差不能单独归因于CPU传输。自然重放沿用该布局的原生缓存状态；这批合成请求不是生产 trace，命中率不能直接外推为生产正常命中率。','','PP2 按 Prefill/Decode、前缀长度、缓存模式，分别从 native/early-send 与 microbatch 16/8/4 的完整组合中选择有效配置；PP4 使用支持四阶段的 native/early-send 与 microbatch 16/8/4 候选。tail-local feedback 在长上下文压力下触发 slot 复用错误，已排除。PP2 lookahead 的固定消息形状路径不适用本次实际 batch；探索性的动态形状改造也未通过输出校验，均未用于正式比较，依赖该路径的 policy10 尚未取得本负载资格。每个候选两轮 pilot，选择样本排除在正式三轮之外；Decode 候选要求输出与本布局 native control 相同。这里的“最优”指本次已测试的候选，不代表全参数空间最优。PP2 的层划分为20+20，PP4为10+10+10+10；本轮没有搜索其他层划分。','','正式比较的五种布局全部在同一178节点完成，171节点的探索性候选单独留档且未混入正式结果。PP 使用 static memory fraction 0.92，预留原生 GDN 状态复制和通信工作空间；EP 为 0.94。PP Decode Graph 最大捕获 batch 为实际 microbatch 上限16。旧的未开启 HiCache 数据另存，不能与本表混成一个性能排名。','','GPU KV token 池是本配置分配值；大上下文 P=130,304 为此前 EP4 容量附近的固定负载，本轮不会将其称为所有 HiCache 配置下新的绝对最大容量。缓存恢复、排队、重算都计入服务吞吐，Expert kernel 效率不能由端到端比例单独推断。','','正式数据的 token 数、输出长度、吞吐、均值延迟均从逐请求样本独立重算；前缀 hashes 跨布局一致。Guard 后检、自有模型清理、租约释放和冻结输入 SHA256 均已通过。','','## 原始数据','','- summary.json：完整指标、容量、拓扑与 PP 选择。','- full_metrics.csv：每轮吞吐与实际 GPU/CPU 命中。','- validation.json：独立核验与稳定文件读入 SHA256 收据。']
    for name in names:doc.append(f'- [{name}]({ROOT}/runs/{name})：冻结配置、完整原始样本、调优 pilots、rank 证据、日志与 Guard 收尾。')
    (ROOT/'REPORT.md').write_text('\n'.join(doc)+'\n')

if __name__=='__main__':main()
