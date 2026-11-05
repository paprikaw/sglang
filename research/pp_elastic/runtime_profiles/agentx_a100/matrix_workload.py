"""Four independent official windows per resident layout; bounded diagnostics."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import time
import traceback


def node_control(run, root, level, export=None):
    def one(node):
        cfg = root['node_configs'][node]
        cmd = [cfg['python'], str(run/'code/matrix_node.py'), '--root', cfg['matrix_local_root'], '--level', str(level)]
        if export:
            cmd += ['--export', str(export/node)]
        started = time.time_ns()
        p = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', node, shlex.join(cmd)],
                           capture_output=True, text=True, timeout=60, check=True)
        return dict(remote=json.loads(p.stdout), local_before_ns=started, local_after_ns=time.time_ns())
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(one, root['nodes']))


def snapshots(endpoints, key, http):
    with cf.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        infos = list(pool.map(lambda e: http(e['host'],e['port'],'/get_server_info',key,timeout=30), endpoints))
    return {e['id']: [{k:s.get(k) for k in ('matrix_observer','experiment_decode_service',
             'memory_usage','effective_max_running_requests_per_dp','max_mamba_cache_size')}
             for s in info['internal_states']] for e, info in zip(endpoints, infos)}


def qualify_overhead(run, root, arm, base, cfg, env, endpoints, key, body, flush, http, pair_builder):
    from agentx_replay import request_one
    pairs = pair_builder(endpoints)
    routing_plan = [dict(prefill=p['id'], prefill_rank=p['rank'],
                         decode=d['id'] if d else None, decode_rank=d['rank'] if d else None)
                    for p, d in pairs]
    evidence = []
    comparisons=[]
    selected=0
    for candidate in (1,):
        candidate_comparisons=[]
        for c in (8,128):
            tokens = [([9000+i]+body*128)[:512] for i in range(c)]
            def execute():
                begin = time.perf_counter()
                with cf.ThreadPoolExecutor(max_workers=c) as pool:
                    futures = [pool.submit(request_one, pairs[i%len(pairs)][0],pairs[i%len(pairs)][1],key,t,64,i)
                               for i,t in enumerate(tokens)]
                    records = [f.result() for f in futures]
                return dict(wall_s=time.perf_counter()-begin, requests=records,
                            mean_tpot_ms=statistics.mean(r['tpot_ms'] for r in records))
            trial_group=[]
            for trial, level in enumerate((0,candidate,candidate,0,0,candidate,candidate,0)):
                base.check(run,cfg,env,True)
                node_control(run,root,level)
                flush()
                execute()  # Identical cache and graph warmup, outside timed repeat.
                value=execute()
                row=dict(concurrency=c,trial=trial,candidate_level=candidate,collector_level=level,**value)
                evidence.append(row);trial_group.append(row)
                base.h.save(run/'evidence'/arm/'collector_overhead.json',dict(completed=False,measurements=evidence))
            for metric in ('wall_s','mean_tpot_ms'):
                off=[x[metric] for x in trial_group if x['collector_level']==0]
                on=[x[metric] for x in trial_group if x['collector_level']==candidate]
                candidate_comparisons.append(dict(candidate_level=candidate,concurrency=c,metric=metric,
                    off_mean=statistics.mean(off),on_mean=statistics.mean(on),
                    relative_change=statistics.mean(on)/statistics.mean(off)-1,
                    off_stdev=statistics.stdev(off),on_stdev=statistics.stdev(on)))
        comparisons.extend(candidate_comparisons)
        if all(x['relative_change']<=.01 for x in candidate_comparisons):
            selected=candidate
            break
    result=dict(completed=True,level=selected,comparisons=comparisons,measurements=evidence,routing_plan=routing_plan,
                policy='Try lifecycle-only level1; require mean penalty <=1% at both widths for wall and TPOT. Otherwise keep official client/routing and existing counters. Constant CPU memory monitor is present in both arms.',
                scope='Synthetic fixed 512-token prompts / 64 output; includes client scheduling noise; no zero-overhead claim')
    base.h.save(run/'evidence'/arm/'collector_overhead.json',result)
    node_control(run,root,0,run/'diagnostics'/arm)
    return 0


def run_points(namespace,run,root,arm,base,cfg,env,endpoints,key,body,flush,http):
    if arm.startswith('Q_'):
        return qualify_overhead(run,root,arm,base,cfg,env,endpoints,key,body,flush,http,namespace['warmup_pairs'])
    # Re-qualify the cheap CPU observer on each A100 layout; no H100 overhead import.
    qualification_path = run/'evidence'/arm/'collector_overhead.json'
    if not qualification_path.exists():
        qualify_overhead(run,root,arm,base,cfg,env,endpoints,key,body,flush,http,namespace['warmup_pairs'])
    gates=[json.loads(qualification_path.read_text())]
    assert gates[0]['completed']
    level=gates[0]['level']
    base.h.save(run/'collector_selection.json',dict(level=level,gates=[arm],
        gpu_events=False,optional_sample_interval_s=2 if level==2 else None))
    from cache_analysis import analyze_arm
    from matrix_report import update_report
    env.update(AIPERF_DATASET_WEKA_PARALLEL_WORKERS='4', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_CACHE=root['agentx_tokenizer_cache'])
    failures=[]
    for c in root['matrix_points'][arm]:
        relative=f'{arm}/C{c:03d}'
        out=run/'results'/relative
        if (out/'completed.json').exists():
            assert json.loads((out/'metrics.json').read_text())['qualified']
            continue
        out.mkdir(parents=True,exist_ok=True)
        local_output=Path(cfg['matrix_local_root'])/'client'/arm/f'C{c:03d}'/'replay'
        child=None; started=time.monotonic()
        try:
            base.check(run,cfg,env,True)
            node_control(run,root,0)
            flush()
            reset=snapshots(endpoints,key,http)
            for states in reset.values():
                for s in states:
                    o=s['matrix_observer']
                    assert o and o['fully_idle'] and not o['waiting'] and not o['running'],o
                    assert not o.get('full_evictable_size',0) and not o.get('mamba_evictable_size',0),o
            base.h.save(out/'cache_reset.json',dict(passed=True,states=reset,
                protocol='Native fully-idle flush resets tree/host pools/GDN; independent warmup follows'))
            before=namespace['forward_snapshot'](run,root,arm)
            clock=node_control(run,root,level)
            base.h.save(out/'point_manifest.json',dict(layout=arm,concurrency=c,collector_level=level,
                started_unix_ns=time.time_ns(),node_clock_roundtrip=clock,
                source_commit=cfg['engine_commit'],protocol=root['protocol']))
            command=[root['agentx_python'],str(run/'harness/run.py'),'run','--config',str(run/'endpoints'/f'{arm}.json'),
                '--model',cfg['model_path'],'--tokenizer',root['agentx_tokenizer'],'--local-dataset',root['agentx_dataset'],
                '--output',str(local_output),'--entries',str(root['dataset_entries']),'--concurrency',str(c),
                '--duration',str(root['profile_seconds']),'--warmup-grace',str(root['warmup_grace_seconds']),
                '--warmup-prefill-limit',str(root['warmup_only_prefill_limit']),
                '--wall-timeout',str(root['replay_timeout_seconds']),'--smoke','--seed','42']
            base.h.save(run/'task/progress.json',dict(experiment_id=root['id'],task_id='comparison',
                stage='official_warmup_and_profile_'+relative,timestamp=time.time(),
                valid_points=len(list(run.glob('results/*/*/completed.json'))),expected=root['expected_points']))
            with (out/'wrapper.log').open('wb') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    base.check(run,cfg,env,True)
                    if time.monotonic()-started > root['replay_timeout_seconds']+90:
                        raise TimeoutError('Point exceeded replay and cleanup budget')
                    time.sleep(2)
            if child.returncode:
                raise RuntimeError(f'Official replay exit {child.returncode}')
            shutil.copytree(local_output,out/'replay')
            manifest=json.loads((out/'replay/manifest.json').read_text())
            assert manifest['status']=='completed' and not manifest['adapter_active_after_cleanup']
            metrics=analyze_arm(run,relative)
            metrics.update(layout=arm,concurrency=c,collector_level=level)
            base.h.save(out/'metrics.json',metrics)
            if not metrics['qualified']:
                raise RuntimeError('Incomplete profile/cache accounting')
            after=namespace['forward_snapshot'](run,root,arm)
            base.h.save(out/'forward_counts.json',dict(before=before,after=after,
                delta=namespace['forward_delta'](before,after),includes_official_warmup=True))
            base.h.save(out/'final_states.json',snapshots(endpoints,key,http))
            node_control(run,root,0,run/'telemetry')
            base.h.save(out/'completed.json',dict(passed=True,elapsed_seconds=time.monotonic()-started,
                profile_requests=metrics['profile_requests'],finished_unix_ns=time.time_ns()))
        except Exception as e:
            failures.append(relative)
            base.h.save(out/'failed.json',dict(error_type=type(e).__name__,message=str(e),
                traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=40)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            if local_output.exists() and not (out/'replay').exists():
                shutil.copytree(local_output,out/'replay')
            update_report(run)
    base.h.save(run/'results'/arm/'layout_finished.json',dict(failures=failures,
                  completed=len(root['matrix_points'][arm])-len(failures)))
    return 0
