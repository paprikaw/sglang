"""Prepare bounded real-expert runs with existing, unchanged Guard contracts."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
DEST = BASE / 'compute_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def replace(text, old, new):
    assert old in text, old[:100]
    return text.replace(old, new)


def main():
    DEST.mkdir(exist_ok=True)
    for model in ('glm', 'hy3'):
        target = DEST / model
        target.mkdir(exist_ok=True)
        for source in (BASE / model).glob('*.py'):
            assert not (target / source.name).exists(), 'Do not edit an already frozen run'
            shutil.copy2(source, target / source.name)
        shutil.copy2(BASE / 'ce_rdma_r1/frozen/harness/geometry.py', target / 'communication_geometry.py')
        (target / 'matched_geometry.py').write_text('''from communication_geometry import TOKENS, layout as routed_layout

def point(model, total, cohort):
    experts,hidden=(160,5120) if model=='glm' else (192,4096)
    world=8 if cohort=='whole' else 4
    tokens=total//2 if cohort.startswith('half') else total
    slot=int(cohort[-1]) if cohort.startswith('half') else 0
    offset=slot*tokens if cohort.startswith('half') else 0
    shape=routed_layout(world,tokens,experts,offset,1024)
    resident=experts//world
    ids=[[] for _ in range(resident)]
    for source,row in zip(shape['ids'],shape['routes']):
        for token,route in zip(source,row):
            for expert in route:
                if expert<resident:ids[expert].append(token)
    counts=shape['counts'][:resident]
    assert list(map(len,ids))==counts
    assert sum(counts)==total*(2 if cohort=='equal_call' else 1)
    return dict(layout='ep8' if world==8 else 'pp2_ep4' if cohort.startswith('half') else 'ep4_equal',
        cohort=cohort,pp_slot=slot,global_logical_tokens=total,global_unique_tokens=total,
        microbatch_logical_tokens=tokens,ep_size=world,local_experts=resident,
        active_experts=sum(n>0 for n in counts),counts=counts,local_assignments=sum(counts),
        local_token_ids=ids,source_rows_per_ep_rank=tokens//world,
        attention_tp_physical_copies=1,expected_m=(tokens*8+experts)//experts,
        route_sha256=shape['route_sha256'],hidden_size=hidden,
        gemm_flops=6*hidden*1536*sum(counts))
''')
        common = (target / 'common.py').read_text()
        # Reuse existing node-local JIT artifacts, not weights or measurements.
        common = replace(common, 'run.name\n env.update', "'r1'\n env.update")
        (target / 'common.py').write_text(common)
        if model == 'glm':
            layouts = (target / 'layouts.py').read_text()
            layouts += '''\n# New matched geometry overrides only routing/count construction.\nfrom communication_geometry import TOKENS as GLOBAL_LOGICAL_TOKENS\nfrom matched_geometry import point\ndef layout(global_tokens,name,slot=0,duplication=1):\n    assert duplication==1\n    cohort='whole' if name=='ep8' else 'equal_call' if name=='ep4_equal' else 'half'+str(slot)\n    return point('glm',global_tokens,cohort)\n'''
            (target / 'layouts.py').write_text(layouts)
            worker = (target / 'worker.py').read_text()
            worker = replace(worker, 'authorization = authenticate(run)', "authorization = authenticate(run)\n    cfg=json.loads((run/'config.json').read_text())")
            worker = replace(worker, "    recv = torch.randn((40, 1024, HIDDEN), device='cuda', dtype=torch.bfloat16) * .125", "    channel=torch.arange(HIDDEN,device='cuda')[None,:]")
            worker = replace(worker, "        e = point['local_experts']", """        e = point['local_experts']
        recv=torch.zeros((e,1024,HIDDEN),device='cuda',dtype=torch.bfloat16)
        for expert,token_ids in enumerate(point['local_token_ids']):
            if not token_ids:continue
            ids=torch.tensor(token_ids,device='cuda')[:,None]
            values=(ids*7+channel*3)%61-30
            values[:,0]=ids[:,0]%128;values[:,1]=ids[:,0]//128
            recv[expert,:len(token_ids)].copy_((values/16).to(torch.bfloat16))""")
            worker = replace(worker, 'return invoke, output, runner_input', 'return invoke, output, runner_input, recv')
            worker = replace(worker, 'for b in GLOBAL_LOGICAL_TOKENS:', "for b in cfg['token_sweep']:")
            worker = replace(worker, "layout(b, 'pp2_ep4', 1)]", "layout(b, 'pp2_ep4', 1), layout(b,'ep4_equal')]")
            worker = replace(worker, 'for p, (invoke, output, _) in zip(points[:2], prepared[:2]):', 'for p, (invoke, output, _, recv) in zip(points, prepared):')
            worker = replace(worker, "for mode in ('eager', 'cuda_graph'):", "for mode in ('eager',):")
            worker = replace(worker, 'for invoke, _, _ in prepared:', 'for invoke, _, _, _ in prepared:')
            worker = replace(worker, 'samples = [[], [], []]', 'samples = [[] for _ in points]')
            worker = replace(worker, 'order = (0, 1, 2) if block % 2 == 0 else (2, 1, 0)', 'order = tuple(range(len(points))) if block % 2 == 0 else tuple(reversed(range(len(points))))')
            worker = replace(worker, 'for p, (invoke, _, _), values in zip(points, prepared, samples):', 'for p, (invoke, _, _, _), values in zip(points, prepared, samples):')
            worker = replace(worker, "'fixed_expert_capacity': 1024,", "'fixed_expert_capacity': 1024, 'actual_world_size': 1, 'timed_communication': False, 'source_copies': 1,")
        else:
            worker = (target / 'worker.py').read_text()
            worker = replace(worker, 'from guard_authorization import validate_guard', 'from guard_authorization import validate_guard\nfrom matched_geometry import point as matched_point')
            worker = replace(worker, "all_x=torch.randn((max(TOKENS),HIDDEN),device='cuda',dtype=torch.bfloat16)*.125", """ids=torch.arange(4096,device='cuda')[:,None];channel=torch.arange(HIDDEN,device='cuda')[None,:]
    values=(ids*7+channel*3)%61-30
    values[:,0]=ids[:,0]%128;values[:,1]=ids[:,0]//128
    all_x=(values/16).to(torch.bfloat16).contiguous()""")
            worker = replace(worker, 'global_tokens=None,owner=0):', 'global_tokens=None,owner=0,matched=None):')
            worker = replace(worker, '        x=all_x[:total]', "        x=all_x[torch.tensor([t for row in matched['local_token_ids'] for t in row],device='cuda')] if matched else all_x[:total]")
            worker = replace(worker, '        return meta,invoke', '        if matched:meta.update(matched)\n        return meta,invoke')
            start=worker.index('    # 14 exact global batch sizes;')
            end=worker.index('    assert len(results)==112',start)
            worker=worker[:start]+'''    # Ten primary doubling points plus four explicit tile-transition controls.
    for total in cfg['token_sweep']:
        cases=[]
        for cohort in ('whole','half0','half1','equal_call'):
            p=matched_point('hy3',total,cohort)
            cases.append(compact(p['local_assignments'],p['active_experts'],p['local_experts'],
                cohort,p['counts'],p['pp_slot'],total,0,p))
        measure(cases,('eager','cuda_graph'))
        del cases
        save(run/'profile/partial.json',dict(passed=None,rows=results))
        print('HY3_MATCHED_LAYER_DONE',total,flush=True)

'''+worker[end:]
            worker=replace(worker, "backend='HY3 BF16 production Triton expert core and LL-layout bridge'", "backend='HY3 BF16 production Triton expert core; routing/sorting excluded'")
        (target / 'worker.py').write_text(worker)
        for path in target.glob('*.py'):
            ast.parse(path.read_text())
        grids = [list(2**n for n in range(3,12)), list(2**n for n in range(4,13))] if model=='glm' else [[8,16,24,32,48,64,96,128,192,256,512,1024,2048,4096]]
        for index, grid in enumerate(grids):
            run=target/('run_'+chr(97+index));run.mkdir()
            (run/'profile/task').mkdir(parents=True)
            oldrun=BASE/model/'runs/r1'
            shutil.copy2(oldrun/'selection.json',run/'selection.json')
            config=json.loads((oldrun/'config.json').read_text())
            spec=json.loads((oldrun/'dashboard_spec.json').read_text())
            config['id']=f'{model}_expert_matched_tokens_20261024_v2_'+chr(97+index)
            config['token_sweep']=grid
            config['measurement_mode']='eager' if model=='glm' else 'eager_primary_and_cuda_graph_control'
            config['primary_doubling_tokens']=[2**n for n in range(3,13)]
            contract=config['model_cache']
            contract['reason']='Real single-layer Expert compute with communication-matched token counts and both comparison boundaries; no full model or banks'
            contract['selection']={'path':str(run/'selection.json'),'sha256':sha(run/'selection.json')}
            for key,filename in [('entrypoint','controller_r2.py'),('worker','worker.py')]:
                contract[key]={'path':str(target/filename),'sha256':sha(target/filename)}
            contract['command']=['python3',str(target/'controller_r2.py'),'--run',str(run)]
            contract['worker_command'][1]=str(target/'worker.py');contract['worker_command'][3]=str(run)
            save(run/'config.json',config)
            spec['id']=config['id'];spec['title']=config['id'];spec['model_cache']=contract
            spec['port']=24970+index if model=='glm' else 24972
            spec['runtime_dir']=str(run/'dashboard_guard')
            frozen=[]
            for item in spec['frozen_inputs']:
                path=Path(item['path'])
                if path==oldrun/'selection.json':path=run/'selection.json'
                elif path.parent==BASE/model:path=target/path.name
                frozen.append({'path':str(path),'sha256':sha(path)})
            for path in (target/'communication_geometry.py',target/'matched_geometry.py',run/'config.json'):
                frozen.append({'path':str(path),'sha256':sha(path)})
            spec['frozen_inputs']=frozen
            # Retain the exact accepted node/task/selection/cleanup contract.
            for task in spec['tasks']:
                def convert(value):
                    if isinstance(value,str):return value.replace(str(oldrun),str(run)).replace(str(BASE/model),str(target))
                    if isinstance(value,list):return [convert(x) for x in value]
                    if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
                    return value
                transformed=convert(task);task.clear();task.update(transformed)
            save(run/'dashboard_spec.json',spec)
            print(json.dumps({'model':model,'run':str(run),'expected_rows':config['expected_points'],'token_sweep':grid}))


if __name__=='__main__':
    main()
