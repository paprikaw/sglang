"""Replay only deterministic first placements from frozen graph/router files."""
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'experiments/agentx_pd_dynamic_20261030/runs'
NAMES=[
    'two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot',
    'two_p_six_d_3ep2_e8_v4_gdn256_20261031_control',
    'two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control']


def main():
    rows=[];evidence=[];physical=[]
    for index,name in enumerate(NAMES):
        run=ROOT/name;cfg=json.loads((run/'config.json').read_text())
        source=run/'code/agentx_replay.py'
        sys.path.insert(0,str(source.parent))
        spec=importlib.util.spec_from_file_location(f'placement_replay_{index}',source)
        replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
        arm='pd_2p6d_pp2_ep4ep2' if index!=1 else 'pd_2p6d_pp2_3ep2'
        case=next(c for c in cfg['cases'] if c['id']=='agentx_c128')
        graph_data=json.loads(Path(cfg['graphs_file']).read_text())['graphs']
        graphs=[g for g in graph_data if g['id'] in case['graph_ids']]
        router=replay.Router([e for e in cfg['endpoints'][arm] if e['role']=='decode'],
                             'sticky_hash',len(graphs),case['copies'])
        vertices={}
        for copy in range(case['copies']):
            prefix=f'agentx-agentx_c128-repeat1:copy{copy}:'
            for graph in graphs:
                block=replay.build_vertices(graph,prefix)
                for v in block.values():v['admitted_at']=0
                vertices.update(block)
        initial=sorted([(key,v) for key,v in vertices.items() if replay.due_time(v,{}, {})==0])
        assert len(initial)==128
        lanes={};mapping={};counts=collections.Counter()
        for key,v in initial:
            lane=replay.assign_clone_balanced_lane(lanes,v['sid'],v['namespace'])
            target,_=router.pick(v['sid'],v['turn'],lane)
            node=target['host'];local=cfg['node_configs'][node]
            inst_index=int(target['id'].rsplit('.r',1)[1])
            layout=next(x for x in local['layouts'] if x['id']==arm)
            gpu=layout['instances'][inst_index]['gpus'][target['rank']]
            owner=f'{node}:gpu{gpu}'
            mapping[v['sid']]=owner
            counts[owner]+=1
            rows.append(dict(source_run=name,root_id=v['root_id'],conversation_id=v['sid'],
                owner=owner,instance=target['id'],dp_rank=target['rank'],
                handoff='same_node' if node==cfg['nodes'][0] else 'cross_node',
                first_input_tokens=v['turn']['input_tokens']))
        physical.append(mapping)
        paths=[source,Path(cfg['graphs_file']),run/'config.json']
        patch=run/'code/heterogeneous_owner_routing.py'
        if patch.exists():paths.append(patch)
        evidence.append(dict(run=str(run),initial_roots_per_physical_owner=dict(sorted(counts.items())),
            sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
    assert physical[1]==physical[2], 'Corrected heterogeneous pool must match physical root placements of uniform control'
    with (HERE/'initial_placements.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(scope='CPU reconstruction of first128 root placements from actual frozen graphs; not measured GPU load or future cost-aware routing',
        uniform_and_corrected_physical_placement_equal=True,runs=evidence)
    (HERE/'placement_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for e in evidence:print(Path(e['run']).name,e['initial_roots_per_physical_owner'])


if __name__=='__main__':main()
