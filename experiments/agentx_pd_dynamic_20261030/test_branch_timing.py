"""CPU checks executing pinned native method ASTs and the new replay loop.

Only the three branch methods and their two timestamp helpers are extracted,
unchanged, to avoid installing the full harness runtime. The tests do not claim
end-to-end official harness equivalence.
"""
import ast
import asyncio
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
HARNESS=ROOT.parents[1]/'datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8'
sys.path.insert(0,str(HARNESS/'src'))
from aiperf.common.enums import ConversationBranchMode, CreditPhase
from agentx_branch_timing import branch_dispatch_rules,build_vertices,due_time
import agentx_replay_v4 as replay

native_path=HARNESS/'src/aiperf/timing/branch_orchestrator.py'
native_tree=ast.parse(native_path.read_text())
helpers=[n for n in native_tree.body if isinstance(n,ast.FunctionDef)
         and n.name in ('_as_timestamp_ms','_turn0_timestamp_ms')]
native_class=next(n for n in native_tree.body if isinstance(n,ast.ClassDef) and n.name=='BranchOrchestrator')
methods=[n for n in native_class.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
         and n.name in ('on_credit_issued','_branch_start_timestamp_ms','_child_dispatch_offset_ms')]
assert len(helpers)==2 and len(methods)==3
reference=ast.Module(body=helpers+[ast.ClassDef(name='BranchOrchestrator',bases=[],keywords=[],
                    body=methods,decorator_list=[])],type_ignores=[])
exec(compile(ast.fix_missing_locations(reference),str(native_path),'exec'),globals())


def turn(tokens, timestamp_ms, api_ms=10, delay_ms=0, branches=(), prerequisites=()):
    return dict(timestamp=timestamp_ms,api_time_ms=api_ms,delay=delay_ms,
        input_tokens=tokens,output_tokens=2,hash_ids=[],source_outer_idx=tokens,
        source_inner_idx=None,source_kind='fixture',branch_ids=list(branches),
        prerequisites=list(prerequisites),replay_predecessors=[])


def graph(child_start=5,parent_api=10,mode='spawn'):
    return dict(id='parent',block_size=64,conversations=[
        dict(session_id='parent',branches=[dict(branch_id='b',mode=mode,
            dispatch_timing='post',child_conversation_ids=['child'],
            start_timestamp_ms=child_start,is_background=False)],
            turns=[turn(1,0,parent_api,branches=['b']),
                   turn(3,20,10,10,prerequisites=[dict(kind='spawn_join',branch_id='b')])]),
        dict(session_id='child',branches=[],turns=[turn(2,child_start)])])


async def native_rules(g):
    metadata={}
    for conv in g['conversations']:
        metadata[conv['session_id']]=NS(replay_scope_id=g['id'],
            turns=[NS(timestamp_ms=t.get('timestamp'),api_time_ms=t.get('api_time_ms'),
                      branch_ids=t['branch_ids']) for t in conv['turns']],
            branches=[NS(**dict(b,mode=ConversationBranchMode(b['mode']))) for b in conv['branches']])
    obj=object.__new__(BranchOrchestrator)
    obj._cleaning_up=False
    obj._accelerated_warmup_started=False
    obj._cs=NS(get_metadata=metadata.__getitem__)
    obj._parent_locks=defaultdict(asyncio.Lock)
    obj._overlap_dispatched_branches=set()
    observed={}
    for conv in g['conversations']:
        sid=conv['session_id']
        for i,t in enumerate(conv['turns']):
            issue_origins={}
            async def capture(credit,branch_ids,*,dispatch_origin_ms=None):
                issue_origins.update({bid:dispatch_origin_ms for bid in branch_ids})
            obj._spawn_children_and_register_gates=capture
            credit=NS(conversation_id=sid,turn_index=i,x_correlation_id=sid,
                      agent_depth=0 if sid==g['id'] else 1,phase=CreditPhase.PROFILING)
            await obj.on_credit_issued(credit)
            for bid in t['branch_ids']:
                branch=next(b for b in metadata[sid].branches if b.branch_id==bid)
                if bid in issue_origins:
                    event,origin='issued',issue_origins[bid]
                else:
                    event='complete'
                    origin=obj._branch_start_timestamp_ms(branch) if branch.mode==ConversationBranchMode.SPAWN else None
                for child in branch.child_conversation_ids:
                    offset=obj._child_dispatch_offset_ms(origin,NS(metadata=metadata[child]))
                    observed[child]=dict(anchor=(sid,i),event=event,delay_s=offset/1000)
    return observed


class BranchTimingTests(unittest.TestCase):
    def compare_native(self,g):
        actual=branch_dispatch_rules(g)
        expected=asyncio.run(native_rules(g))
        self.assertEqual(set(actual),set(expected))
        for child,ref in expected.items():
            self.assertEqual(actual[child]['anchor'],ref['anchor'])
            self.assertEqual(actual[child]['event'],ref['event'])
            self.assertAlmostEqual(actual[child]['delay_s'],ref['delay_s'],places=12)
        return actual

    def test_all_exported_graphs_match_native_branch_methods(self):
        cfg=json.loads((ROOT/'runs/eight_gpu_col_ep4_durable_r3/config.json').read_text())
        catalog=json.loads(Path(cfg['graphs_file']).read_text())['graphs']
        self.assertEqual(len(catalog),24)
        for g in catalog:
            with self.subTest(graph=g['id']):
                self.compare_native(g)
                vertices=build_vertices(g,'test:copy0:')
                self.assertEqual(len(vertices),sum(len(c['turns']) for c in g['conversations']))

    def test_overlap_is_relative_to_parent_issue(self):
        rule=self.compare_native(graph())['child']
        self.assertEqual(rule['event'],'issued')
        self.assertEqual(rule['delay_s'],.005)

    def test_forty_hour_source_gap_is_relative_to_branch_on_return(self):
        g=graph(child_start=145004210,parent_api=11568)
        rule=self.compare_native(g)['child']
        self.assertEqual((rule['event'],rule['delay_s']),('complete',0))

    def test_later_child_keeps_offset_and_missing_branch_start_falls_back(self):
        g=graph(child_start=100,parent_api=10)
        b=g['conversations'][0]['branches'][0]
        b.pop('start_timestamp_ms')
        b['child_conversation_ids'].append('later')
        g['conversations'].append(dict(session_id='later',branches=[],turns=[turn(4,30299)]))
        rules=self.compare_native(g)
        self.assertAlmostEqual(rules['later']['delay_s'],30.199)

    def test_fork_waits_for_parent_return(self):
        rule=self.compare_native(graph(mode='fork'))['child']
        self.assertEqual((rule['event'],rule['delay_s']),('complete',0))

    def test_nested_overlap_waits_for_return_as_native_does(self):
        g=graph()
        child=g['conversations'][1]
        child['branches']=[dict(branch_id='grand',mode='spawn',dispatch_timing='post',
            child_conversation_ids=['grandchild'],start_timestamp_ms=6,is_background=True)]
        child['turns'][0]['branch_ids']=['grand']
        g['conversations'].append(dict(session_id='grandchild',branches=[],turns=[turn(4,6)]))
        rule=self.compare_native(g)['grandchild']
        self.assertEqual((rule['event'],rule['delay_s']),('complete',0))

    def test_pre_session_protocol_is_explicitly_rejected(self):
        g=graph();g['conversations'][0]['branches'][0]['dispatch_timing']='pre'
        with self.assertRaisesRegex(ValueError,'Pre-session'):
            build_vertices(g,'test:copy0:')

    def test_join_requires_both_time_and_child_completion(self):
        v=build_vertices(graph(),'test:copy0:')['test:copy0:parent',1]
        v['admitted_at']=0
        self.assertIsNone(due_time(v,{('test:copy0:parent',0):1},{}))
        self.assertEqual(due_time(v,{('test:copy0:parent',0):1,('test:copy0:child',0):2},{}),2)
        self.assertEqual(due_time(v,{('test:copy0:parent',0):1,('test:copy0:child',0):.5},{}),1.01)

    def test_replay_issues_overlap_once_and_respects_join(self):
        child_started=threading.Event()
        times={}
        lock=threading.Lock()
        def request(prefill,decode,key,tokens,output,lane):
            token=tokens[0];start=time.perf_counter()
            with lock:
                self.assertNotIn(token,times)
                times[token]=[start,None]
            if token==1:
                self.assertTrue(child_started.wait(.5),'Child waited for parent completion')
                time.sleep(.005)
            elif token==2:
                child_started.set();time.sleep(.04)
            else:
                time.sleep(.005)
            end=time.perf_counter();times[token][1]=end
            elapsed=(end-start)*1000
            return dict(started=start,finished=end,ttft_ms=1,tpot_ms=elapsed-1,
                        e2e_ms=elapsed,output_ids=[1,2],output_tokens=2,meta_info={})
        with patch.object(replay,'token_ids',side_effect=lambda g,t,*args:[t['input_tokens']]),\
             patch.object(replay,'request_one',side_effect=request):
            result=replay.replay([graph()],[dict(id='cpu',ep=1,role='colocated')],
                'sticky_hash',1,'',[1],lambda:None,'test')
        self.assertEqual(sorted(times),[1,2,3])
        self.assertLess(times[2][0],times[1][1])
        self.assertGreaterEqual(times[3][0],times[2][1])
        self.assertEqual(result['completed_requests'],3)
        self.assertEqual(result['branch_timing_policy'],'official_branch_issue_and_return_v1')
        self.assertEqual(result['idle_compressed_s'],0)
        self.assertTrue(all('credit_issued_virtual_s' in r for r in result['requests']))


if __name__=='__main__':
    unittest.main(verbosity=2)
