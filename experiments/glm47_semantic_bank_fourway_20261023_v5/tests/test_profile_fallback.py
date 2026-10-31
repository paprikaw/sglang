"""Profile failure cannot erase production or masquerade as completion."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import measure
import controller
from common import CONSUMER_ORDER,execution_arms


def production_rows():
    return [dict(requests=[{'actual':'timed'}],repeatability={'passed':True}),
            dict(requests=[{'actual':'repeat'}])]


def test_diagnostic_failure_preserves_both_validated_production_cohorts(tmp_path,monkeypatch):
    rows=production_rows();saved={}
    def fail(*args):raise ValueError('GPU marker event count mismatch: observed=0')
    monkeypatch.setattr(measure,'profile_once',fail)
    monkeypatch.setattr(measure,'save',lambda path,value:saved.update({Path(path).name:copy.deepcopy(value)}))
    result=measure.finish_diagnostics('unused',tmp_path,{'arm':'ep8_mtp_on'},rows)
    assert result==[rows[0]]
    for name,request in [('p4096_decode_results.json','timed'),('p4096_decode_repeat_results.json','repeat')]:
        assert saved[name]['performance_valid'] is True
        assert saved[name]['profile_valid'] is False
        assert saved[name]['requests']==[{'actual':request}]
    assert saved['profile_failure.json']['profile_only'] is True
    assert 'observed=0' in saved['profile_failure.json']['message']


def test_success_requires_explicit_strict_profile_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(measure,'profile_once',lambda *args: {'trace_exists':True})
    monkeypatch.setattr(measure,'save',lambda *args:None)
    row=measure.finish_diagnostics('unused',tmp_path,{'arm':'ep8_mtp_on'},production_rows())[0]
    assert row['performance_valid'] and not row['profile_valid']
    assert 'Strict GPU profile audit missing' in row['profile_failure']['message']
    monkeypatch.setattr(measure,'profile_once',lambda *args: {'profile_valid':True})
    row=measure.finish_diagnostics('unused',tmp_path,{'arm':'ep8_mtp_on'},production_rows())[0]
    assert row['performance_valid'] and row['profile_valid'] and row['profile_failure'] is None


def test_no_diagnostic_fallback_before_repeatability_or_for_cancellation(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(measure,'profile_once',lambda *args:calls.append('called'))
    rows=production_rows();rows[0]['repeatability']['passed']=False
    with pytest.raises(RuntimeError,match='valid repeated production'):
        measure.finish_diagnostics('unused',tmp_path,{'arm':'ep8_mtp_on'},rows)
    assert calls==[]
    def cancel(*args):raise KeyboardInterrupt()
    monkeypatch.setattr(measure,'profile_once',cancel)
    with pytest.raises(KeyboardInterrupt):
        measure.finish_diagnostics('unused',tmp_path,{'arm':'ep8_mtp_on'},production_rows())


@pytest.mark.parametrize('changes',[
    {'performance_valid':False},{'cleanup_verified':False},{'failure':{'message':'weight provenance missing'}},
    {'profile_failure':None},{'profile_valid':None},{'role':'producer'}])
def test_only_clean_profile_failure_can_continue(changes):
    row=dict(role='consumer',passed=False,performance_valid=True,profile_valid=False,
             profile_failure={'message':'actual GPU activity missing'},failure=None,cleanup_verified=True)
    assert controller.may_continue(row)
    row.update(changes)
    assert not controller.may_continue(row)


@pytest.mark.parametrize('failure_kind,expected_arms,expected_performance,expected_profiles',[
    ('one_profile',4,4,3),('all_profile',4,4,0),('performance',1,0,0),('cleanup',1,0,0)])
def test_controller_keeps_four_arm_goal_and_continues_only_diagnostics(tmp_path,monkeypatch,failure_kind,
                                                                    expected_arms,expected_performance,expected_profiles):
    cfg=dict(id='fallback-unit',reuse_semantic_bank=True,source_semantic_bank_ready={'source':'unchanged'})
    cfg['arms']=execution_arms(cfg);cfg['arm_order']=list(cfg['arms'])
    monkeypatch.setattr(controller,'config',lambda run:cfg)
    monkeypatch.setattr(controller,'authenticate',lambda *args,**kwargs:None)
    monkeypatch.setattr(controller.signal,'signal',lambda *args:None)
    monkeypatch.setattr(sys,'argv',['controller.py','--run',str(tmp_path)])
    for key in ('EXPERIMENT_GUARD_LEASE_RECEIPT','EXPERIMENT_GUARD_ALLOCATION_FILE',
                'EXPERIMENT_GUARD_ACQUIRED_NODES','EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS'):
        monkeypatch.setenv(key,'unit-only-no-GPU')
    seen=[];saved={}
    def arm(run,selected,env,cancelled):
        index=len(seen);seen.append(selected['arm'])
        failed_profile=failure_kind=='all_profile' or (failure_kind=='one_profile' and index==0)
        performance_valid=failure_kind!='performance'
        clean=failure_kind!='cleanup'
        return dict(arm=selected['arm'],role='consumer',performance_valid=performance_valid,
            profile_valid=performance_valid and not failed_profile,cleanup_verified=clean,
            profile_failure={'message':'profile only'} if failed_profile else None,
            failure={'message':'production/source gate failed'} if not performance_valid else None,
            passed=performance_valid and not failed_profile and clean)
    monkeypatch.setattr(controller,'run_arm',arm)
    monkeypatch.setattr(controller,'save',lambda path,value:saved.update({str(path.relative_to(tmp_path)):value}))
    assert controller.main()==1
    assert len(seen)==expected_arms
    result=saved['task/results.json']
    assert result['passed'] is False and result['expected']==result['expected_performance']==result['expected_profiles']==4
    assert result['performance_points']==expected_performance
    assert result['profile_points']==result['valid_points']==expected_profiles
    assert saved['task/progress.json']['stage']=='failed'
    assert saved['task/cleanup.json']['passed'] is (failure_kind!='cleanup')


def test_bootstrap_only_inside_profile_cold_after_production_saved_gate():
    tree=ast.parse((ROOT/'runtime/scheduler_hooks.py').read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Name) and node.func.id=='bootstrap']
    assert len(calls)==1
    profile_ifs=[node for node in ast.walk(tree) if isinstance(node,ast.If)
                 and ast.unparse(node.test)=="action == 'profile_cold'"]
    assert any(calls[0] in list(ast.walk(node)) for node in profile_ifs)
    profile_block=next(node for node in profile_ifs if calls[0] in list(ast.walk(node)))
    text=ast.unparse(profile_block)
    assert text.index('p4096_decode_repeat_results.json')<text.index('bootstrap(self)')<text.index('arm_profile(self')


def test_profile_once_runs_cpu_saved_trace_audit_before_profile_valid():
    tree=ast.parse((ROOT/'runtime/measure.py').read_text())
    method=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='profile_once')
    text=ast.unparse(method)
    assert text.index('audit_activity(out)')<text.index('profile_valid=True')
    assert 'TIMED' not in text


def test_runtime_invoked_activity_auditor_is_frozen_with_inputs():
    source=(ROOT/'prepare_measure.py').read_text()
    assert "ROOT/'analysis/stage_activity.py'" in source
