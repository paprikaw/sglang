import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys
from unittest.mock import Mock

import pytest

profile=Path(__file__).parents[1]/'pp_elastic/runtime_profiles/agentx_a100'
sys.path.insert(0,str(profile))
spec=importlib.util.spec_from_file_location('reactive_cleanup_helpers',profile/'process_helpers.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


@pytest.mark.parametrize('whole_node,app,known,expected',[
    (True,'999,GPU-other,42',[],False),
    (False,'999,GPU-other,42',[],True),
    (False,'101,GPU-owned,42',[],False),
    (False,'202,GPU-owned-child,42',[202],False),
    (False,'',[],True),(True,'',[],True),
])
def test_instance_cleanup_rejects_own_cuda_but_allows_the_retained_server(monkeypatch,whole_node,app,known,expected):
    monkeypatch.setattr(h.subprocess,'run',lambda *a,**kw:NS(stdout=app))
    monkeypatch.setattr(h,'owned_group',lambda _:[])
    result=h.gpu_cleanup_evidence([dict(pid=101)],require_node_gpu_idle=whole_node,known_member_pids=known)
    assert result['cleanup_verified'] is expected
    assert result['node_gpu_idle_observed'] is (not bool(app))


@pytest.mark.parametrize('whole_node',[True,False])
def test_cleanup_retains_the_strict_default_and_forwards_explicit_instance_scope(monkeypatch,whole_node):
    observed=[]
    def evidence(ids,**kw):
        observed.append(kw['require_node_gpu_idle'])
        return dict(cleanup_verified=True)
    monkeypatch.setattr(h,'gpu_cleanup_evidence',evidence)
    kwargs={} if whole_node else dict(require_node_gpu_idle=False)
    assert h.cleanup([],**kwargs)['cleanup_verified']
    assert observed==[whole_node]


def test_teardown_remembers_a_cuda_child_after_its_proc_entry_disappears(monkeypatch):
    identity=dict(pid=101,pgid=101,start_ticks=1)
    groups=iter([[dict(pid=202,state='R')]])
    monkeypatch.setattr(h,'process_record',lambda _:None)
    monkeypatch.setattr(h,'owned_group',lambda _:next(groups,[]))
    monkeypatch.setattr(h.os,'killpg',Mock())
    observed=[]
    def evidence(ids,**kw):
        observed.append(set(kw['known_member_pids']))
        return dict(cleanup_verified=True)
    monkeypatch.setattr(h,'gpu_cleanup_evidence',evidence)
    child=dict(identity=identity,process=NS(wait=lambda **kw:0),log_thread=NS(join=lambda **kw:None,is_alive=lambda:False))
    assert h.cleanup([child],require_node_gpu_idle=False)['cleanup_verified']
    assert observed==[{101,202}]
