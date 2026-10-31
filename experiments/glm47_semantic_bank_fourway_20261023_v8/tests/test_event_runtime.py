"""CPU-only external-event lifecycle tests; these are not GPU measurements."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
spec=importlib.util.spec_from_file_location('event_runtime_test',ROOT/'runtime/stage_activity.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def test_each_replay_is_read_before_event_overwrite_including_drafts():
    calls=[]
    class Event:
        def __init__(self,label,timestamp):self.label=label;self.timestamp=timestamp
        def synchronize(self):calls.append(('complete',self.label))
        def elapsed_time(self,other):calls.append(('read',self.label,other.label));return other.timestamp-self.timestamp
    def pair(label,start,end):
        return dict(label=label,layer_id=1,name=label,stream=9,start=Event(label+'start',start),end=Event(label+'end',end))
    v=dict(anchor=pair('graph',10,20),pairs=[pair('attention',11,13),pair('expert',14,19)])
    first=r.read_variant(v)
    # A second replay re-records every event: the durable Python row must keep
    # the first replay's values rather than reference mutable Event objects.
    v['anchor']['start'].timestamp=30;v['anchor']['end'].timestamp=45
    for p in v['pairs']:p['start'].timestamp+=20;p['end'].timestamp+=23
    second=r.read_variant(v)
    assert first['graph_span_ms']==10 and second['graph_span_ms']==15
    assert first['intervals'][0]['elapsed_ms']==2 and second['intervals'][0]['elapsed_ms']==5
    assert [x for x in calls if x[0]=='complete']==[('complete','graphend')]*2
    assert first['component_fences']==second['component_fences']==0
    assert first['read_before_same_graph_reuse']

def test_new_pair_is_distinct_external_timing_and_actual_current_stream():
    created=[]
    def event(**kwargs):
        assert kwargs==dict(enable_timing=True,external=True)
        value=object();created.append(value);return value
    torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda:SimpleNamespace(cuda_stream=765),Event=event))
    a=r.new_pair(torch,'attention',layer_id=2);b=r.new_pair(torch,'attention',layer_id=3)
    assert len({id(x) for x in created})==4 and a['stream']==b['stream']==765
    assert a['layer_id']==2 and b['layer_id']==3

def functions():
    tree=ast.parse((ROOT/'runtime/stage_activity.py').read_text())
    return {node.name:ast.unparse(node) for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}

def test_capture_skips_warmups_and_keeps_backend_shape_role_generation():
    f=functions();capture=f['capture'];instrumented=f['instrumented']
    assert instrumented.index('is_current_stream_capturing()')<instrumented.index('new_pair(')
    assert 'id(self), shape_key, role, generation' in capture
    assert "self._glm_event_variants[shape_key] = v" in capture
    assert 'synchronize(' not in instrumented and 'forward_fn()' in instrumented
    assert 'pending' in capture and 'Incomplete actual event Graph capture' in capture

def test_replay_reads_every_actual_graph_before_return_not_outer_draft_only():
    f=functions();replay=f['replay']
    assert replay.index('original_replay(')<replay.index('read_variant(v)')<replay.index('return value')
    assert 'target_count' not in replay # Actual draft multiplicity is not guessed/truncated.
    assert f['read_variant'].count('.synchronize(')==1
    assert '.synchronize(' not in f['scope'] and '.synchronize(' not in f['segment']
    assert 'apply_masked_activation' not in (ROOT/'runtime/stage_activity.py').read_text()

def test_profile_only_transport_timeout_is_bounded_without_changing_production():
    tree=ast.parse((ROOT/'runtime/measure.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='batch_request')
    body=ast.unparse(fn)
    assert "cfg.get('profile_only_followup')" in body
    assert '90' in body and '180' in body and '1800' in body
    assert 'generate_batch(url, payload, timeout=timeout)' in body

def test_event_runtime_contains_no_collector_and_fences_only_read_or_idle():
    source=(ROOT/'runtime/stage_activity.py').read_text();assert 'torch.profiler' not in source
    f=functions()
    for name,body in f.items():
        if '.synchronize(' in body:assert name in ('read_variant','finish_profile')
    assert 'output_ids' in f['process'] and '.cpu(' not in f['process']
    assert "_LAYER.set(int(self.layer_id))" in f['layer']
    assert "p['end'].record()" in f['end'] and f['end'].index('value = b(')<f['end'].index("p['end'].record()")


def test_v8_preparation_cannot_create_new_production_measurements():
    source=(ROOT/'prepare_measure.py').read_text()
    assert "if args.profile_source_run is None:raise ValueError('v8 is external-event profile-only" in source
    assert "profile_collector_lifecycles=0" in source and 'external_event_qualification' in source
    assert "ROOT/'analysis/event_spans.py'" in source
