"""CPU-only tests of measured event validation and exclusive accounting."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'analysis/event_spans.py'
SPEC = importlib.util.spec_from_file_location('tested_event_spans', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def interval(label, a, b, layer=0):
    return dict(label=label, layer_id=layer, stream=7, start_ms=a, end_ms=b, elapsed_ms=b-a)


def replay(role='target', layout='ep8', rank=0):
    layers = [0] if role == 'draft' else list(range(92)) if layout == 'ep8' else list(range(46) if rank < 4 else range(46,92))
    rows = []
    for i, layer in enumerate(layers):
        a = i * 10.0
        rows.append(interval('attention', a, a+2, layer))
        if role == 'target' and layer < 3:
            rows.append(interval('dense_ffn', a+2, a+9, layer))
        else:
            for label, x, y in [('deepep_dispatch',2,3), ('expert_routed',3,6),
                                ('expert_shared',6,8), ('deepep_combine',8,9)]:
                rows.append(interval(label,a+x,a+y,layer))
    return dict(index=0, segment_index=0, role=role, phase='target_decode' if role=='target' else 'draft',
        target_ordinal=1, graph_span_ms=len(layers)*10.0, anchor_stream=7,
        intervals=rows, layer_ids=layers, batch_size=4, request_ids=['a','b','c','d'],
        completion_fences=1, component_fences=0, read_before_same_graph_reuse=True)


def rank_fixture():
    segments=[];replays=[];accepted=[]
    for i in range(12):
        row=replay();row.update(index=i,segment_index=i,target_ordinal=i+1)
        replays.append(row)
        segments.append({k:row[k] for k in ('index','target_ordinal','phase','request_ids','batch_size')})
        accepted.append(dict(target_segment_index=i,output_tokens_by_request={rid:1 for rid in row['request_ids']},output_tokens=4))
    raw=dict(rank=0,layout='ep8',mtp=False,arm='ep8_mtp_off',measurement=M.MEASUREMENT,
        cupti_enabled=False,production_timing_affected=False,diagnostic_cadence_changed=True,
        segments=segments,replays=replays,accepted=accepted,pp_boundaries=[])
    receipt=dict(passed=True,rank=0,arm='ep8_mtp_off',measurement=M.MEASUREMENT)
    return raw,receipt


class EventSpanTests(unittest.TestCase):
    def test_rank_fixture_uses_actual_outputs_and_twelve_replays(self):
        raw,receipt=rank_fixture();value=M.analyze_rank(raw,receipt)
        self.assertTrue(value['passed']);self.assertEqual(value['target']['count'],12)
        self.assertIsNone(value['draft'])
        self.assertTrue(all(x['actual_committed_local_output_tokens']==4 for x in value['replays']))

    def test_missing_or_inconsistent_actual_outputs_rejected(self):
        for kind in ('missing','sum','requests'):
            raw,receipt=rank_fixture()
            if kind=='missing':raw['accepted'].pop()
            elif kind=='sum':raw['accepted'][0]['output_tokens']=12
            else:raw['accepted'][0]['output_tokens_by_request']={'wrong':4}
            with self.assertRaises(ValueError):M.analyze_rank(raw,receipt)

    def test_absent_draft_cannot_qualify_mtp(self):
        raw,receipt=rank_fixture();raw['mtp']=True
        for x in raw['segments']+raw['replays']:x['phase']='target_verify'
        with self.assertRaises(ValueError):M.analyze_rank(raw,receipt)

    def test_union_deduplicates_nested_shared_scope(self):
        p=M.partition(10,[interval('expert_shared',1,8),interval('expert_shared',2,6)])
        self.assertEqual(p['expert_shared'],7)
        self.assertEqual(p['other_graph'],3)

    def test_nested_attention_tp_is_communication(self):
        p=M.partition(10,[interval('attention',1,9),interval('nccl_tp',3,5)])
        self.assertEqual(p['attention'],6)
        self.assertEqual(p['nccl_tp'],2)
        self.assertEqual(p['overlap'],0)

    def test_cross_category_overlap_is_not_guessed(self):
        p=M.partition(10,[interval('expert_shared',1,6),interval('deepep_dispatch',4,8)])
        self.assertEqual(p['expert_shared'],3)
        self.assertEqual(p['deepep_dispatch'],2)
        self.assertEqual(p['overlap'],2)
        self.assertEqual(sum(p.values()),10)

    def test_ep_target_layer_and_partition_coverage(self):
        a=M.analyze_replay(replay(),layout='ep8',rank=0)
        self.assertEqual(a['layer_count'],92)
        self.assertAlmostEqual(sum(a['exclusive_ms'].values()),920)

    def test_pp_stage_layer_ranges_and_draft_local_zero(self):
        for rank in (0,4):
            a=M.analyze_replay(replay(layout='pp2ep4',rank=rank),layout='pp2ep4',rank=rank)
            self.assertEqual(a['layer_count'],46)
        self.assertEqual(M.analyze_replay(replay('draft'),layout='ep8',rank=0)['layer_count'],1)

    def test_missing_layer_or_component_rejected(self):
        for change in ('layer','component'):
            row=replay()
            if change=='layer':row['layer_ids'].pop()
            else:row['intervals']=[x for x in row['intervals'] if not (x['layer_id']==5 and x['label']=='attention')]
            with self.assertRaises(ValueError):M.analyze_replay(row,layout='ep8',rank=0)

    def test_invalid_timestamp_or_independent_elapsed_rejected(self):
        for key,value in [('start_ms',float('nan')),('end_ms',99999),('elapsed_ms',-1),('elapsed_ms',5)]:
            row=replay();row['intervals'][0][key]=value
            with self.assertRaises(ValueError):M.analyze_replay(row,layout='ep8',rank=0)

    def test_completion_reuse_and_cross_stream_checks(self):
        for key,value in [('completion_fences',0),('component_fences',1),('read_before_same_graph_reuse',False)]:
            row=replay();row[key]=value
            with self.assertRaises(ValueError):M.analyze_replay(row,layout='ep8',rank=0)
        row=replay();row['intervals'][0]['stream']=8
        with self.assertRaises(ValueError):M.analyze_replay(row,layout='ep8',rank=0)

    def test_actual_b4_required(self):
        row=replay();row['request_ids']=['a','a','b','c']
        with self.assertRaises(ValueError):M.analyze_replay(row,layout='ep8',rank=0)

    def test_zero_comm_or_expert_is_not_filled_from_estimate(self):
        row=replay();row['intervals']=[x for x in row['intervals'] if x['label']!='deepep_combine']
        with self.assertRaises(ValueError):M.analyze_replay(row,layout='ep8',rank=0)


if __name__=='__main__':unittest.main()
