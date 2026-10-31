"""CPU checks of the actual TBO metadata methods, with small vector stand-ins.

AST extraction avoids importing CUDA extensions on a login node. GPU numerical
and real-model checks are separate, under Experiment Guard.
"""
import argparse
import ast
import dataclasses
import __future__
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / ".engine-worktrees/qwen35-ep2-tbo-20261027/python/sglang/srt"


class Vector(list):
    @property
    def shape(self):
        return (len(self),)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Vector(result) if isinstance(key, slice) else result

    def sum(self):
        return sum(self)


class Mode:
    def is_target_verify(self):
        return False


def extract(path, node):
    tree = ast.parse(path.read_text())
    chosen = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == node)
    return chosen


def methods():
    fields = extract(ENGINE / "model_executor/forward_batch_info.py", "ForwardBatch")
    names = [n.target.id for n in fields.body if isinstance(n, ast.AnnAssign)]
    batch_type = dataclasses.make_dataclass("ForwardBatch", [(n, object, dataclasses.field(default=None)) for n in names])
    namespace = dict(ForwardBatch=batch_type, dataclasses=dataclasses,
        get_parallel=lambda:SimpleNamespace(attn_tp_size=1,moe_dense_tp_size=1),
        _is_hip=False, split_spec_info=lambda **kwargs:None,
        _compute_extend_num_tokens=lambda *args:None, SimpleNamespace=SimpleNamespace)
    fn = extract(ENGINE / "batch_overlap/two_batch_overlap.py", "filter_batch")
    cls = ast.ClassDef(name="Preparer", bases=[], keywords=[], body=[fn], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    exec(compile(module,"native_filter_batch", "exec",flags=__future__.annotations.compiler_flag),namespace)
    fn = extract(ENGINE / "layers/attention/tbo_backend.py", "_build_tbo_child_replay_fb_view")
    module = ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    exec(compile(module,"native_replay_child_view", "exec",flags=__future__.annotations.compiler_flag),namespace)
    return batch_type, namespace["Preparer"].filter_batch, namespace["_build_tbo_child_replay_fb_view"]


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_type, method, replay = methods()
        cls.filter = staticmethod(method)
        cls.replay = staticmethod(replay)

    def parent(self, n=4):
        v = lambda offset:Vector(range(offset,offset+n))
        return self.batch_type(input_ids=v(100),positions=v(4096),out_cache_loc=v(200),
            req_pool_indices=v(10),seq_lens=v(4097),seq_lens_cpu=v(4097),
            mamba_track_indices=v(30),mamba_track_mask=Vector([True,False]*((n+1)//2))[:n],
            mamba_track_seqlens=v(4096),batch_size=n,forward_mode=Mode(),global_dp_buffer_len=n)

    def child(self,parent,start,end):
        return self.filter(parent,start_token_index=start,end_token_index=end,
            start_seq_index=start,end_seq_index=end,out_num_token_non_padded=end-start)

    def test_disjoint_request_states_and_original_unchanged(self):
        p=self.parent()
        before=list(p.mamba_track_indices)
        a,b=self.child(p,0,2),self.child(p,2,4)
        self.assertEqual(a.mamba_track_indices,[30,31])
        self.assertEqual(b.mamba_track_indices,[32,33])
        self.assertEqual(a.mamba_track_mask,[True,False])
        self.assertEqual(b.mamba_track_seqlens,[4098,4099])
        self.assertEqual(p.mamba_track_indices,before)
        self.assertEqual(a.tbo_parent_token_range,(0,2))
        self.assertEqual(b.tbo_parent_token_range,(2,4))

    def test_padding_keeps_child_real_counts(self):
        p=self.parent();p._original_batch_size=3
        self.assertEqual(self.child(p,0,2)._original_batch_size,2)
        self.assertEqual(self.child(p,2,4)._original_batch_size,1)

    def test_empty_child(self):
        child=self.child(self.parent(),0,0)
        self.assertEqual(child.batch_size,0)
        self.assertEqual(child.mamba_track_indices,[])
        self.assertEqual(child.tbo_padded_len,0)

    def test_invalid_state_count_rejected(self):
        p=self.parent();p.mamba_track_mask=Vector([False])
        with self.assertRaises(AssertionError):self.child(p,0,2)

    def test_unhandled_cow_is_not_silently_dropped(self):
        p=self.parent();p.mamba_cow_src_indices=Vector([15])
        with self.assertRaisesRegex(Exception,"mamba_cow_src_indices"):self.child(p,0,2)

    def test_replay_tracks_match_eager_split(self):
        p=self.parent()
        for start,end in [(0,2),(2,4)]:
            child=self.replay(p,child_bs=end-start,seq_slice=slice(start,end),
                tok_slice=slice(start,end),token_num_per_seq=1)
            eager=self.child(p,start,end)
            for name in ["input_ids","req_pool_indices","mamba_track_indices","mamba_track_mask","mamba_track_seqlens"]:
                self.assertEqual(getattr(child,name),getattr(eager,name),name)

    def test_optional_tracking(self):
        p=self.parent();p.mamba_track_indices=p.mamba_track_mask=p.mamba_track_seqlens=None
        child=self.replay(p,child_bs=2,seq_slice=slice(0,2),tok_slice=slice(0,2),token_num_per_seq=1)
        self.assertIsNone(child.mamba_track_indices)
        self.assertIsNone(self.child(p,0,2).mamba_track_mask)


if __name__ == "__main__":
    unittest.main(verbosity=2)
