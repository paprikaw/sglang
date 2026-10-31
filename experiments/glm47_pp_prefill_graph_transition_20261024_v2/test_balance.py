"""CPU coverage/byte mapping tests; never create or modify an actual bank."""
import ast
from pathlib import Path
import sys
import os

import pytest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
from semantic_bank import geometry,producer_geometry,import_plan,copy_page_layers


@pytest.mark.parametrize('rank',range(8))
def test_existing_46_46_bank_covers_balanced_destination_exactly(rank):
    stage=rank//4
    dest=geometry(rank=rank,pp_rank=stage,pp_size=2,attn_tp_rank=rank%4,attn_tp_size=4,
        attn_dp_rank=0,attn_dp_size=1,target_start=0 if stage==0 else 50,
        target_end=50 if stage==0 else 92,mtp=True,target_partition=(50,42))
    manifest=dict(pieces=[dict(geometry=producer_geometry(r)) for r in range(8)])
    plan=import_plan(manifest,dest)
    found=[]
    for entry in plan:
        src=entry['piece']['geometry']
        assert (src['kv_head_start'],src['kv_head_end'])==(dest['kv_head_start'],dest['kv_head_end'])
        for src_index,dst_index in entry['layer_pairs']:
            assert src['component_layers'][src_index]==dest['component_layers'][dst_index]
            found.append(dst_index)
    assert sorted(found)==list(range(len(dest['component_layers'])))
    assert len(plan)==(2 if stage==0 else 1)
    # Producer schema is immutable; the existing files remain 46/46.
    assert producer_geometry(rank)['target_start']==stage*46


def test_layer_copy_preserves_nonzero_payload_across_stage_boundary():
    import numpy as np
    # Six axes match the page-slice helper; small dimensions isolate the index map.
    a=np.arange(2*2*46*2*2*2,dtype=np.uint16).reshape(2,2,46,2,2,2)
    b=(10000+np.arange(2*2*47*2*2*2,dtype=np.uint16)).reshape(2,2,47,2,2,2)
    dest=np.zeros((2,2,50,2,2,2),dtype=np.uint16)
    copy_page_layers(dest,a,[(i,i) for i in range(46)])
    copy_page_layers(dest,b,[(i,i+46) for i in range(4)])
    assert np.array_equal(dest[:,:,:46],a)
    assert np.array_equal(dest[:,:,46:],b[:,:,:4])


def test_native_target_partition_uses_exact_requested_ranges(monkeypatch):
    source=ROOT.parents[1]/'.engine-worktrees/glm47-pp-spec-microbatch-invariant-20261023/python/sglang/srt/distributed/utils.py'
    module=ast.parse(source.read_text())
    fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='get_pp_indices')
    namespace=dict(os=os)
    exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=[fn],type_ignores=[])),namespace)
    monkeypatch.setenv('SGLANG_PP_LAYER_PARTITION','50,42')
    assert [namespace['get_pp_indices'](92,r,2) for r in (0,1)]==[(0,50),(50,92)]


def test_invalid_or_unmatched_destination_rejected():
    args=dict(rank=0,pp_rank=0,pp_size=2,attn_tp_rank=0,attn_tp_size=4,
              attn_dp_rank=0,attn_dp_size=1,target_start=0,target_end=50,mtp=True)
    with pytest.raises(ValueError,match='cover every target layer'):
        geometry(**args,target_partition=(50,40))
    with pytest.raises(ValueError,match='global target layer range'):
        geometry(**args,target_partition=(46,46))
