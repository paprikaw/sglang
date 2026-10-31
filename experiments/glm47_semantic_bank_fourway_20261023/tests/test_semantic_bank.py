"""CPU-only layout/key/publication and direct-host import checks; no SGLang/GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'runtime' / 'semantic_bank.py'
spec = importlib.util.spec_from_file_location('semantic_bank_under_test', MODULE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def prompts():
    return [[index + 1] * 4097 for index in range(8)]


def identity():
    return dict(engine_commit=b.ENGINE_COMMIT, model_config_sha256='a' * 64,
                model_index_sha256='b' * 64, model_revision='frozen-model-revision')


def manifest():
    return dict(pieces=[dict(geometry=b.producer_geometry(rank), filename=f'rank{rank}.bin')
                        for rank in range(8)])


@pytest.mark.parametrize('pp,mtp', [(1, False), (1, True), (2, False), (2, True)])
def test_fourway_all_rank_component_request_head_mapping(pp, mtp):
    for rank in range(8):
        stage = rank // 4 if pp == 2 else 0
        dp = 0 if pp == 2 else rank // 4
        geom = b.geometry(rank=rank, pp_rank=stage, pp_size=pp,
                          attn_tp_rank=rank % 4, attn_tp_size=4,
                          attn_dp_rank=dp, attn_dp_size=2 if pp == 1 else 1,
                          target_start=46 * stage if pp == 2 else 0,
                          target_end=46 * (stage + 1) if pp == 2 else 92, mtp=mtp)
        plan = b.import_plan(manifest(), geom)
        assert sorted(dst for entry in plan for _, dst in entry['layer_pairs']) == list(range(len(geom['component_layers'])))
        assert all(entry['piece']['geometry']['kv_head_start'] == (rank % 4) * 2 for entry in plan)
        assert geom['request_indices'] == (list(range(8)) if pp == 2 else list(range(dp * 4, dp * 4 + 4)))
        assert ('draft/0' in geom['component_layers']) == (mtp and (pp == 1 or stage == 1))


def test_rank_partition_and_attention_tp_fail_closed():
    bad = b.producer_geometry(0)
    with pytest.raises(ValueError):
        b.geometry(rank=0, pp_rank=0, pp_size=2, attn_tp_rank=0, attn_tp_size=8,
                   attn_dp_rank=0, attn_dp_size=1, target_start=0, target_end=46, mtp=True)
    with pytest.raises(ValueError):
        b.geometry(rank=4, pp_rank=1, pp_size=2, attn_tp_rank=0, attn_tp_size=4,
                   attn_dp_rank=0, attn_dp_size=1, target_start=45, target_end=92, mtp=True)
    assert bad['component_layers'][-1] == 'target/45'


def test_exact_total_payload_no_second_full_copy():
    size = sum(8 * 63 * b.page_bytes(b.producer_geometry(rank)) for rank in range(8))
    assert size == 12287213568
    assert size / 1024**3 == 11.443359375
    for rank in range(8):
        assert 2 * b.page_bytes(b.producer_geometry(rank)) < b.MAX_SCRATCH_BYTES


def test_identity_binds_full_tail_and_lookahead_not_topology():
    source = prompts()
    original = b.bank_identity(source, identity())
    edited = copy.deepcopy(source)
    edited[0][4032] += 1
    changed = b.bank_identity(edited, identity())
    assert original['cached_target_token_sha256'] == changed['cached_target_token_sha256']
    assert original['draft_bigram_raw_sha256'] != changed['draft_bigram_raw_sha256']
    assert original['prompt_sha256'] != changed['prompt_sha256']
    edited = copy.deepcopy(source)
    edited[0][-1] += 1
    assert b.bank_identity(edited, identity()) != original


def test_identity_rejects_missing_wrong_engine_and_semantic_shape():
    for changed in ({}, dict(identity(), engine_commit='f' * 40),
                    dict(identity(), model_config_sha256='invalid')):
        with pytest.raises(ValueError):
            b.bank_identity(prompts(), changed)
    with pytest.raises(ValueError):
        b.bank_identity(prompts()[:4], identity())
    with pytest.raises(ValueError):
        b.bank_identity([list(range(4097))] * 8, identity())


def test_plan_rejects_component_missing_or_duplicated():
    geom = b.producer_geometry(4)
    missing = manifest()
    missing['pieces'] = [piece for piece in missing['pieces'] if piece['geometry']['rank'] != 4]
    with pytest.raises(ValueError, match='Missing/overlapping'):
        b.import_plan(missing, geom)
    doubled = manifest()
    doubled['pieces'].append(copy.deepcopy(doubled['pieces'][4]))
    with pytest.raises(ValueError, match='Missing/overlapping'):
        b.import_plan(doubled, geom)


def test_strided_byte_copy_preserves_target_and_draft_independently():
    source = np.arange(2 * 2 * 3 * 2 * 2 * 2, dtype=np.uint8).reshape(2, 2, 3, 2, 2, 2)
    backing = np.zeros((2, 6, 2, 2, 2, 2), dtype=np.uint8)
    destination = backing[:, 2:4]
    b.copy_page_layers(destination, source, [(0, 1), (2, 0)])
    assert np.array_equal(destination[:, :, 0], source[:, :, 2])
    assert np.array_equal(destination[:, :, 1], source[:, :, 0])
    assert not backing[:, :2].any()
    with pytest.raises(ValueError):
        b.copy_page_layers(destination, source[0], [(0, 0)])


def test_native_key_metadata_calls_engine_hash_on_bigram_and_plain_keys():
    class Key:
        def __init__(self, eagle): self.eagle = eagle
        def __len__(self): return b.CACHED
        def hash_page(self, start, end, prior):
            return hashlib.sha256(f'{self.eagle},{start},{end},{prior}'.encode()).hexdigest()
    with patch.object(b, '_native_key', side_effect=lambda tokens, eagle: Key(eagle)):
        mtp = b.native_key_metadata(prompts()[0], True)
        plain = b.native_key_metadata(prompts()[0], False)
    assert mtp['raw_token_count'] == 4033 and plain['raw_token_count'] == 4032
    assert len(mtp['page_hashes']) == 63
    assert mtp['page_hashes'] != plain['page_hashes']


def test_publication_requires_actual_two_node_exit_evidence(tmp_path):
    with pytest.raises(ValueError, match='exit/owned'):
        b.finalize_bank(tmp_path, prompts(), identity(), {})
    with pytest.raises(ValueError, match='evidence'):
        b.finalize_bank(tmp_path, prompts(), identity(), dict(producer_exited=True, cleanup_passed=True))
    proof = tmp_path / 'exit.json'
    proof.write_text('{}')
    with pytest.raises(ValueError, match='evidence'):
        b.finalize_bank(tmp_path, prompts(), identity(),
                        dict(producer_exited=True, cleanup_passed=True,
                             evidence=[dict(path=str(proof), sha256='0' * 64)] * 2))
    other = tmp_path / 'exit2.json'
    other.write_text('{}')
    with pytest.raises(ValueError, match='changed/missing'):
        b.finalize_bank(tmp_path, prompts(), identity(),
                        dict(producer_exited=True, cleanup_passed=True,
                             evidence=[dict(path=str(path), sha256='0' * 64) for path in (proof, other)]))


def test_publication_rejects_ready_mutation_and_unsafe_paths(tmp_path):
    (tmp_path / 'READY.json').write_text('{}')
    with pytest.raises(RuntimeError, match='immutable'):
        b.finalize_bank(tmp_path, prompts(), identity(), {})
    for filename in ('../outside.bin', '.', 'missing.bin'):
        with pytest.raises(ValueError):
            b._piece_path(tmp_path, filename)
    link = tmp_path / 'alias.bin'
    link.symlink_to(tmp_path / 'READY.json')
    with pytest.raises(ValueError):
        b._piece_path(tmp_path, link.name)


@pytest.mark.parametrize('corrupt_piece', [False, True])
def test_full_publication_and_reload_hash_every_rank_piece(tmp_path, corrupt_piece):
    """Tiny byte payload exercises actual publication, not a forged READY file."""
    def producer(rank):
        return dict(rank=rank, component_layers=['target/0'], kv_head_start=0, kv_head_end=2)
    def key(tokens, eagle):
        return dict(is_bigram=eagle, raw_token_count=b.CACHED + int(eagle),
                    raw_token_sha256=b.token_sha(tokens[:b.CACHED + int(eagle)]),
                    page_hashes=[hashlib.sha256(f'{tokens[0]}-{eagle}-{i}'.encode()).hexdigest()
                                 for i in range(2)], implementation='native_RadixKey.hash_page')
    proof_paths = [tmp_path / 'owned-exit-node0.json', tmp_path / 'owned-exit-node1.json']
    for path in proof_paths:
        path.write_text(json.dumps(dict(passed=True, owned_processes_exited=True)))
    proof = dict(producer_exited=True, cleanup_passed=True,
                 evidence=[dict(path=str(path), sha256=b.sha_file(path)) for path in proof_paths])
    with patch.object(b, 'REQUESTS', 2), patch.object(b, 'CACHED', 4), \
         patch.object(b, 'PAGE', 2), patch.object(b, 'HEAD_DIM', 2), \
         patch.object(b, 'producer_geometry', side_effect=producer), \
         patch.object(b, 'native_key_metadata', side_effect=key):
        semantic = prompts()[:2]
        logical = b.bank_identity(semantic, identity())
        for rank in range(8):
            geom = producer(rank)
            length = b.CACHED // b.PAGE * b.page_bytes(geom)
            payload = bytes([rank]) * length * 2
            (tmp_path / f'rank{rank}.bin').write_bytes(payload)
            record = dict(schema=b.SCHEMA, complete_rank=True, identity=logical,
                          bank_identity_sha256=b._digest(logical), geometry=geom, axes=b.AXES,
                          filename=f'rank{rank}.bin', bytes=len(payload),
                          sha256=hashlib.sha256(payload).hexdigest(), synthetic_kv=False,
                          native_hicache_quiescent=True, pending_counts={'pending': 0},
                          requests=[dict(request_index=i, offset=i * length, bytes=length,
                                         sha256=hashlib.sha256(payload[i * length:(i + 1) * length]).hexdigest(),
                                         mtp_key=key(semantic[i], True), plain_key=key(semantic[i], False))
                                    for i in range(2)])
            (tmp_path / f'rank{rank}.json').write_text(json.dumps(record))
        if corrupt_piece:
            (tmp_path / 'rank7.bin').write_bytes(bytes([99]) * len(payload))
            with pytest.raises(ValueError, match='content checksum'):
                b.finalize_bank(tmp_path, semantic, identity(), proof)
            assert not (tmp_path / 'READY.json').exists()
        else:
            published = b.finalize_bank(tmp_path, semantic, identity(), proof)
            assert published['state'] == 'READY'
            assert published['guard_post_pending'] and not published['archive_qualified']
            assert published['total_bytes'] == len(payload) * 8
            assert b.load_manifest(tmp_path, semantic, identity()) == published
            altered = copy.deepcopy(published)
            altered['pieces'][7]['geometry']['rank'] = 0
            (tmp_path / 'READY.json').write_text(json.dumps(altered))
            with pytest.raises(ValueError, match='integrity'):
                b.load_manifest(tmp_path, semantic, identity())


class Tensor:
    """Small CPU tensor facade; tests do not import torch or create CUDA state."""
    device = 'cpu'
    def __init__(self, value): self.value = np.asarray(value)
    def view(self, dtype): return Tensor(self.value.view(dtype))
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()


class Host:
    def __init__(self, corrupt_readback=False):
        self.buffer = np.zeros((2, 16, 1, 2, 2), dtype=np.uint16)
        self.next = 4
        self.freed = []
        self.corrupt_readback = corrupt_readback
    def alloc(self, count):
        result = Tensor(np.arange(self.next, self.next + count))
        self.next += count
        return result
    def free(self, indices): self.freed.extend(indices.tolist())
    def get_data_page(self, first, flat=True):
        result = self.buffer[:, int(first):int(first) + b.PAGE]
        if flat:
            result = result.flatten()
            if self.corrupt_readback:
                result[0] += 1
        return Tensor(result)


def run_tiny_import(tmp_path, *, corrupt_source=False, corrupt_readback=False):
    geom = dict(component_layers=['target/0'], request_indices=[0, 1],
                kv_head_start=0, kv_head_end=2)
    source_geom = dict(component_layers=['target/0', 'draft/0'], kv_head_start=0, kv_head_end=2)
    shape = (2, 2, 2, 2, 2, 2, 2, 2)
    values = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    path = tmp_path / 'rank0.bin'
    path.write_bytes(values.tobytes())
    key = lambda tokens, eagle: dict(is_bigram=eagle, page_hashes=['native0', 'native1'])
    piece = dict(filename=path.name, geometry=source_geom,
                 requests=[dict(request_index=i, sha256=hashlib.sha256(values[i].tobytes()).hexdigest(),
                                plain_key=key([i], False)) for i in range(2)])
    if corrupt_source:
        changed = bytearray(path.read_bytes())
        changed[0] ^= 1
        path.write_bytes(changed)
    saved = dict(pieces=[piece], bank_identity_sha256='bank', manifest_sha256='manifest')
    host = Host(corrupt_readback)
    cache = SimpleNamespace(root_node=SimpleNamespace(children={}), is_eagle=False, nodes={})
    def insert(root, key, indices, hashes):
        cache.nodes[key[0]] = indices.tolist()
        root.children[key[0]] = True
        return 0
    cache._insert_helper_host = insert
    patches = [patch.object(b, name, value) for name, value in
               [('REQUESTS', 2), ('CACHED', 4), ('PAGE', 2), ('HEAD_DIM', 2)]]
    for manager in patches: manager.start()
    try:
        with patch.dict(sys.modules, torch=SimpleNamespace(uint8=np.uint8)), \
             patch.object(b, 'load_manifest', return_value=saved), \
             patch.object(b, '_runtime', return_value=(cache, host, geom)), \
             patch.object(b, 'quiescent', return_value={'pending': 0}), \
             patch.object(b, 'native_key_metadata', side_effect=key), \
             patch.object(b, '_native_key', side_effect=lambda tokens, eagle: (tokens[0], eagle)), \
             patch.object(b, '_match', return_value=SimpleNamespace(device_indices=[], host_hit_length=4)):
            receipt = b.import_rank(SimpleNamespace(), tmp_path, [[0], [1]], {})
    finally:
        for manager in reversed(patches): manager.stop()
    return receipt, host, cache, values


def test_direct_host_import_new_slots_slices_draft_and_checks_readback(tmp_path):
    receipt, host, cache, source = run_tiny_import(tmp_path)
    assert cache.nodes == {0: [4, 5, 6, 7], 1: [8, 9, 10, 11]}
    assert receipt['page_readback_verified'] and receipt['new_native_host_slots']
    assert receipt['cpu_source_bytes_read'] == source.nbytes
    assert receipt['cpu_host_bytes_written'] == source.nbytes // 2
    assert receipt['cpu_host_readback_bytes'] == receipt['cpu_host_bytes_written']
    for index, row in enumerate(receipt['imported_requests']):
        assert row['cpu_host_sha256'] == row['source_view_sha256']
        assert len(row['page_sha256']) == 2
        for page in range(2):
            first = cache.nodes[index][page * 2]
            actual = host.buffer[:, first:first + 2].flatten().view(np.uint8)
            assert actual.tobytes() == source[index, page, :, :, :1].tobytes()


def test_direct_host_import_rejects_corrupt_source_before_radix_publication(tmp_path):
    with pytest.raises(RuntimeError, match='content checksum'):
        run_tiny_import(tmp_path, corrupt_source=True)


def test_direct_host_import_rejects_failed_host_readback(tmp_path):
    with pytest.raises(RuntimeError, match='readback checksum'):
        run_tiny_import(tmp_path, corrupt_readback=True)


def test_module_has_no_cuda_or_distributed_launch_api():
    source = MODULE.read_text()
    for forbidden in ('torch.cuda', 'cuda.synchronize', 'torch.distributed', 'subprocess', 'os.system'):
        assert forbidden not in source
