"""Portable, immutable *semantic* GLM KV pages, never synthetic KV.

The physical files are the eight PP producer's bounded page streams.  The
manifest gives every byte logical (request, component/layer, KV head) coordinates;
consumers slice them directly without a second canonical-sized materialization.
Only CPU operations are used here.  Call export_rank/import_rank at confirmed
Scheduler idle, BEFORE the measured cohort.  Final publication is a controller
operation AFTER producer exit and Guard cleanup, not a worker operation.
"""
from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time

SCHEMA = 'glm47-semantic-bank-v2'
ENGINE_COMMIT = '9e48f876a8eb2d7460317bc713bfa8680d78aea5'
REQUESTS, RAW_PREFIX, CACHED, PAGE = 8, 4096, 4032, 64
TARGET_LAYERS, KV_HEADS, HEAD_DIM, ATTN_TP = 92, 8, 128, 4
ITEM_BYTES = 2
MAX_BANK_BYTES = 12 * 1024**3
MAX_SCRATCH_BYTES = 16 * 1024**2
AXES = ['request', 'page', 'kv', 'token', 'component_layer', 'kv_head', 'head_dim', 'item_byte']


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def token_sha(tokens):
    return hashlib.sha256(struct.pack(f'<{len(tokens)}I', *tokens)).hexdigest()


def bank_identity(prompts, identity):
    """Topology-independent identity; full input also binds draft lookahead."""
    if len(prompts) != REQUESTS or any(len(row) != RAW_PREFIX + 1 for row in prompts):
        raise ValueError('Bank requires eight exact P4096 + Q1 semantic prompts')
    if any(type(token) is not int or not 0 <= token < 2**32 for row in prompts for token in row):
        raise ValueError('Invalid token ID')
    if len({tuple(row[:PAGE]) for row in prompts}) != REQUESTS:
        raise ValueError('Distinct first pages required for bounded request ownership')
    required = ('engine_commit', 'model_config_sha256', 'model_index_sha256', 'model_revision')
    if any(not identity.get(key) for key in required):
        raise ValueError('Missing pinned model/engine identity')
    if identity['engine_commit'] != ENGINE_COMMIT:
        raise ValueError('Unqualified engine revision')
    for key in ('model_config_sha256', 'model_index_sha256'):
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Invalid {key}')
    return dict(schema=SCHEMA, model=dict(identity), dtype='bfloat16', item_bytes=ITEM_BYTES,
                page_size=PAGE, cached_tokens=CACHED, raw_prefix_tokens=RAW_PREFIX,
                target_layers=TARGET_LAYERS, draft_layers=1, kv_heads=KV_HEADS,
                head_dim=HEAD_DIM, semantic_kv=True,
                prompt_sha256=[token_sha(row) for row in prompts],
                cached_target_token_sha256=[token_sha(row[:CACHED]) for row in prompts],
                draft_bigram_raw_sha256=[token_sha(row[:CACHED + 1]) for row in prompts])


def geometry(*, rank, pp_rank, pp_size, attn_tp_rank, attn_tp_size,
             attn_dp_rank, attn_dp_size, target_start, target_end, mtp,
             target_partition=(46,46)):
    """Pure mapping, including Attention TP rather than the EP world's TP8."""
    if attn_tp_size != ATTN_TP or not 0 <= attn_tp_rank < ATTN_TP:
        raise ValueError('Only Attention TP4 is qualified')
    if pp_size == 2 and attn_dp_size == 1:
        if pp_rank not in (0, 1) or attn_dp_rank != 0 or rank != pp_rank * 4 + attn_tp_rank:
            raise ValueError('Invalid PP2 rank mapping')
        if len(target_partition)!=2 or any(type(n) is not int or n<=0 for n in target_partition) or sum(target_partition)!=TARGET_LAYERS:
            raise ValueError('PP partition must cover every target layer exactly once')
        expected = (sum(target_partition[:pp_rank]),sum(target_partition[:pp_rank+1]))
        request_indices = list(range(REQUESTS))
        has_draft = bool(mtp and pp_rank == 1)
    elif pp_size == 1 and attn_dp_size == 2:
        if pp_rank != 0 or attn_dp_rank not in (0, 1) or rank != attn_dp_rank * 4 + attn_tp_rank:
            raise ValueError('Invalid EP8 Attention DP mapping')
        expected = (0, TARGET_LAYERS)
        request_indices = list(range(attn_dp_rank * 4, attn_dp_rank * 4 + 4))
        has_draft = bool(mtp)
    else:
        raise ValueError('Only EP8 TP4DP2 or PP2xEP4 TP4 supported')
    if (target_start, target_end) != expected:
        raise ValueError('Live global target layer range differs from qualified mapping')
    layers = [f'target/{i}' for i in range(target_start, target_end)]
    if has_draft:
        layers.append('draft/0')
    return dict(rank=rank, pp_rank=pp_rank, pp_size=pp_size,
                attn_tp_rank=attn_tp_rank, attn_tp_size=attn_tp_size,
                attn_dp_rank=attn_dp_rank, attn_dp_size=attn_dp_size,
                target_start=target_start, target_end=target_end, mtp=bool(mtp),
                request_indices=request_indices, component_layers=layers,
                kv_head_start=attn_tp_rank * 2, kv_head_end=(attn_tp_rank + 1) * 2)


def producer_geometry(rank):
    return geometry(rank=rank, pp_rank=rank // 4, pp_size=2, attn_tp_rank=rank % 4,
                    attn_tp_size=4, attn_dp_rank=0, attn_dp_size=1,
                    target_start=(rank // 4) * 46, target_end=(rank // 4 + 1) * 46, mtp=True)


def page_shape(geom):
    return (2, PAGE, len(geom['component_layers']), 2, HEAD_DIM, ITEM_BYTES)


def page_bytes(geom):
    return 2 * PAGE * len(geom['component_layers']) * 2 * HEAD_DIM * ITEM_BYTES


def _native_key(tokens, eagle):
    from sglang.srt.mem_cache.radix_cache import RadixKey
    return RadixKey(array('q', tokens[:CACHED + int(eagle)]), is_bigram=bool(eagle))


def native_key_metadata(tokens, eagle):
    """Use the live engine hash implementation, including shifted draft bigrams."""
    key = _native_key(tokens, eagle)
    if len(key) != CACHED:
        raise ValueError('Native key length mismatch')
    hashes, previous = [], None
    for offset in range(0, CACHED, PAGE):
        previous = key.hash_page(offset, offset + PAGE, previous)
        hashes.append(previous)
    return dict(is_bigram=bool(eagle), raw_token_count=CACHED + int(eagle),
                raw_token_sha256=token_sha(tokens[:CACHED + int(eagle)]),
                page_hashes=hashes, implementation='native_RadixKey.hash_page')


def validate_key_record(tokens, eagle, record):
    """Controller-safe validation; consumers also recompute live native hashes.

    No SGLang/native-extension imports on the login-node controller.  The exact
    hash lists were generated by the frozen native producer and must agree on
    every producer rank, then agree with the native consumer before any import.
    """
    hashes = record.get('page_hashes', [])
    if (record.get('is_bigram') is not bool(eagle)
            or record.get('raw_token_count') != CACHED + int(eagle)
            or record.get('raw_token_sha256') != token_sha(tokens[:CACHED + int(eagle)])
            or record.get('implementation') != 'native_RadixKey.hash_page'
            or len(hashes) != CACHED // PAGE
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in '0123456789abcdef' for c in value) for value in hashes)):
        raise ValueError('Producer native key metadata/token boundary invalid')


def _piece_path(bank, filename):
    if not isinstance(filename, str) or Path(filename).name != filename or filename in ('.', '..'):
        raise ValueError('Unsafe bank filename')
    path = Path(bank) / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing/nonregular bank piece {filename}')
    return path


def _full_component():
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    return ComponentType.FULL


def _cache_mode(cache):
    # UnifiedRadixCache deliberately does not forward TreeCore.is_eagle.
    return bool(cache.tree_core.is_eagle)


def _root(cache):
    return cache.resolve_node_handle(cache.root_node_handle())


def _node_full(cache, handle):
    node = cache.resolve_node_handle(handle)
    if node is None:
        raise RuntimeError('Missing native unified radix node handle')
    return node, node.component(_full_component())


def _insert_host(cache, key, indices, hashes):
    # FULL host ownership is committed by insert_host itself. Auxiliary component
    # transfers are not needed: _runtime rejects every non-FULL component/pool.
    result = cache.tree_core.insert_host(cache.root_node_handle(), key, indices, hashes)
    cache._apply_cache_actions(result.cache_actions)
    if (result.cache_actions or result.prefix_len != 0 or result.total_len != CACHED
            or result.host_insert_dropped or result.inserted_host_node is None):
        raise RuntimeError('Unified host insert was dropped, partial, or retained actions')
    _, state = _node_full(cache, result.inserted_host_node)
    if state.host_value is None or state.value is not None:
        raise RuntimeError('Unified host insert did not commit CPU-only FULL state')
    return result


def _runtime(scheduler):
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.mem_cache.hicache_storage import PoolName
    p = get_parallel()
    from common import config
    active=config(Path(os.environ['GLM_RUN_DIR']))
    cache = scheduler.tree_cache
    host = cache.host_pool_group.get_pool(PoolName.KV)
    if (tuple(cache.tree_components) != (_full_component(),)
            or set(cache.host_pool_group.entry_map) != {PoolName.KV}
            or cache.host_pool_group.anchor_entry.host_pool is not host):
        raise RuntimeError('Bank requires native unified FULL-only single KV host pool')
    _root(cache)  # Resolve the actual native NodeId, not a legacy TreeNode.
    runner = scheduler.tp_worker.model_runner
    model = runner.model
    geom = geometry(rank=int(p.world_rank), pp_rank=int(scheduler.ps.pp_rank),
                    pp_size=int(scheduler.ps.pp_size), attn_tp_rank=int(p.attn_tp_rank),
                    attn_tp_size=int(p.attn_tp_size), attn_dp_rank=int(p.attn_dp_rank),
                    attn_dp_size=int(p.attn_dp_size), target_start=int(model.start_layer),
                    target_end=int(model.end_layer), mtp=_cache_mode(cache),
                    target_partition=active.get('target_pp_partition',(46,46)))
    if (host.layout != 'page_first' or str(host.dtype) != 'torch.bfloat16'
            or host.layer_num != len(geom['component_layers']) or host.head_num != 2
            or host.head_dim != HEAD_DIM or cache.page_size != PAGE
            or str(host.kv_buffer.device) != 'cpu'):
        raise RuntimeError('Live native host-pool layout does not match bank mapping')
    if getattr(host, 'target_layer_num', None) != geom['target_end'] - geom['target_start']:
        raise RuntimeError('Target/draft packed layout identity mismatch')
    draft_count = len(getattr(host, 'mtp_draft_device_pools', ()))
    if draft_count != int('draft/0' in geom['component_layers']):
        raise RuntimeError('Live MTP draft pool absent/unexpected')
    if cache.enable_storage:
        raise RuntimeError('Direct host import/export requires disabled L3 backend')
    return cache, host, geom


def quiescent(scheduler):
    cache = scheduler.tree_cache
    c = cache.cache_controller
    counts = {key: len(getattr(c, key)) for key in
              ('load_queue', 'write_queue', 'ack_load_queue', 'ack_write_queue')}
    counts.update({key: len(getattr(cache, key)) for key in
                   ('ongoing_load_back', 'ongoing_write_through', 'ongoing_prefetch', 'ongoing_backup')})
    counts['ongoing_tree_insert'] = int(cache.tree_core.has_ongoing_insert())
    if not scheduler.is_fully_idle() or any(counts.values()):
        raise RuntimeError(f'Bank operation requires native idle/quiescent: {counts}')
    return counts


def _match(cache, tokens):
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    return cache.match_prefix(MatchPrefixParams(key=_native_key(tokens, _cache_mode(cache))))


def _host_indices(cache, match):
    import torch
    chain, node = [], cache.resolve_node_handle(match.last_host_node)
    root, visited = _root(cache), set()
    while node is not root:
        if node is None or node.id in visited:
            raise RuntimeError('Broken/cyclic native unified host ancestry')
        visited.add(node.id)
        node, state = _node_full(cache, node)
        if state.host_value is None or state.value is not None:
            raise RuntimeError('Export requires complete CPU-only host ancestry')
        chain.append(state.host_value)
        node = cache.resolve_node_handle(node.parent)
    if not chain:
        raise RuntimeError('Empty native unified host ancestry')
    result = torch.cat(list(reversed(chain)))
    if result.numel() != CACHED or str(result.device) != 'cpu':
        raise RuntimeError('Unexpected host index coverage')
    values = result.tolist()
    if len(set(values)) != CACHED:
        raise RuntimeError('Aliased physical host slots')
    for offset in range(0, CACHED, PAGE):
        first = values[offset]
        if first % PAGE or values[offset:offset + PAGE] != list(range(first, first + PAGE)):
            raise RuntimeError('Host allocation is not physical page-contiguous')
    return result


def _host_page_array(host, first):
    import torch
    # The K/V halves are strided in the full host pool; flatten copies at most
    # one local page, never a complete request/bank.  No GPU tensor is read.
    page = host.get_data_page(int(first), flat=True)
    if str(page.device) != 'cpu':
        raise RuntimeError('Host page unexpectedly on GPU')
    return page.view(torch.uint8).numpy()


def _host_page_bytes(host, first):
    return _host_page_array(host, first).tobytes()


def export_rank(scheduler, bank_dir, prompts, identity):
    begun = time.monotonic()
    bank = Path(bank_dir)
    bank.mkdir(parents=True, exist_ok=True)
    if (bank / 'READY.json').exists():
        raise RuntimeError('Published bank is immutable')
    logical = bank_identity(prompts, identity)
    cache, host, geom = _runtime(scheduler)
    if geom != producer_geometry(geom['rank']) or not _cache_mode(cache):
        raise RuntimeError('Bank producer must be PP2xEP4 with real MTP Prefill')
    pending = quiescent(scheduler)
    rank = geom['rank']
    final = bank / f'rank{rank}.bin'
    receipt_path = bank / f'rank{rank}.json'
    if final.exists() or receipt_path.exists():
        raise RuntimeError('Refusing to mix bank producer attempts')
    temporary = bank / f'.rank{rank}.{os.getpid()}.incoming'
    stream_hash = hashlib.sha256()
    rows, position = [], 0
    with temporary.open('xb') as handle:
        for index in geom['request_indices']:
            match = _match(cache, prompts[index])
            if len(match.device_indices) or match.host_hit_length != CACHED:
                raise RuntimeError('Semantic producer prefix is not CPU4032/GPU0')
            indices = _host_indices(cache, match)
            digest = hashlib.sha256()
            start = position
            for offset in range(0, CACHED, PAGE):
                data = _host_page_bytes(host, indices[offset])
                if len(data) != page_bytes(geom):
                    raise RuntimeError('Packed target/draft page byte count mismatch')
                handle.write(data)
                digest.update(data)
                stream_hash.update(data)
                position += len(data)
            rows.append(dict(request_index=index, offset=start, bytes=position - start,
                             sha256=digest.hexdigest(),
                             mtp_key=native_key_metadata(prompts[index], True),
                             plain_key=native_key_metadata(prompts[index], False)))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, final)
    if position > MAX_BANK_BYTES:
        raise RuntimeError('Producer exceeds single bank disk payload bound')
    record = dict(schema=SCHEMA, passed=True, complete_rank=True, identity=logical,
                  bank_identity_sha256=_digest(logical), geometry=geom, axes=AXES,
                  filename=final.name, bytes=position, sha256=stream_hash.hexdigest(),
                  requests=rows, native_hicache_quiescent=True, pending_counts=pending,
                  semantic_prefill_required=True, synthetic_kv=False,
                  gpu_tensor_reads=0, bounded_scratch_bytes=2 * page_bytes(geom),
                  provenance=dict(hostname=socket.gethostname(), pid=os.getpid(),
                      run_dir=os.environ.get('GLM_RUN_DIR'),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()),
                  elapsed_s=time.monotonic() - begun)
    _save(receipt_path, record)
    return record


def _validate_piece_metadata(piece, rank, logical):
    geom = producer_geometry(rank)
    if (piece.get('schema') != SCHEMA or piece.get('complete_rank') is not True
            or piece.get('identity') != logical or piece.get('bank_identity_sha256') != _digest(logical)
            or piece.get('geometry') != geom or piece.get('axes') != AXES
            or piece.get('filename') != f'rank{rank}.bin'
            or piece.get('synthetic_kv') is not False or piece.get('native_hicache_quiescent') is not True
            or any(piece.get('pending_counts', {'missing': 1}).values())):
        raise ValueError(f'Producer rank{rank} identity/geometry/idle gate failed')
    per_request = CACHED // PAGE * page_bytes(geom)
    rows = piece.get('requests', [])
    if len(rows) != REQUESTS or piece.get('bytes') != REQUESTS * per_request:
        raise ValueError('Incomplete producer request bytes')
    for index, row in enumerate(rows):
        if (row.get('request_index') != index or row.get('offset') != index * per_request
                or row.get('bytes') != per_request or len(row.get('sha256', '')) != 64):
            raise ValueError('Noncanonical producer request byte intervals')
    return geom


def finalize_bank(bank_dir, prompts, identity, producer_exit_proof):
    """Controller-only publication, AFTER verified producer exit/owned cleanup.

    The controller verifies live process/Guard state.  This function independently
    checks its explicit gate and hashes the cited evidence and ALL producer bytes.
    It does not pretend to inspect remote GPU processes itself.
    """
    bank = Path(bank_dir)
    if (bank / 'READY.json').exists():
        raise RuntimeError('Published bank is immutable')
    gates = ('producer_exited', 'cleanup_passed')
    if any(producer_exit_proof.get(key) is not True for key in gates):
        raise ValueError('Producer exit/owned cleanup must be verified before publication')
    evidence = producer_exit_proof.get('evidence', [])
    if (len(evidence) < 2 or any(not isinstance(item, dict) for item in evidence)
            or len({item.get('path') for item in evidence}) < 2):
        raise ValueError('Missing producer exit evidence')
    for item in evidence:
        path = Path(item['path'])
        if not path.is_absolute() or not path.is_file() or sha_file(path) != item['sha256']:
            raise ValueError('Producer exit evidence changed/missing')
    logical = bank_identity(prompts, identity)
    pieces, reference_keys = [], {}
    for rank in range(8):
        piece = json.loads(_piece_path(bank, f'rank{rank}.json').read_text())
        _validate_piece_metadata(piece, rank, logical)
        path = _piece_path(bank, piece['filename'])
        if path.stat().st_size != piece['bytes'] or sha_file(path) != piece['sha256']:
            raise ValueError('Producer piece content checksum failed')
        for row in piece['requests']:
            index = row['request_index']
            for mode, eagle in (('mtp_key', True), ('plain_key', False)):
                validate_key_record(prompts[index], eagle, row[mode])
                key_id = (index, mode)
                if key_id in reference_keys and row[mode] != reference_keys[key_id]:
                    raise ValueError('Producer native page-key identities differ across ranks')
                reference_keys[key_id] = row[mode]
        pieces.append(piece)
    total_bytes = sum(piece['bytes'] for piece in pieces)
    if total_bytes > MAX_BANK_BYTES:
        raise ValueError('Bank disk payload exceeds frozen 12GiB bound')
    manifest = dict(schema=SCHEMA, state='READY', identity=logical,
                    bank_identity_sha256=_digest(logical), pieces=pieces,
                    total_bytes=total_bytes,
                    publication_gate=producer_exit_proof,
                    published_time=time.time(), one_semantic_prefill_producer=True,
                    second_canonical_copy=False,
                    guard_post_pending=True, archive_qualified=False,
                    qualification_note='Operational import READY only; final run Guard/model-cache/cleanup gates archival qualification')
    manifest['manifest_sha256'] = _digest(manifest)
    _save(bank / 'READY.json', manifest)
    return manifest


def load_manifest(bank_dir, prompts, identity):
    manifest = json.loads(_piece_path(bank_dir, 'READY.json').read_text())
    unsigned = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    logical = bank_identity(prompts, identity)
    if (manifest.get('schema') != SCHEMA or manifest.get('state') != 'READY'
            or manifest.get('manifest_sha256') != _digest(unsigned)
            or manifest.get('identity') != logical or manifest.get('bank_identity_sha256') != _digest(logical)
            or len(manifest.get('pieces', [])) != 8):
        raise ValueError('Bank manifest identity/integrity failed')
    for rank, piece in enumerate(manifest['pieces']):
        _validate_piece_metadata(piece, rank, logical)
        path = _piece_path(bank_dir, piece['filename'])
        if path.stat().st_size != piece['bytes']:
            raise ValueError('Bank rank piece length changed')
    if (manifest.get('total_bytes') != sum(p['bytes'] for p in manifest['pieces'])
            or manifest['total_bytes'] > MAX_BANK_BYTES):
        raise ValueError('Bank total byte count mismatch')
    return manifest


def import_plan(manifest, geom):
    """Every needed logical component has exactly one producer; no head alias."""
    result = []
    requested = geom['component_layers']
    for piece in manifest['pieces']:
        source = piece['geometry']
        if (source['kv_head_start'], source['kv_head_end']) != (geom['kv_head_start'], geom['kv_head_end']):
            continue
        pairs = [(src, requested.index(name)) for src, name in enumerate(source['component_layers']) if name in requested]
        if pairs:
            result.append(dict(piece=piece, layer_pairs=pairs))
    covered = [dest for entry in result for _, dest in entry['layer_pairs']]
    if sorted(covered) != list(range(len(requested))):
        raise ValueError('Missing/overlapping logical target/draft/head coverage')
    return result


def copy_page_layers(destination, source, layer_pairs):
    """Byte-preserving CPU copy. Arrays end in [head_dim, item_byte]."""
    if (destination.ndim != 6 or source.ndim != 6 or destination.shape[:2] != source.shape[:2]
            or destination.shape[3:] != source.shape[3:]):
        raise ValueError('Incompatible page axes')
    for src, dst in layer_pairs:
        destination[:, :, dst, :, :, :] = source[:, :, src, :, :, :]


def import_rank(scheduler, bank_dir, prompts, identity):
    import numpy as np
    import torch
    begun = time.monotonic()
    manifest = load_manifest(bank_dir, prompts, identity)
    cache, host, geom = _runtime(scheduler)
    pending = quiescent(scheduler)
    if _root(cache).children:
        raise RuntimeError('Consumer import requires a fresh empty radix tree')
    plan = import_plan(manifest, geom)
    scratch_bound = max(2 * page_bytes(geom),
                        2 * max(page_bytes(p['piece']['geometry']) for p in plan))
    if scratch_bound > MAX_SCRATCH_BYTES:
        raise RuntimeError('Bounded host copy/readback scratch exceeds frozen 16MiB')
    keys = [native_key_metadata(tokens, _cache_mode(cache)) for tokens in prompts]
    for entry in plan:
        for row in entry['piece']['requests']:
            mode = 'mtp_key' if _cache_mode(cache) else 'plain_key'
            if row[mode] != keys[row['request_index']]:
                raise RuntimeError('Consumer native plain/bigram keys differ from producer')
    source_maps = [(entry, np.memmap(_piece_path(bank_dir, entry['piece']['filename']),
                                  dtype=np.uint8, mode='r',
                                  shape=(REQUESTS, CACHED // PAGE) + page_shape(entry['piece']['geometry'])))
                   for entry in plan]
    source_bytes, copied_bytes, readback_bytes, receipts = 0, 0, 0, []
    try:
        for index in geom['request_indices']:
            indices = cache.host_pool_group.alloc(CACHED)
            if indices is None:
                raise RuntimeError('Existing native CPU host pool cannot fit bank')
            insert_attempted = False
            try:
                slots = indices.tolist()
                for offset in range(0, CACHED, PAGE):
                    first = slots[offset]
                    if first % PAGE or slots[offset:offset + PAGE] != list(range(first, first + PAGE)):
                        raise RuntimeError('New native host slots are not page-contiguous')
                digests = [hashlib.sha256() for _ in source_maps]
                expected_digest, readback_digest = hashlib.sha256(), hashlib.sha256()
                page_checksums = []
                for page in range(CACHED // PAGE):
                    destination = host.get_data_page(slots[page * PAGE], flat=False)
                    if str(destination.device) != 'cpu':
                        raise RuntimeError('Consumer destination unexpectedly on GPU')
                    destination_raw = destination.view(torch.uint8).numpy()
                    destination_bytes = destination_raw.reshape(page_shape(geom))
                    if not np.shares_memory(destination_raw, destination_bytes):
                        raise RuntimeError('Native host view reshape would make a detached copy')
                    expected_page = np.empty(page_shape(geom), dtype=np.uint8)
                    for source_index, (entry, mapping) in enumerate(source_maps):
                        source = mapping[index, page]
                        # Hash exactly the same contiguous local producer page that
                        # is copied; no second file pass or giant temporary buffer.
                        digests[source_index].update(memoryview(source))
                        source_bytes += source.nbytes
                        copy_page_layers(expected_page, source, entry['layer_pairs'])
                    destination_bytes[...] = expected_page
                    expected_hash = hashlib.sha256(memoryview(expected_page)).hexdigest()
                    readback = _host_page_array(host, slots[page * PAGE])
                    if hashlib.sha256(memoryview(readback)).hexdigest() != expected_hash:
                        raise RuntimeError('Native host page readback checksum failed')
                    expected_digest.update(memoryview(expected_page))
                    readback_digest.update(memoryview(readback))
                    page_checksums.append(expected_hash)
                    readback_bytes += readback.nbytes
                    copied_bytes += destination_bytes.nbytes
                    del readback, expected_page
                for (entry, _), digest in zip(source_maps, digests):
                    if digest.hexdigest() != entry['piece']['requests'][index]['sha256']:
                        raise RuntimeError('Imported semantic request content checksum failed')
                key = _native_key(prompts[index], _cache_mode(cache))
                # Once invoked, TreeCore may own the slots even if action or
                # postcondition validation fails. Never free a linked host page;
                # a failed attempt is reclaimed by the supervised cache reset.
                insert_attempted = True
                _insert_host(cache, key, indices, keys[index]['page_hashes'])
                match = _match(cache, prompts[index])
                if len(match.device_indices) or match.host_hit_length != CACHED:
                    raise RuntimeError('Imported native prefix is not CPU4032/GPU0')
                receipts.append(dict(request_index=index, cpu_tokens=match.host_hit_length,
                                     gpu_tokens=len(match.device_indices),
                                     native_key=keys[index], component_layers=geom['component_layers'],
                                     cpu_host_sha256=readback_digest.hexdigest(),
                                     source_view_sha256=expected_digest.hexdigest(),
                                     page_sha256=page_checksums, page_readback_verified=True))
            finally:
                if not insert_attempted:
                    cache.host_pool_group.free(indices)
    finally:
        for _, mapping in source_maps:
            mapping._mmap.close()
    return dict(schema=SCHEMA, passed=True, bank_identity_sha256=manifest['bank_identity_sha256'],
                manifest_sha256=manifest['manifest_sha256'], geometry=geom,
                imported_requests=receipts, cpu_source_bytes_read=source_bytes,
                cpu_host_bytes_written=copied_bytes, direct_native_host_import=True,
                cpu_host_readback_bytes=readback_bytes, page_readback_verified=True,
                semantic_kv=True, synthetic_kv=False, gpu_tensor_reads=0,
                new_native_host_slots=True, serialized_allocator_state=False,
                second_canonical_copy=False, native_hicache_quiescent=True,
                bounded_scratch_bytes=scratch_bound,
                pending_counts=pending, elapsed_s=time.monotonic() - begun,
                per_request_s=(time.monotonic() - begun) / len(receipts))
