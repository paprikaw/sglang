#!/usr/bin/env python3
"""Read-only canonical-bank discovery, independent of checkpoint names.

Only metadata and stat identities are read. A match is advisory; the ordinary
Guard admission must still prove the exact complete live bank before loading.
Loader qualification describes a model layout and capacity, never a model path.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def layout_sha(config):
    # Checkpoint provenance can change without changing the serving layout.
    # Keep all architecture, dtype, quantization and tensor-shape settings.
    ignored = {'_name_or_path', 'name_or_path', '_commit_hash', 'transformers_version'}
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items() if k not in ignored}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    return json_sha(normalize(config))


def record_hints(spec, runtime_root):
    """Guard hook: bounded path hints, never cache readiness or admission proof.

    Preserve paths from frozen node specs so a later checkpoint outside the
    configured HF catalog can also be found. Each node has just one latest file.
    """
    contract = spec.get('model_cache', {})
    if contract.get('policy') != 'required' or contract.get('provider') != 'sglang_canonical_full_weight_v1':
        return 0
    count = 0
    root = Path(runtime_root) / 'model_hints'
    for node, descriptor in contract.get('node_specs', {}).items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', node):
            continue
        try:
            path = Path(descriptor['path'])
            st = path.lstat()
            if not path.is_absolute() or path.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o022:
                continue
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != descriptor.get('sha256'):
                continue
            ns = json.loads(raw)
            if (ns.get('node') != node or ns.get('cache_scheme') != 'canonical_full_weight'
                    or not Path(ns['model_path']).is_absolute()):
                continue
            hint = {k: ns[k] for k in ('node', 'expected_job_id', 'source_commit', 'model_path')}
            hint.update(timestamp=time.time(), node_spec=descriptor,
                        authority='path_hint_requires_live_full_bank_match')
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, name = tempfile.mkstemp(prefix='.' + node + '.', dir=root)
            try:
                with os.fdopen(fd, 'w') as handle:
                    json.dump(hint, handle)
                os.replace(name, root / (node + '.json'))
            finally:
                Path(name).unlink(missing_ok=True)
            count += 1
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return count


def model_hint(node, bank, runtime_root):
    if not runtime_root:
        return None
    path = Path(runtime_root) / 'model_hints' / (node + '.json')
    try:
        st = path.lstat()
        if path.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o022 or st.st_size > 32768:
            return None
        hint = json.loads(path.read_text())
        if (hint.get('node') == node and hint.get('source_commit') == bank.get('source_commit')
                and str(hint.get('expected_job_id')) == str(bank.get('expected_job_id'))
                and Path(hint['model_path']).is_absolute()):
            return hint['model_path']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def checkpoint_paths(roots, *, max_depth=4, max_directories=1024):
    """Bounded directory discovery, including ordinary HF snapshot layouts."""
    pending = [(Path(root), 0) for root in roots]
    seen = set()
    visited = 0
    while pending:
        path, depth = pending.pop(0)
        try:
            resolved = path.resolve(strict=True)
            if resolved in seen or not resolved.is_dir():
                continue
            seen.add(resolved)
            visited += 1
            if visited > max_directories:
                raise ValueError('Model discovery directory bound exceeded')
            if (resolved / 'config.json').is_file():
                yield resolved
                continue
            if depth < max_depth:
                with os.scandir(resolved) as entries:
                    children = sorted((Path(e.path) for e in entries
                                       if e.is_dir() and e.name not in
                                       ('.git', 'blobs', '.locks', '__pycache__')))
                if len(children) + visited + len(pending) > max_directories:
                    raise ValueError('Model discovery directory bound exceeded')
                pending.extend((child, depth + 1) for child in children)
        except (FileNotFoundError, PermissionError):
            continue


def verified_status(bank, node, client_class):
    """Authenticate the daemon without claiming a GPU or changing its cache."""
    if socket.gethostname().split('.')[0] != node or bank.get('uid') != os.getuid():
        raise ValueError('Cache node/user identity differs')
    manifest = bank['manifest']
    if manifest.get('node') != node or manifest.get('cache_id') != bank['daemon_id']:
        raise ValueError('Cache manifest identity differs')
    pid = manifest['pid']
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    if int(stat[19]) != manifest['process_start_ticks'] or stat[0] == 'Z':
        raise ValueError('Cache process identity changed')
    cgroup = Path(f'/proc/{pid}/cgroup').read_text()
    if not re.search(r'(?:^|/)job[_/]' + re.escape(str(bank['expected_job_id'])) + r'(?:/|$)', cgroup, re.M):
        raise ValueError('Cache daemon belongs to another Slurm job')
    with client_class(bank['socket_path'], timeout=3) as client:
        status = client.status()
    expected = dict(uid=os.getuid(), pid=pid, start_ticks=manifest['process_start_ticks'],
                    source_commit=bank['source_commit'], max_bytes=bank['max_bytes'],
                    headroom_bytes=bank['headroom_bytes'])
    if (status.get('ok') is not True or 'canonical_groups_v1' not in status.get('features', [])
            or any(status.get(k) != v for k, v in expected.items())):
        raise ValueError('Live cache daemon status differs')
    entries = status.get('entries', [])
    if (not entries or any(e.get('state') != 'ready' or e.get('evict_pending')
                           or type(e.get('size')) is not int or e['size'] <= 0 for e in entries)
            or len({e['key'] for e in entries}) != len(entries)
            or status.get('reserved_bytes') != 0):
        raise ValueError('Live canonical bank is incomplete or changing')
    return status


def match_models(paths, status, describe_model):
    ready = {e['key']: e['size'] for e in status['entries']}
    models, failures = [], 0
    for path in paths:
        try:
            desc = describe_model(str(path))
            keys = [item['key'] for item in desc['items']]
            if not keys or len(keys) != len(desc['shards']):
                continue
            if not all(ready.get(key) == shard['size'] for key, shard in zip(keys, desc['shards'])):
                continue
            config_path = Path(desc['model_root']) / 'config.json'
            raw = config_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != desc['fingerprint']['metadata_sha256']['config.json']:
                raise ValueError('Model config changed during discovery')
            models.append(dict(model_root=desc['model_root'], group_key=desc['group_key'],
                metadata_sha256=desc['fingerprint']['metadata_sha256'],
                total_bytes=desc['total_bytes'], shard_count=desc['shard_count'],
                keys=keys, layout_sha256=layout_sha(json.loads(raw))))
        except (OSError, ValueError, KeyError, RuntimeError):
            # One incomplete download or unsupported checkpoint must not hide
            # a different complete model that really is cached on this node.
            failures += 1
    return models, failures


def probe(payload):
    bank = payload['bank']
    cache_module = Path(payload['engine_root']) / 'python/sglang/srt/weight_cache'
    sys.path.insert(0, str(cache_module))
    spec = importlib.util.spec_from_file_location('backfill_canonical', cache_module / 'cpu_canonical.py')
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    from cpu_protocol import CpuWeightCacheClient
    status = verified_status(bank, payload['node'], CpuWeightCacheClient)
    settings = payload['discovery']
    roots = payload.get('extra_paths', []) + settings['roots']
    paths = list(checkpoint_paths(roots, max_depth=settings.get('max_depth', 4),
                                 max_directories=settings.get('max_directories', 1024)))
    if len(paths) > settings.get('max_models', 128):
        raise ValueError('Model discovery candidate bound exceeded')
    models, failures = match_models(paths, status, canonical.describe_model)
    preferred = next((m['model_root'] for m in models
                      if m['model_root'] in payload.get('extra_paths', [])), None)
    return dict(timestamp=time.time(), daemon_id=bank['daemon_id'], preferred_model_root=preferred,
                expected_job_id=bank['expected_job_id'], source_commit=bank['source_commit'],
                status=status, models=models, inspected_models=len(paths), metadata_failures=failures)


def discover(node, bank, recipes, settings, runtime_root=None):
    script = Path(__file__).resolve()
    if hashlib.sha256(script.read_bytes()).hexdigest() != settings.get('script_sha256'):
        raise ValueError('Model discovery source changed without qualification')
    compatible = [r for r in recipes if r.get('engine_commit') == bank.get('source_commit')]
    if not compatible:
        return dict(models=[], reason='waiting_for_compatible_recipe')
    hint = model_hint(node, bank, runtime_root)
    payload = dict(node=node, bank=bank, discovery=settings, extra_paths=[hint] if hint else [],
                   engine_root=compatible[0]['node_template']['engine_root'])
    command = shlex.join(['/usr/bin/python3', '-B', str(script), 'probe'])
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4', node, command],
        input=json.dumps(payload) + '\n', text=True, capture_output=True, timeout=45, check=True)
    found = json.loads(result.stdout)
    if any(found.get(k) != bank.get(k) for k in ('daemon_id', 'expected_job_id', 'source_commit')):
        raise ValueError('Discovered model belongs to another cache identity')
    return found


def current_recipes(node, bank, observation, templates):
    """Bind a qualified layout to the discovered current checkpoint/revision."""
    result = []
    models = observation.get('models', [])
    preferred = [m for m in models if m['model_root'] == observation.get('preferred_model_root')]
    # A complete live match for the latest Guard model fixes the model choice.
    # Missing loader qualification must not quietly choose an older cached bank.
    for model in preferred or models:
        for template in templates:
            rule = template.get('checkpoint_qualification', {})
            if (template.get('engine_commit') != bank.get('source_commit')
                    or model.get('layout_sha256') != rule.get('layout_sha256')
                    or not 0 < model.get('total_bytes', 0) <= rule.get('max_bank_bytes', 0)):
                continue
            recipe = copy.deepcopy(template)
            recipe['template_id'] = template['id']
            recipe['id'] = 'cache-' + template['id'] + '-' + json_sha(
                [model, bank['daemon_id'], bank['expected_job_id'], bank['source_commit']])[:16]
            recipe['fallback'] = False
            recipe['node_template']['model_path'] = model['model_root']
            recipe['node_cache_keys'] = {node: model['keys']}
            recipe['discovered_model'] = dict(model, daemon_id=bank['daemon_id'],
                expected_job_id=bank['expected_job_id'], source_commit=bank['source_commit'])
            recipe['node_runner'] = rule['node_runner']
            result.append(recipe)
    # Prefer the current Guard path hint only after a live full-bank match.
    # Otherwise use an explicit stable policy; directory timestamps are not
    # evidence of model recency.
    result.sort(key=lambda r: (r['discovered_model']['model_root'] != observation.get('preferred_model_root'),
                              -r['discovered_model']['total_bytes'],
                              r['discovered_model']['model_root'], r['template_id']))
    reason = ('reuse_current_model' if result else observation.get('reason') or
              ('waiting_for_model_loader' if observation.get('models') else 'waiting_for_model_identity'))
    return result, reason


if __name__ == '__main__':
    if sys.argv[1:] != ['probe']:
        raise SystemExit('Expected probe; JSON payload on stdin')
    print(json.dumps(probe(json.loads(sys.stdin.readline()))))
