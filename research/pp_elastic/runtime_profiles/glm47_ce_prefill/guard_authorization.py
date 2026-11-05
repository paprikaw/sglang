"""Read-only authentication of the supervisor-owned GPU lease."""
import hashlib,hmac,json,os,socket,stat,subprocess,sys,time,signal
from pathlib import Path
from common import ROOT, GUARD, save
sys.path.insert(0,str(Path(GUARD).resolve().parent))
from artifact_visibility import ArtifactVisibilityTimeout,wait_for_stable_json
GUARD_KEYS=('EXPERIMENT_GUARD_LEASE_RECEIPT','EXPERIMENT_GUARD_ALLOCATION_FILE','EXPERIMENT_GUARD_ACQUIRED_NODES')

def process_ticks(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        return int(raw[raw.rfind(')') + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None

def visible_json(path, receipt_dir, timeout=15):
    path = Path(path)
    key = hashlib.sha256(str(path.absolute()).encode()).hexdigest()[:16]
    destination = Path(receipt_dir) / f'{key}.json'
    try:
        value, receipt = wait_for_stable_json(path, timeout_s=timeout)
    except ArtifactVisibilityTimeout as error:
        save(destination, error.receipt)
        raise
    save(destination, receipt)
    return value

def validate_guard(run, cfg, env, receipt_dir, require_parent=False):
    """Authenticate an existing lease using a read-only registry snapshot."""
    assert all(env.get(key) for key in GUARD_KEYS), 'Guard supervisor required'
    path = Path(env[GUARD_KEYS[0]])
    mode = path.stat()
    assert mode.st_uid == os.getuid() and stat.S_IMODE(mode.st_mode) == 0o600
    receipt = visible_json(path, receipt_dir)
    allocation = visible_json(env[GUARD_KEYS[1]], receipt_dir)
    registry = visible_json(Path(receipt['registry_dir']) / 'registry.json', receipt_dir)
    lease = next((item for item in registry['leases'] if item['lease_id'] == receipt['lease_id']), None)
    assert lease and lease['state'] == 'active', 'Guard lease is absent or inactive'
    assert hmac.compare_digest(hashlib.sha256(receipt['token'].encode()).hexdigest(), lease['token_sha256'])
    assert time.time() - lease['heartbeat_at'] <= lease['ttl_s'], 'Guard heartbeat expired'
    assert receipt['experiment_id'] == lease['experiment_id'] == allocation['experiment_id'] == cfg['id']
    assert receipt['lease_id'] == allocation['lease_id']
    # Bind this authorization to the Guard specification, without repeating
    # source, frozen-input, import, GPU-idle or config-baseline checks.
    digest = hashlib.sha256((Path(run) / 'dashboard_spec.json').read_bytes()).hexdigest()
    assert lease['spec_sha256'] == receipt['spec_sha256'] == digest
    nodes = sorted(cfg['nodes'])
    assert len(nodes) == len(set(nodes)) == 2
    assert set(allocation['tasks']) == set(lease['allocation']) == set(cfg['groups'])
    for task_id, group in cfg['groups'].items():
        assert sorted(allocation['tasks'][task_id]) == sorted(lease['allocation'][task_id]) == sorted(group['nodes'])
    assert sorted(allocation['nodes']) == sorted(lease['nodes']) == nodes
    assert sorted(env[GUARD_KEYS[2]].split(',')) == nodes
    if require_parent:
        assert lease['owner_host'] == socket.gethostname().split('.')[0]
        assert lease['owner_pid'] == os.getppid(), 'Controller must be a direct Guard child'
        assert process_ticks(os.getppid()) == lease['owner_start_ticks']
    return receipt

def authorization_signature(payload, token):
    return hmac.new(token.encode(), json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), hashlib.sha256).hexdigest()

def close_remote_workers(procs):
    # Closing SSH stdin tells the remote supervisor to stop its process tree.
    for proc in procs:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    # Full-model workers need up to 11s owned-tree shutdown plus 20s
    # post-exit GPU evidence. Keep their SSH control channel alive to save it.
    deadline = time.monotonic() + 45
    for proc in procs:
        try:
            proc.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
    for proc in procs:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()
