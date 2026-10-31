#!/usr/bin/env python3
"""Dashboard lifecycle gate. Does not substitute for engine or placement preflight."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import urlopen

CONTROL = Path(__file__).resolve().parents[3]
TEMPLATES = CONTROL/'dashboards/templates'
REQUIRED_CHECKS = ('environment', 'native_artifacts', 'config_baseline', 'booked_nodes',
                   'node_exclusivity', 'source_tag')


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_for(path):
    path = Path(path).resolve(); spec = read(path)
    assert spec.get('schema_version') == 1, 'Unsupported experiment spec schema'
    assert re.fullmatch(r'[A-Za-z0-9_.-]+', spec['id']), 'Unsafe experiment ID'
    assert spec.get('tasks'), 'Explicit tasks required'
    ids = [t['id'] for t in spec['tasks']]
    assert len(ids) == len(set(ids)), 'Duplicate task IDs'
    assert 1024 <= spec['port'] <= 65535, 'Use an unprivileged port'
    for task in spec['tasks']:
        assert Path(task['root']).is_absolute(), 'Task roots must be absolute'
        assert task['expected'] > 0, 'Expected valid point count required'
    return spec


def matching(spec, library=TEMPLATES):
    candidates = []
    for path in sorted(library.glob('*/template.json')):
        item = read(path)
        if item.get('schema_version') != 1 or item['protocol'] != spec['protocol']:
            continue
        if spec['engine'] not in item['engines'] and '*' not in item['engines']:
            continue
        required = set(spec.get('features', [])); supported = set(item['features'])
        if not required <= supported:
            continue
        candidates.append((len(supported-required), -item['version'], item['id'], path.parent))
    assert candidates, 'No compatible dashboard template. Add and validate a tracked template before launch.'
    if spec.get('template'):
        candidates = [x for x in candidates if x[2] == spec['template']]
        assert candidates, 'Pinned template does not support this experiment'
    return sorted(candidates)[0][3]


def template_hash(template):
    files = sorted(p for p in template.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(template)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def receipt_dir(spec):
    # This is runtime state, never the template repository.
    return Path(spec.get('runtime_dir', Path(spec['tasks'][0]['root'])/'dashboard_guard'))


def prepare(path):
    spec = spec_for(path); template = matching(spec)
    row = dict(schema_version=1, experiment_id=spec['id'], timestamp=time.time(),
               spec_sha256=digest(path), template=template.name,
               template_sha256=template_hash(template), template_path=str(template),
               url=f'http://127.0.0.1:{spec["port"]}', read_only=True)
    out = receipt_dir(spec); existing = out/'selection.json'
    if existing.exists():
        old = read(existing)
        assert old['spec_sha256'] == row['spec_sha256'], 'Existing experiment specification changed; use a new ID/runtime directory'
        assert old['template_sha256'] == row['template_sha256'], 'Existing template changed; pin version or register a new dashboard revision'
        return old
    atomic(existing, row)
    return row


def http_json(url):
    with urlopen(url, timeout=8) as response:
        assert response.status == 200, f'Dashboard HTTP {response.status}'
        return json.load(response)


def dashboard_check(path):
    spec = spec_for(path); row = prepare(path)
    health = http_json(row['url']+'/api/health')
    assert health.get('ok') and health.get('read_only'), 'Dashboard health failure'
    if spec.get('attach_existing'):
        expected = hashlib.sha256(str(Path(spec['tasks'][0]['root']).resolve()).encode()).hexdigest()
        assert health.get('root_sha256') == expected, 'Port belongs to a different experiment'
        assert health.get('adapter') == spec['protocol'], 'Dashboard adapter mismatch'
    else:
        assert health.get('experiment_id') == spec['id'], 'Port belongs to another experiment'
        assert health.get('spec_sha256') == row['spec_sha256'], 'Dashboard specification mismatch'
        assert health.get('template_sha256') == row['template_sha256'], 'Dashboard template mismatch'
    status = http_json(row['url']+'/api/status')
    assert time.time()-status['timestamp'] < 20, 'Stale dashboard status'
    assert {t['id'] for t in status['tasks']} == {t['id'] for t in spec['tasks']}, 'Dashboard omits experiment tasks'
    running = [t for t in status['tasks'] if t.get('live')]
    node_sets = [set(t.get('nodes', next(s.get('nodes',[]) for s in spec['tasks'] if s['id']==t['id']))) for t in running]
    assert all(not a & b for i,a in enumerate(node_sets) for b in node_sets[i+1:]), 'Active tasks share a GPU node'
    assert len(set().union(*node_sets)) <= spec.get('max_active_nodes', 100000), 'Active GPU node limit exceeded'
    return spec, row, status


def engine_check(spec):
    assert not spec.get('attach_existing'), 'Observational attachment is not launch authorization'
    path = Path(spec['engine_preflight']); evidence = read(path)
    assert evidence.get('passed') is True, 'Engine guard did not pass'
    assert 0 <= time.time()-evidence['timestamp'] <= spec.get('preflight_max_age_s',3600), 'Engine guard evidence expired'
    assert evidence['experiment_id'] == spec['id'] and evidence['engine'] == spec['engine'], 'Wrong guard evidence'
    expected_nodes = sorted(set(n for t in spec['tasks'] for n in t.get('nodes',[])))
    assert sorted(evidence['nodes']) == expected_nodes, 'Guard placement changed'
    for check in REQUIRED_CHECKS:
        assert evidence['checks'].get(check) is True, f'Engine guard missing: {check}'
    if 'kv-checkpoints' in spec.get('features',[]):
        assert evidence['checks'].get('checkpoint') is True, 'Live compatible checkpoint proof required'
    inputs = spec.get('frozen_inputs', [])
    assert inputs, 'Frozen source/config/workload references required'
    for item in inputs:
        assert digest(item['path']) == item['sha256'], f'Frozen input changed: {Path(item["path"]).name}'
    assert evidence.get('frozen_inputs') == inputs, 'Guard does not bind the frozen input fingerprints'
    sources = spec.get('sources', [])
    assert sources and evidence.get('sources') == sources, 'Pinned source identities required'
    for source in sources:
        def git(*args):
            return subprocess.check_output(['git','-C',source['worktree'],*args],text=True).strip()
        assert git('rev-parse','HEAD') == source['commit'], 'Source commit moved'
        assert git('rev-parse',source['tag']+'^{commit}') == source['commit'], 'Source tag mismatch'
        assert git('cat-file','-t',source['tag']) == 'tag', 'Annotated immutable tag required'
        assert not git('status','--porcelain','--untracked-files=no'), 'Tracked source changes after preflight'
    return dict(path=str(path), sha256=digest(path))


def check(path, phase='pre', exit_code=None):
    spec,row,status = dashboard_check(path)
    report = dict(timestamp=time.time(), phase=phase, experiment_id=spec['id'],
                  dashboard=row, state=status.get('state'), valid=status.get('valid'),
                  expected=status.get('expected'), tasks=[dict(id=t['id'],state=t['state'],live=t['live']) for t in status['tasks']])
    if phase == 'pre':
        report['engine_preflight'] = engine_check(spec)
        report['passed'] = True
    elif phase == 'observe':
        report.update(passed=True, launch_authorization=False)
    else:
        # Completion requires validated results AND verified cleanup, not just process exit.
        report['exit_code'] = exit_code
        report['passed'] = (exit_code == 0 and status.get('state') == 'complete'
                            and status.get('valid') == status.get('expected')
                            and all(t['state']=='complete' and not t['live'] for t in status['tasks'])
                            and status.get('cleanup_verified') is True)
    destination = receipt_dir(spec)/'checks'/f'{time.time_ns()}-{phase}.json'
    atomic(destination, report)
    if phase == 'post':
        assert report['passed'], f'Postcheck failed; evidence preserved at {destination}'
    return report


def load_adapter(template, spec):
    descriptor = read(template/'template.json')
    path = (template/descriptor['adapter']).resolve()
    assert path.parent == template.resolve(), 'Adapter path escapes template'
    sys.path.insert(0,str(template))
    module_spec = importlib.util.spec_from_file_location('experiment_dashboard_adapter', path)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    module.configure(spec)
    return module


def serve(path):
    spec = spec_for(path); selection = prepare(path); template = matching(spec)
    adapter = load_adapter(template,spec)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            from urllib.parse import urlsplit
            route = urlsplit(self.path).path
            try:
                if route == '/api/status':
                    body = json.dumps(adapter.status(),ensure_ascii=False).encode(); kind = 'application/json'
                elif route == '/api/health':
                    adapter.status()
                    body = json.dumps(dict(ok=True,read_only=True,experiment_id=spec['id'],
                        spec_sha256=selection['spec_sha256'],template_sha256=selection['template_sha256'],timestamp=time.time())).encode()
                    kind = 'application/json'
                else:
                    assets = read(template/'template.json')['assets']
                    name = 'index.html' if route == '/' else route.lstrip('/')
                    if name not in assets: return self.send_error(404)
                    body = (template/name).read_bytes()
                    kind = {'.html':'text/html','.js':'text/javascript','.css':'text/css'}[Path(name).suffix]
                self.send_response(200); self.send_header('Content-Type',kind+'; charset=utf-8')
                self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body)))
                self.end_headers(); self.wfile.write(body)
            except Exception as e:
                self.send_error(503,type(e).__name__)
        def do_POST(self): self.send_error(405,'Read-only dashboard')
        def log_message(self,*args): pass
    print(json.dumps(selection),flush=True)
    ThreadingHTTPServer(('127.0.0.1',spec['port']),Handler).serve_forever()


def run(path, command, exit_after_run=False):
    spec = spec_for(path); selection = prepare(path)
    assert not spec.get('attach_existing'), 'Observe-only dashboard cannot launch an experiment'
    assert command, 'Experiment command required after --'
    server = None; child = None; log = None
    old_handlers = {}
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    for signum in (signal.SIGTERM,signal.SIGINT):
        old_handlers[signum] = signal.signal(signum,interrupted)
    try:
        # Refuse a foreign listener; reuse only exactly matching dashboard identity.
        try:
            http_json(selection['url']+'/api/health')
            dashboard_check(path)
        except (URLError,ConnectionError,TimeoutError):
            log=(receipt_dir(spec)/'server.log').open('a')
            server=subprocess.Popen([sys.executable,__file__,'serve','--spec',str(path)],stdout=log,stderr=subprocess.STDOUT)
            for _ in range(60):
                assert server.poll() is None, 'Dashboard failed to start; inspect server.log'
                try: dashboard_check(path);break
                except (URLError,ConnectionError,TimeoutError): time.sleep(.2)
            else: raise RuntimeError('Dashboard start timed out')
        check(path,'pre')
        child=subprocess.Popen(command)
        ret=child.wait()
        check(path,'post',ret)
        print(json.dumps(dict(experiment_exit_code=ret,postcheck='passed',url=selection['url'])),flush=True)
        if server and not exit_after_run:
            print('Results dashboard remains supervised in this foreground session; Ctrl-C closes it.',flush=True)
            server.wait()
        return ret
    finally:
        if child and child.poll() is None:
            child.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):child.wait(timeout=30)
        if server and server.poll() is None:
            server.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):server.wait(timeout=10)
        if log:log.close()
        for signum,handler in old_handlers.items():signal.signal(signum,handler)


def add_template(source, name):
    assert re.fullmatch(r'[a-z0-9][a-z0-9-]+',name), 'Unsafe template ID'
    source=Path(source).resolve(); item=read(source/'template.json')
    assert item['id']==name and item['schema_version']==1, 'Template descriptor mismatch'
    assert not (TEMPLATES/name).exists(), 'Immutable template version exists; use a new version ID'
    paths=list(source.rglob('*'))
    assert not any(p.is_symlink() for p in paths), 'Template symlinks forbidden'
    assert all(p.suffix in ('.py','.js','.css','.html','.json') for p in paths if p.is_file()), 'Only reviewed template source files may enter library'
    for key in ('adapter',):
        assert (source/item[key]).resolve().parent==source, 'Unsafe adapter path'
    assert all((source/a).resolve().parent==source for a in item['assets']), 'Unsafe asset path'
    shutil.copytree(source,TEMPLATES/name)
    return dict(template=name,needs_validation_and_git_commit=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('match','prepare','serve','check','run'):
        p=sub.add_parser(name);p.add_argument('--spec',type=Path,required=True)
        if name=='check':
            p.add_argument('--phase',choices=('pre','post','observe'),default='pre');p.add_argument('--exit-code',type=int)
        if name=='run':
            p.add_argument('--exit-after-run',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER)
    p=sub.add_parser('add-template');p.add_argument('--source',type=Path,required=True);p.add_argument('--id',required=True)
    args=parser.parse_args()
    if args.action=='match': result=dict(template=matching(spec_for(args.spec)).name)
    elif args.action=='prepare': result=prepare(args.spec)
    elif args.action=='check': result=check(args.spec,args.phase,args.exit_code)
    elif args.action=='serve': return serve(args.spec)
    elif args.action=='run': return run(args.spec,args.command[1:] if args.command[:1]==['--'] else args.command,args.exit_after_run)
    else: result=add_template(args.source,args.id)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try: sys.exit(main())
    except (AssertionError,ValueError,KeyError,OSError) as error:
        print(json.dumps(dict(passed=False,error=str(error)),ensure_ascii=False),file=sys.stderr)
        sys.exit(2)
