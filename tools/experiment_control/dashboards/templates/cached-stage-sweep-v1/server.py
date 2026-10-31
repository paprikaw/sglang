"""Passive stage-experiment monitor: local files only, never SSH or inference."""
import argparse
import csv
import io
import json
import re
import time
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from progress_detail import task_details

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PP_ORIGINAL = ROOT.parent/'glm47_cached_stage_scaling_ppmetrics_20261002'
CACHE = {}
SPEC = {}


def configure(spec):
    global ROOT, PP_ORIGINAL, SPEC
    SPEC=spec
    ROOT=Path(spec['tasks'][0]['root'])
    PP_ORIGINAL=Path(spec.get('original_pp_root',ROOT.parent/'glm47_cached_stage_scaling_ppmetrics_20261002'))


def read_json(path, default=None):
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if path in CACHE and CACHE[path][0] == stamp:
            return CACHE[path][1]
        value = json.loads(path.read_text())
        CACHE[path] = (stamp, value)
        return value
    except (OSError, ValueError):
        return default


def tail(path, size=16000):
    try:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size-size))
            return handle.read().decode(errors='replace')
    except OSError:
        return ''


def controller_live(state, root):
    try:
        process = Path('/proc',str(state['controller_pid']))
        args = (process/'cmdline').read_bytes().split(bytes([0]))
        cwd = (process/'cwd').resolve()
        for arg in args:
            if b'controller_' not in arg or not arg.endswith(b'.py'):
                continue
            path = Path(arg.decode())
            path = path if path.is_absolute() else cwd/path
            if path.resolve().parent == root.resolve():
                return True
        return False
    except (OSError,KeyError):
        return False


def valid_cells(arm, roots):
    cells, invalid, seen = [], [], set()
    for base in dict.fromkeys(path.resolve() for path in roots):
        for path in sorted((base/'measurements'/arm).glob('measure*/*/trial*/result.json')):
            result = read_json(path,{})
            if result.get('kind') not in ('decode','prefill'):
                continue
            if not result.get('passed') or not result.get('checks') or not all(result['checks'].values()):
                invalid.append(str(path))
                continue
            if result['case'] in seen:
                raise ValueError('Duplicate accepted cell; explicit provenance required')
            seen.add(result['case'])
            score = result['score']
            row = dict(arm=arm,case=result['case'],kind=result['kind'],context=result['context_tokens'],
                       append=result.get('append_tokens'),output=result.get('output_tokens'),
                       rate=result.get('request_rate'),concurrency=result.get('concurrency'),
                       requests=score.get('cohort_requests',result.get('concurrency')),
                       completed=score.get('cohort_completed',result.get('concurrency')),
                       censored=score.get('unfinished_requests',0),path=str(path))
            if result['kind']=='decode':
                row.update(mean_s=score['tpot_request_mean_s']['mean'],p99_s=score['tpot_request_mean_s']['p99'],
                           throughput=score['output_tokens_per_s'],metric='TPOT',
                           remaining_TTOT_mean_s=score['decode_remaining_s']['mean'],
                           remaining_TTOT_p99_s=score['decode_remaining_s']['p99'],
                           TTFT_diagnostic_mean_s=result['diagnostic_latencies']['ttft_preparation_s']['mean'],
                           TTFT_diagnostic_p99_s=result['diagnostic_latencies']['ttft_preparation_s']['p99'],
                           E2E_diagnostic_mean_s=result['diagnostic_latencies']['e2e_including_preparation_s']['mean'],
                           E2E_diagnostic_p99_s=result['diagnostic_latencies']['e2e_including_preparation_s']['p99'])
            else:
                row.update(mean_s=score['completion_latency_s']['mean'],p99_s=score['completion_latency_s']['p99'],
                           throughput=score['computed_prefill_tokens_per_s'],metric='completion',
                           completed_requests_per_s=score['completed_requests_per_s'])
            cells.append(row)
    return cells,invalid


def status():
    plan = read_json(ROOT/'two_node_plan.json',{})
    pp_next = Path(plan.get('pp_root',str(ROOT.parent/'glm47_cached_stage_scaling_pp_twonn_20261002')))
    specs = [(t['id'],t['label'],Path(t['root']),t['attempt'],t['expected']) for t in SPEC['tasks']]
    cells,cards,nodes,errors = [],[],[],[]
    for arm,label,base,attempt,expected in specs:
        task=next(t for t in SPEC['tasks'] if t['id']==arm)
        accepted,bad = valid_cells(arm,[Path(r) for r in task.get('accepted_roots',[str(base)])])
        cells.extend(accepted);errors.extend(bad)
        ctrl = read_json(base/'controllers'/f'{attempt}_{arm}'/'state.json',{})
        live = controller_live(ctrl,base)
        worker_stops=[read_json(p.with_name('stopped.json'),{}) for p in (base/'runs'/arm/attempt).glob('*/*/process.json')]
        idle=[read_json(Path(p),{}) for p in task.get('cleanup_evidence_files',[])]
        cleanup=(bool(worker_stops) and all('returncode' in r for r in worker_stops)
                 and len(idle)==len(task.get('nodes',[])) and bool(idle)
                 and all(r.get('GPU_idle') is True for r in idle))
        complete = len(accepted)==expected and ctrl.get('stage')=='measurement_complete' and not live and cleanup
        mode = 'failed' if ctrl.get('stage')=='failed' else 'complete' if complete else 'running' if live else 'queued'
        progress = read_json(base/'measurements'/arm/attempt/'progress.json',{})
        underway = []
        for started in (base/'measurements'/arm/attempt).glob('*/trial*/started.json'):
            if not started.with_name('result.json').exists():
                underway.append(read_json(started,{}))
        current = max(underway,key=lambda x:x.get('timestamp',0),default={}) if live else {}
        cards.append(dict(arm=arm,label=label,valid=len(accepted),expected=expected,state=mode,
                          stage=ctrl.get('stage'),live=live,attempt=attempt,current=current.get('case'),
                          last_case=progress.get('case'),controller_pid=ctrl.get('controller_pid'),
                          accepted_kinds={kind:sum(c['kind']==kind for c in accepted) for kind in ('decode','prefill')}))
        if arm=='dp2_tp4' or live:
            for path in sorted((base/'runs'/arm/attempt).glob('*/*/process.json')):
                stopped = read_json(path.with_name('stopped.json'))
                sample = None
                for line in reversed(tail(path.with_name('node_metrics.jsonl'),4000).splitlines()):
                    try: sample=json.loads(line);break
                    except ValueError: continue
                log=tail(path.with_name('server.log'))
                percentages=re.findall(r'Loading safetensors checkpoint shards:\s*(\d+)%',log)
                util=[];memory=[];power=[]
                if sample:
                    for line in sample.get('gpu_csv','').splitlines():
                        try:
                            fields=[float(x.strip()) for x in line.split(',')]
                            memory.append(fields[1]);util.append(fields[2]);power.append(fields[4])
                        except (ValueError,IndexError): pass
                nodes.append(dict(node=path.parent.name,arm=arm,attempt=attempt,
                    state='stopped' if stopped else 'running' if live else 'unknown',
                    loading_percent=int(percentages[-1]) if percentages else None,
                    utilization=sum(util)/len(util) if util else None,
                    memory_peak_mib=max(memory) if memory else None,power_w=sum(power) if power else None,
                    sample_age_s=time.time()-sample['timestamp'] if sample else None))
    gate_path=ROOT/'stages/dp2_tp4/measure1/capacity/result.json'
    gate=read_json(gate_path,{})
    frozen=read_json(ROOT/'frozen_rates.json',{})
    queue=read_json(ROOT/'controllers/two_node_serial1/state.json',{})
    running_arms={card['arm'] for card in cards if card['live']}
    if running_arms:
        nodes=[node for node in nodes if node['arm'] in running_arms]
    failed=any(card['state']=='failed' for card in cards) or queue.get('stage')=='failed'
    complete=all(card['state']=='complete' for card in cards)
    if SPEC.get('require_serial_completion'): complete=complete and queue.get('stage')=='two_serial_arms_complete'
    tasks=task_details(cards,specs,read_json)
    for task in tasks: task['nodes']=next(t.get('nodes',[]) for t in SPEC['tasks'] if t['id']==task['id'])
    return dict(timestamp=time.time(),valid=len(cells),expected=sum(t['expected'] for t in SPEC['tasks']),cards=cards,nodes=nodes,
        tasks=tasks, concurrent_tasks=sum(c['live'] for c in cards),cleanup_verified=complete,
        cells=cells,invalid_count=len(errors),retained_nodes=plan.get('retained_nodes',[]),
        gpu_node_limit=2,gpu_count=8,queue=queue,
        state='failed' if failed else 'complete' if complete else 'running' if any(c['live'] for c in cards) else 'waiting',
        valid_requests=sum(c.get('completed') or 0 for c in cells),
        capacity_gate_passed=bool(gate.get('passed') and gate.get('capacity_gate_passed')),
        frozen_rates=frozen.get('rates'),pp_bank_prefixes=32,pp_bank_tokens=1966080,
        source_DP='041ce35b5e',source_PP='97ab987e8a',
        caveat='P 固定 context 长度、追加长度和速率；DP 基底为 session0/1，PP为session0，不是完全相同的输入。')


class Handler(BaseHTTPRequestHandler):
    def respond(self,body,content_type):
        self.send_response(200);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.end_headers()
        if self.command!='HEAD': self.wfile.write(body)

    def do_GET(self):
        route=urlsplit(self.path).path
        try:
            if route=='/api/status':
                return self.respond(json.dumps(status(),ensure_ascii=False).encode(),'application/json; charset=utf-8')
            if route=='/api/health':
                status()
                return self.respond(json.dumps(dict(ok=True,read_only=True,adapter='cached-stage-sweep',
                    root_sha256=hashlib.sha256(str(ROOT.resolve()).encode()).hexdigest(),timestamp=time.time())).encode(),'application/json')
            if route=='/api/metrics.csv':
                rows=status()['cells'];stream=io.StringIO()
                fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['arm','case']
                writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
                return self.respond(stream.getvalue().encode('utf-8-sig'),'text/csv; charset=utf-8')
            allowed={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if route not in allowed: return self.send_error(404)
            kind='text/html' if route in ('/','/index.html') else 'text/javascript' if route.endswith('.js') else 'text/css'
            self.respond((HERE/allowed[route]).read_bytes(),kind+'; charset=utf-8')
        except Exception as error:
            self.send_error(500,type(error).__name__)

    do_HEAD=do_GET

    def do_POST(self):
        self.send_error(405,'Read-only monitor')

    def log_message(self,*_):
        pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18471)
    args=parser.parse_args();print(json.dumps(dict(url=f'http://127.0.0.1:{args.port}',read_only=True)),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
