"""Explicit progress schema; no missing values represented as zero successes."""
import json
import os
from pathlib import Path
import time

SPEC={}
def configure(spec):
    global SPEC
    SPEC=spec

def read(path):
    try:
        assert path.stat().st_size <= 1_000_000, 'Progress file too large'
        return json.loads(path.read_text())
    except (OSError,ValueError): return {}

def process_matches(row,task):
    # PID reuse is not liveness evidence. Require start ticks and exact script path.
    try:
        process=Path('/proc',str(row['controller_pid']))
        text=(process/'stat').read_text(); ticks=int(text[text.rfind(')')+2:].split()[19])
        assert ticks==row['process_start_ticks']
        script=Path(task['controller_script']).resolve();cwd=(process/'cwd').resolve()
        args=(process/'cmdline').read_bytes().split(b'\0')
        return any((Path(arg.decode()) if arg.startswith(b'/') else cwd/arg.decode()).resolve()==script for arg in args if arg.endswith(b'.py'))
    except (OSError,KeyError,ValueError,AssertionError): return False

def status():
    tasks=[]
    for task in SPEC['tasks']:
        root=Path(task['root']); row=read(root/task.get('progress_file','progress.json'))
        live=process_matches(row,task)
        valid=row.get('valid_points');expected=task['expected']
        assert valid is None or isinstance(valid,int) and 0<=valid<=expected, 'Invalid accepted point count'
        cleanup=read(root/task.get('cleanup_file','cleanup.json'))
        complete=(row.get('stage')=='complete' and valid==expected and not live
                  and cleanup.get('passed') is True and cleanup.get('model_processes_exited') is True
                  and cleanup.get('gpu_idle') is True)
        age=max(0,time.time()-row['timestamp']) if row.get('timestamp') else None
        mode=('failed' if row.get('stage')=='failed' else 'complete' if complete else
              'running' if live else 'interrupted' if row and row.get('stage')!='queued' else 'queued')
        point={k:row[k] for k in ('case','trial','phase','planned_requests','dispatched_requests',
               'completed_requests','failed_requests','inflight_requests','score_window_s',
               'progress_percent','elapsed_s','note') if k in row}
        point.update(estimated=False,basis='client_telemetry',sample_age_s=age,stale=age is None or age>10)
        tasks.append(dict(id=task['id'],label=task['label'],state=mode,live=live,stage=row.get('stage'),
                          nodes=task.get('nodes',[]),valid=valid,expected=expected,point=point,
                          timestamp=row.get('timestamp'),sample_age_s=age,cleanup_verified=complete))
    complete=all(t['state']=='complete' for t in tasks)
    known=all(t['valid'] is not None for t in tasks)
    return dict(timestamp=time.time(),title=SPEC.get('title',SPEC['id']),tasks=tasks,
                valid=sum(t['valid'] for t in tasks) if known else None,
                expected=sum(t['expected'] for t in tasks),cleanup_verified=complete,
                concurrent_tasks=sum(t['live'] for t in tasks),
                state='complete' if complete else 'failed' if any(t['state'] in ('failed','interrupted') for t in tasks)
                      else 'running' if any(t['live'] for t in tasks) else 'waiting')
