"""Optional 1 Hz client telemetry on a separate writer thread, never inference.

Progress is diagnostic only. The caller's arrival schedule and timing remain
unchanged; no per-request disk writes, no retries, no request payloads exported.
"""
import contextvars
import functools
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

_active=contextvars.ContextVar('experiment_live_progress',default=None)


def counters(rows, wire=True):
    completed=[r for r in rows if 'end_mono' in r]
    return dict(started_requests=len(rows),
                dispatched_requests=sum('http_body_sent_mono' in r for r in rows) if wire else None,
                completed_requests=sum(r.get('passed') is True for r in completed),
                failed_requests=sum(r.get('passed') is not True for r in completed),
                inflight_requests=len(rows)-len(completed))


class Progress:
    def __init__(self):
        self.path=None;self.fields={};self.rows=[];self.wire=True;self.done=threading.Event()
        self.thread=None;self.write_failures=0;self.max_write_ms=0.

    def begin(self,out,name,trial):
        self.path=Path(out)/'live_progress.json'
        self.fields=dict(case=name,trial=trial,point_started_wall=time.time(),phase='context_and_append_warmup')
        d=re.fullmatch(r'D-C(\d+)-O(\d+)-N(\d+)',name)
        if d:
            self.wire=False;self.fields.update(phase='decode_preparation',planned_requests=int(d[3]),
                                             context_tokens=int(d[1]),output_tokens=int(d[2]))
        self.thread=threading.Thread(target=self.loop,name='experiment-progress-writer',daemon=True)
        self.thread.start()

    def snapshot(self):
        now=time.time();fields=dict(self.fields); fields['timestamp']=now
        fields['elapsed_s']=max(0,now-fields['point_started_wall'])
        fields.update(counters(list(self.rows),self.wire))
        arrival=fields.get('arrival_start_wall');window=fields.get('score_window_s')
        if arrival is not None and window is not None and fields['phase'] not in ('validated','invalid','validation'):
            elapsed=now-arrival
            fields['phase']='arrival_warmup' if elapsed<30 else 'score_window' if elapsed<30+window else 'drain_or_validation'
            fields['progress_percent']=min(100,max(0,elapsed-30)*100/window)
            fields['window_remaining_estimate_s']=max(0,30+window-elapsed)
        fields['telemetry_write_failures']=self.write_failures
        fields['telemetry_max_write_ms']=self.max_write_ms
        return fields

    def write(self):
        start=time.monotonic();tmp=None
        try:
            tmp=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
            tmp.write_text(json.dumps(self.snapshot(),separators=(',',':'))+'\n')
            os.replace(tmp,self.path)
        except OSError:
            self.write_failures+=1
            if tmp:
                try:tmp.unlink(missing_ok=True)
                except OSError:pass
        finally:self.max_write_ms=max(self.max_write_ms,(time.monotonic()-start)*1000)

    def loop(self):
        while not self.done.is_set():
            self.write();self.done.wait(1.)
        self.write()

    def close(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=2.)


def begin(out,name,trial):
    if _active.get():_active.get().begin(out,name,trial)

def observe(rows):
    if _active.get():_active.get().rows=rows

def phase(name,**fields):
    if _active.get():_active.get().fields.update(phase=name,**fields)

def tracked_point(fn):
    @functools.wraps(fn)
    async def wrapped(*args,**kw):
        progress=Progress();token=_active.set(progress)
        try:
            result=await fn(*args,**kw)
            phase('validated' if result.get('passed') else 'invalid')
            return result
        finally:
            progress.close();_active.reset(token)
    return wrapped
