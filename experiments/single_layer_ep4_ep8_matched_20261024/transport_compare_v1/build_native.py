"""Guard-authenticated CPU compilation of the Hy3 CE shape specialization.

The five CE-only H5120 constants become H4096. Host, RDMA kernels, launch
configuration, notification protocol and remaining CUDA objects are unchanged.
Both transport policies use the resulting identical native for Hy3.
"""
import argparse
import hmac
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
from common import ROOT, DEEP, CACHED_NATIVE_RECEIPT, NATIVE_SO, NATIVE_SHA, save, sha
from guard_authorization import authorization_signature, process_ticks, validate_guard
from process_control import stop_owned, track_descendants


def base():
    result=json.loads(CACHED_NATIVE_RECEIPT.read_text())
    assert result['passed'] and result['native_sha256']==NATIVE_SHA
    return result


def inputs():
    result=base()
    # Include every header/source from the pinned base plus actual borrowed objects.
    paths=[CACHED_NATIVE_RECEIPT,NATIVE_SO,ROOT/'hy3_internode_ll.cu']
    names=subprocess.check_output(['git','-C',str(DEEP),'ls-files'],text=True).splitlines()
    paths += [DEEP/name for name in names if (DEEP/name).is_file()]
    paths += [Path(o['path']) for o in result['input_objects']]
    return list(dict.fromkeys(paths))


def recipe(run):
    result=base();root=Path(run)/'candidate'
    old=CACHED_NATIVE_RECEIPT.parent
    replacements={str(old/'build/internode_ll.o'):str(root/'build/internode_ll.o'),
                  str(old/'build/dlink.o'):str(root/'build/dlink.o'),
                  str(NATIVE_SO):str(root/'python'/NATIVE_SO.name),
                  str(DEEP/'csrc/kernels/internode_ll.cu'):str(ROOT/'hy3_internode_ll.cu')}
    commands=[[replacements.get(a,a) for a in c] for c in result['commands']]
    commands[0].insert(1,'-I'+str(DEEP/'csrc/kernels'))
    assert len(commands)==3 and '-c' in commands[0] and '-dlink' in commands[1] and '-shared' in commands[2]
    assert str(ROOT/'hy3_internode_ll.cu') in commands[0]
    return commands


def verify_inputs(run):
    frozen={r['path']:r['sha256'] for r in json.loads((Path(run)/'frozen_inputs.json').read_text())['files']}
    assert sha(NATIVE_SO)==NATIVE_SHA
    for path in inputs():
        assert sha(path)==frozen[str(path.resolve())],str(path)
    original=(DEEP/'csrc/kernels/internode_ll.cu').read_text()
    end=original.index('template <int kNumThreads>')
    assert original[:end].count('5120')==5
    assert (ROOT/'hy3_internode_ll.cu').read_text()==original[:end].replace('5120','4096')+original[end:]
    for row in base()['input_objects']:
        assert sha(row['path'])==row['sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run.resolve();cfg=json.loads((run/'config.json').read_text())
    packet=json.loads(sys.stdin.readline());payload=packet['payload']
    node=socket.gethostname().split('.')[0]
    guard=validate_guard(run,cfg,payload['guard_env'],run/'artifact_visibility/build')
    assert hmac.compare_digest(packet['signature'],authorization_signature(payload,guard['token']))
    assert payload['node']==node and node==cfg['nodes'][0]
    assert payload['phase']=='native-build' and payload['lease_id']==guard['lease_id']
    assert payload['config_sha256']==sha(run/'config.json')
    verify_inputs(run)
    root=run/'candidate';(root/'build').mkdir(parents=True,exist_ok=False);(root/'python').mkdir()
    native=root/'python'/NATIVE_SO.name
    cancelled={'signal':0}
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(signum,lambda number,frame:cancelled.update(signal=number))
    link=select.poll();link.register(sys.stdin.fileno(),select.POLLIN|select.POLLHUP|select.POLLERR)
    started=time.time();passed=False;error=None;commands=recipe(run)
    try:
        with (root/'build.log').open('w') as log:
            for index,command in enumerate(commands):
                validate_guard(run,cfg,payload['guard_env'],run/'artifact_visibility/build')
                child=subprocess.Popen(command,cwd=DEEP,stdout=log,stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL,start_new_session=True)
                owned={child.pid:process_ticks(child.pid)};deadline=time.monotonic()+600
                try:
                    while child.poll() is None:
                        track_descendants(child.pid,owned)
                        if cancelled['signal'] or time.monotonic()>deadline:
                            raise InterruptedError('Compilation cancelled or timed out')
                        if link.poll(100):
                            os.read(sys.stdin.fileno(),4096)
                            raise InterruptedError('Controller connection closed or cancelled')
                    if child.returncode:raise RuntimeError(f'Build stage {index} failed')
                finally:
                    assert stop_owned(child,owned),'Build descendants still running'
        verify_inputs(run);passed=True
    except BaseException as exc:
        error=str(exc);raise
    finally:
        save(root/'build_result.json',dict(passed=passed,node=node,
             host_base_native_sha256=NATIVE_SHA,host_commit=base()['source_commit'],
             cuda_base_commit=base()['cuda_override_commit'],
             cuda_source_path=str(ROOT/'hy3_internode_ll.cu'),cuda_source_sha256=sha(ROOT/'hy3_internode_ll.cu'),
             change='Only five CE H5120 constants become H4096; no SM/protocol/other source changes',
             native_path=str(native),native_sha256=sha(native) if passed else None,
             commands=commands,borrowed_objects=base()['input_objects'],
             started_at=started,finished_at=time.time(),error=error,gpu_execution=False,
             model_loaded=False,guard_lease_id=guard['lease_id']))


if __name__=='__main__':main()
