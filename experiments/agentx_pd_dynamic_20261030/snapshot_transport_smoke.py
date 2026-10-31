"""CPU-only per-node mmap notification test; no model or CUDA use."""
import json,mmap,multiprocessing as mp,os,time
from snapshot_transport import open_control,write_control,remove_control


def reader(source,queue):
    fd=open_control(source);view=mmap.mmap(fd,1,access=mmap.ACCESS_READ);os.close(fd)
    queue.put(('ready',os.getpid()))
    last=0;deadline=time.monotonic()+30
    while last<32 and time.monotonic()<deadline:
        epoch=view[0]
        if epoch!=last:queue.put((epoch,os.getpid()));last=epoch
        time.sleep(.005)
    view.close()


if __name__=='__main__':
    source='snapshot-smoke-20261031-'+str(os.getpid());write_control(source,0);q=mp.Queue();children=[mp.Process(target=reader,args=(source,q)) for _ in range(4)]
    try:
        for p in children:p.start()
        assert all(q.get(timeout=10)[0]=='ready' for _ in children)
        for epoch in range(1,33):
            write_control(source,epoch);acks=[q.get(timeout=5) for _ in children]
            assert all(a[0]==epoch for a in acks) and len({a[1] for a in acks})==4
        for p in children:p.join(timeout=5);assert p.exitcode==0
        print(json.dumps(dict(node=os.uname().nodename,passed=True,processes=4,epochs=32,acks=128)))
    finally:
        for p in children:
            if p.is_alive():p.terminate();p.join()
        remove_control(source)
