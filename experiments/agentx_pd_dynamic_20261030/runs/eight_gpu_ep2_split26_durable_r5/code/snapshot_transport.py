"""Owned, node-local control byte for untimed forward-counter snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def local_path(source):
    parent=Path('/dev/shm')/f'agentx-forward-{os.getuid()}'
    parent.mkdir(mode=0o700,exist_ok=True)
    s=parent.lstat()
    assert stat.S_ISDIR(s.st_mode) and s.st_uid==os.getuid() and stat.S_IMODE(s.st_mode)==0o700
    return parent/(hashlib.sha256(str(source).encode()).hexdigest()+'.bin')


def open_control(source):
    path=local_path(source)
    fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    s=os.fstat(fd)
    assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and stat.S_IMODE(s.st_mode)==0o600
    assert s.st_size in (0,1)
    if s.st_size==0:os.pwrite(fd,b'\0',0)
    return fd


def write_control(source,epoch):
    assert 0<=epoch<=254
    fd=open_control(source)
    try:
        os.pwrite(fd,bytes([epoch]),0)
        assert os.pread(fd,1,0)==bytes([epoch])
    finally:os.close(fd)
    return dict(epoch=epoch,path=str(local_path(source)),node=os.uname().nodename)


def remove_control(source):
    path=local_path(source)
    if path.exists():
        assert path.lstat().st_uid==os.getuid() and stat.S_ISREG(path.lstat().st_mode)
        path.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--epoch',type=int);p.add_argument('--remove',action='store_true');a=p.parse_args()
    if a.remove:remove_control(a.source);print(json.dumps({'removed':True}))
    else:print(json.dumps(write_control(a.source,a.epoch)))
