"""Private local CPU counter listener; blocking accept and no periodic polling."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import threading


def process_ticks(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    return int(raw[raw.rfind(')')+2:].split()[19])


def read_packet(connection):
    data=bytearray()
    while b'\n' not in data:
        chunk=connection.recv(4096)
        if not chunk:raise RuntimeError('Counter socket closed before packet')
        data.extend(chunk)
        if len(data)>16384:raise RuntimeError('Counter socket packet exceeds16KiB')
    line,extra=bytes(data).split(b'\n',1)
    if extra.strip():raise RuntimeError('Counter socket trailing packet data')
    return json.loads(line)


def peer_identity(connection):
    return struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))


class CounterListener:
    def __init__(self,path,handler):
        self.path=Path(path);self.handler=handler;self.closed=False
        self.server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            self.server.bind(str(self.path));self.path.chmod(0o600);self.server.listen(2)
        except BaseException:
            self.server.close();raise
        self.thread=threading.Thread(target=self._serve,name='ce-counter-listener',daemon=True)
        self.thread.start();atexit.register(self.close)

    def _serve(self):
        while not self.closed:
            try:
                connection,_=self.server.accept() # Blocking stdlib accept releases GIL.
            except OSError:
                if self.closed:return
                raise
            with connection:
                connection.settimeout(10)
                try:
                    peer_pid,peer_uid,_=peer_identity(connection)
                    if peer_uid!=os.getuid():raise RuntimeError('Counter socket caller UID differs')
                    request=read_packet(connection)
                    if request.get('supervisor_pid')!=peer_pid or request.get('supervisor_start_ticks')!=process_ticks(peer_pid):
                        raise RuntimeError('Counter socket caller process identity differs')
                    response=self.handler(request)
                except Exception as error:
                    response=dict(passed=False,error_type=type(error).__name__,error=str(error))
                try:connection.sendall(json.dumps(response,separators=(',',':')).encode()+b'\n')
                except OSError:pass
                finally:
                    # One post-formal receipt per worker. TERM/KILL need not run
                    # Python atexit, so remove this exact path before termination.
                    if response.get('passed'):self.close()

    def close(self):
        if self.closed:return
        self.closed=True
        try:self.server.shutdown(socket.SHUT_RDWR)
        except OSError:pass
        self.server.close()
        try:self.path.unlink()
        except FileNotFoundError:pass


def start_listener(run,arm,rank,ticks,handler):
    directory=Path('/tmp')/f'sglang-ce-counters-{os.getuid()}'
    directory.mkdir(mode=0o700,exist_ok=True)
    info=directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:
        raise RuntimeError('CE counter socket directory is not private and owned')
    digest=hashlib.sha256((str(Path(run).resolve())+':'+arm).encode()).hexdigest()[:16]
    path=directory/f'{digest}-r{rank}-p{os.getpid()}-t{ticks}.sock'
    if len(str(path).encode())>=108:raise RuntimeError('CE counter socket path exceeds Unix limit')
    return CounterListener(path,handler)


def exchange(identity,request,timeout=10):
    """Local supervisor validates the server PID via kernel SO_PEERCRED."""
    connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.settimeout(timeout)
    with connection:
        info=Path(identity['counter_socket']).lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise RuntimeError('CE counter socket is not private and owned')
        connection.connect(identity['counter_socket'])
        peer_pid,peer_uid,_=peer_identity(connection)
        if peer_pid!=identity['pid'] or peer_uid!=os.getuid() or process_ticks(peer_pid)!=identity['process_start_ticks']:
            raise RuntimeError('CE counter socket server ownership differs')
        packet=dict(request,rank=identity['rank'],pid=identity['pid'],process_start_ticks=identity['process_start_ticks'],
                    supervisor_pid=os.getpid(),supervisor_start_ticks=process_ticks(os.getpid()))
        connection.sendall(json.dumps(packet,separators=(',',':')).encode()+b'\n')
        response=read_packet(connection)
    if not response.get('passed'):raise RuntimeError('CE counter socket rejected request: '+str(response))
    return response
