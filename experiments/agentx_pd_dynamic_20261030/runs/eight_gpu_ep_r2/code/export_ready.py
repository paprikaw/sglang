"""Join two complete rank exports, including gzip CRC and JSON validation."""
import gzip
import hashlib
import json
import time
import zlib

def await_exports(directory, health, timeout=120):
    deadline=time.monotonic()+timeout
    while True:
        health()
        files=sorted(directory.glob('*.trace.json.gz'))
        exports=[]
        if len(files)==2:
            try:
                for path in files:
                    raw=path.read_bytes();data=json.loads(gzip.decompress(raw))
                    if not data.get('traceEvents'):raise ValueError('Empty trace')
                    exports.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
            except (OSError,EOFError,ValueError,zlib.error):exports=[]
        if len(exports)==2:return exports
        if time.monotonic()>deadline:raise TimeoutError('Incomplete two-stage trace export')
        time.sleep(1)
