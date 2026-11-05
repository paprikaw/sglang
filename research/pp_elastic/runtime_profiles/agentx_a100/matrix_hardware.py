"""Low-frequency node samples, outside the inference processes."""
import json
from pathlib import Path
import subprocess
import threading
import time


def start(root):
    root = Path(root)
    stop = threading.Event()

    def loop():
        path = root / 'hardware.jsonl'
        while not stop.wait(2):
            if (root / 'control').read_bytes()[0] < 2:
                continue
            row = dict(kind='hardware', unix_ns=time.time_ns(), monotonic_s=time.perf_counter())
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
                row['gpu_csv'] = p.stdout
                row['gpu_query_exit'] = p.returncode
                row['proc_stat'] = Path('/proc/stat').read_text().splitlines()[0]
                row['meminfo'] = {k: v.strip() for line in Path('/proc/meminfo').read_text().splitlines()
                                  for k, v in [line.split(':', 1)] if k in ('MemAvailable', 'MemFree', 'SwapFree')}
                row['network'] = Path('/proc/net/dev').read_text()
                counters = {}
                for f in Path('/sys/class/infiniband').glob('*/ports/*/counters/*'):
                    if f.name in ('port_xmit_data', 'port_rcv_data', 'port_xmit_wait'):
                        counters[str(f)] = f.read_text().strip()
                row['rdma_counters'] = counters
            except Exception as e:
                row['error'] = type(e).__name__ + ':' + str(e)
            with path.open('a') as out:
                out.write(json.dumps(row) + '\n')

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop, thread
