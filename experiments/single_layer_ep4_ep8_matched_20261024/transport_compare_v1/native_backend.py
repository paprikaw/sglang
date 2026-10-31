"""Capture native initialization and reject a silent NIC-handler fallback."""
import ctypes
import os
from pathlib import Path
import sys


def initialize_with_log(path, initializer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None)
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int
    sys.stdout.flush(); sys.stderr.flush(); libc.fflush(None)
    saved = [os.dup(descriptor) for descriptor in (1, 2)]
    try:
        with path.open('wb') as log:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            try:
                return initializer()
            finally:
                sys.stdout.flush(); sys.stderr.flush(); libc.fflush(None)
    finally:
        for descriptor, original in zip((1, 2), saved):
            os.dup2(original, descriptor)
            os.close(original)


def verify_backend_text(text):
    selected = [line for line in text.splitlines() if 'NIC handler will be ' in line]
    if not selected or any('NIC handler will be CPU with host memory backend.' not in line for line in selected):
        raise ValueError('Missing host-memory-handler activation evidence or actual backend fallback')
    return {'passed': True, 'actual_backend': 'cpu_host_memory', 'selected_lines': selected,
            'fallback_seen': False, 'proof_scope': 'actual NVSHMEM initialization choice, not only environment request'}
