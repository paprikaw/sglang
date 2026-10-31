"""Route authentic loader records by the actual target/draft ModelRunner role."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os
from pathlib import Path

_ROLE = ContextVar('glm_weight_load_role', default=None)
_PID = None

class RoleFilter(logging.Filter):
    def __init__(self, role):
        super().__init__(); self.role = role
    def filter(self, record):
        return _ROLE.get() == self.role

def install():
    global _PID
    if _PID == os.getpid(): return
    root = logging.getLogger()
    formatter = next((h.formatter for h in root.handlers if h.formatter), None)
    directory = Path(os.environ['GLM_RUN_DIR']) / os.environ['GLM_ARM']
    directory /= ('spartan-gpgpu065', 'spartan-gpgpu167')[int(os.environ['GLM_NODE_RANK'])]
    for role in ('target', 'draft'):
        path = directory / (role + '_weight.log')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        handler = logging.StreamHandler(os.fdopen(fd, 'a', buffering=1))
        if formatter: handler.setFormatter(formatter)
        handler.addFilter(RoleFilter(role))
        root.addHandler(handler)
    _PID = os.getpid()

@contextmanager
def loading(is_draft_worker):
    install()
    token = _ROLE.set('draft' if is_draft_worker else 'target')
    try: yield
    finally: _ROLE.reset(token)
