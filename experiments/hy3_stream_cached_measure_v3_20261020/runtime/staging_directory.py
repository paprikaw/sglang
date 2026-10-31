"""Create only this authenticated node owner's fresh bounded-staging directory.

Pure CPU filesystem helper. The caller must authenticate the exact frozen run,
node specification and lease before invoking it. It never reuses or removes an
existing directory, changes permissions, or creates a staging ledger itself.
"""
import os
from pathlib import Path
import re
import stat


def create_staging_parent(run_id, staging_ledger):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("A simple frozen run identifier is required")
    parent = Path("/tmp") / f"{os.getuid()}-{run_id}-staging"
    expected = parent / "registry.json"
    if not isinstance(staging_ledger, str) or staging_ledger != str(expected):
        raise ValueError("Staging ledger differs from the unique UID/run-scoped path")
    # Atomic exclusive creation rejects stale directories, symlinks and any
    # competing owner's path. Do not repair/chmod or inspect their ledger.
    os.mkdir(parent, mode=0o700)
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        current = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
                or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino)):
            raise RuntimeError("Staging directory must remain private and UID-owned")
        if os.listdir(fd):
            raise RuntimeError("Fresh staging directory unexpectedly has contents")
        return {"path": str(parent), "ledger": str(expected), "uid": info.st_uid,
                "device": info.st_dev, "inode": info.st_ino,
                "mode": stat.S_IMODE(info.st_mode), "created_by_node_owner": True}
    finally:
        os.close(fd)
