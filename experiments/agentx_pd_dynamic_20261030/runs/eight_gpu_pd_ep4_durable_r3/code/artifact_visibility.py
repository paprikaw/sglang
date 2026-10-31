#!/usr/bin/env python3
"""Wait for a remote-produced JSON artifact to become visible and stable.

Shared filesystems can report a remote process exit before a newly renamed file
is visible on the controller host. Controllers should use this barrier before
classifying an otherwise successful remote benchmark as failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class ArtifactVisibilityTimeout(TimeoutError):
    def __init__(self, receipt: dict[str, Any]):
        super().__init__(
            f"artifact did not become visible and stable: {receipt['path']}"
        )
        self.receipt = receipt


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def wait_for_stable_json(
    path: str | Path,
    *,
    timeout_s: float = 15.0,
    poll_s: float = 0.05,
    stable_observations: int = 2,
    min_stable_s: float = 0.10,
) -> tuple[Any, dict[str, Any]]:
    """Return parsed JSON after identical, parseable observations are stable.

    The receipt intentionally contains only metadata and a digest, never the
    result payload.
    """
    target = Path(path)
    if timeout_s <= 0 or poll_s <= 0 or stable_observations < 2 or min_stable_s < 0:
        raise ValueError("invalid visibility-barrier parameters")
    started = time.monotonic()
    deadline = started + timeout_s
    attempts = 0
    consecutive = 0
    first_stable_at: float | None = None
    previous: tuple[int, str] | None = None
    last_error = "not observed"
    while True:
        attempts += 1
        now = time.monotonic()
        try:
            raw = target.read_bytes()
            parsed = json.loads(raw)
            signature = (len(raw), hashlib.sha256(raw).hexdigest())
            if signature == previous:
                consecutive += 1
            else:
                previous = signature
                consecutive = 1
                first_stable_at = now
            last_error = ""
            stable_for = now - (first_stable_at if first_stable_at is not None else now)
            if consecutive >= stable_observations and stable_for >= min_stable_s:
                receipt = {
                    "path": str(target.resolve()),
                    "visible": True,
                    "parseable_json": True,
                    "stable": True,
                    "size_bytes": signature[0],
                    "sha256": signature[1],
                    "observations": consecutive,
                    "attempts": attempts,
                    "waited_s": now - started,
                }
                return parsed, receipt
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            previous = None
            consecutive = 0
            first_stable_at = None
            last_error = f"{type(exc).__name__}: {exc}"
        if now >= deadline:
            raise ArtifactVisibilityTimeout(
                {
                    "path": str(target.absolute()),
                    "visible": target.exists(),
                    "parseable_json": False,
                    "stable": False,
                    "attempts": attempts,
                    "waited_s": now - started,
                    "last_error": last_error,
                }
            )
        time.sleep(min(poll_s, max(0.0, deadline - now)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--poll-s", type=float, default=0.05)
    parser.add_argument("--stable-observations", type=int, default=2)
    parser.add_argument("--min-stable-s", type=float, default=0.10)
    parser.add_argument("--receipt")
    args = parser.parse_args()
    try:
        _, receipt = wait_for_stable_json(
            args.path,
            timeout_s=args.timeout_s,
            poll_s=args.poll_s,
            stable_observations=args.stable_observations,
            min_stable_s=args.min_stable_s,
        )
        code = 0
    except ArtifactVisibilityTimeout as exc:
        receipt = exc.receipt
        code = 1
    if args.receipt:
        _atomic_json(Path(args.receipt), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
