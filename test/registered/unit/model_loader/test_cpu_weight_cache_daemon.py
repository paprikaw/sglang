"""CPU-only lifecycle tests of the actual socket/memfd daemon protocol."""

import concurrent.futures
import mmap
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/weight_cache"
sys.path.insert(0, str(SOURCE))
from cpu_protocol import CpuWeightCacheClient, CpuWeightCacheError, fingerprint_key
from cpu_daemon import CacheFailure, CpuWeightCacheDaemon


class TestCpuWeightCacheDaemon(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cpu-weights-test-")
        self.socket_path = str(Path(self.temp.name) / "cache.sock")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(SOURCE / "cpu_daemon.py"),
                "--socket",
                self.socket_path,
                "--max-bytes",
                "256",
                "--headroom-bytes",
                "0",
                "--source-commit",
                "test-source",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.clients = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.fail(self.proc.communicate()[1].decode())
            try:
                self.client = self.new_client()
                self.client.status()
                return
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.02)
        self.fail("daemon did not start")

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.proc.terminate()
        self.proc.communicate(timeout=10)
        self.temp.cleanup()

    def new_client(self):
        client = CpuWeightCacheClient(self.socket_path, timeout=5)
        self.clients.append(client)
        return client

    def identity(self, name="model-a"):
        fingerprint = {
            "model_revision": name,
            "source": "source-a",
            "rank": 0,
            "topology": {"tp": 4, "pp": 1},
            "dtype": "uint8",
        }
        return fingerprint_key(fingerprint), fingerprint

    def publish(self, client, name="model-a", payload=b"correct", size=128):
        key, fingerprint = self.identity(name)
        reservation = client.acquire(key, size, fingerprint)
        self.assertEqual(reservation["state"], "build")
        os.pwrite(reservation["fd"], payload, 0)
        client.publish(
            key,
            reservation["token"],
            fingerprint,
            {"tensor": {"offset": 0, "length": len(payload)}},
            size,
        )
        return key, fingerprint

    def wait_empty(self):
        for _ in range(100):
            if not self.client.status()["entries"]:
                return
            time.sleep(0.01)
        self.fail("dead producer lease did not recover")

    def test_snapshot_survives_producer_process_exit_and_is_read_only(self):
        key, fingerprint = self.identity()
        script = """
import os, sys
sys.path.insert(0, sys.argv[1])
from cpu_protocol import CpuWeightCacheClient, fingerprint_key
import json
fingerprint=json.loads(sys.argv[3]); key=fingerprint_key(fingerprint)
with CpuWeightCacheClient(sys.argv[2]) as client:
    build=client.acquire(key, 128, fingerprint)
    os.pwrite(build['fd'], b'weights-from-process', 0)
    client.publish(key, build['token'], fingerprint, {'bytes':20})
assert 'torch' not in sys.modules
"""
        import json

        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(SOURCE),
                self.socket_path,
                json.dumps(fingerprint),
            ],
            check=True,
            timeout=10,
        )
        fetched = self.client.get(key, fingerprint)
        self.assertEqual(fetched["state"], "ready")
        with mmap.mmap(
            fetched["fd"], fetched["size"], access=mmap.ACCESS_READ
        ) as memory:
            self.assertEqual(memory[:20], b"weights-from-process")
        with self.assertRaises(OSError):
            os.pwrite(fetched["fd"], b"X", 0)
        with self.assertRaises((OSError, PermissionError)):
            mmap.mmap(fetched["fd"], fetched["size"], access=mmap.ACCESS_WRITE)
        self.client.release(fetched["reader_token"], evict=True)
        self.assertEqual(self.client.status()["current_bytes"], 0)

    def test_concurrent_same_identity_has_one_builder_and_waiting_fetch(self):
        key, fingerprint = self.identity()
        builder = self.client.acquire(key, 128, fingerprint)
        another = self.new_client()
        self.assertEqual(another.acquire(key, 128, fingerprint)["state"], "building")
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(another.get, key, fingerprint, 2)
            time.sleep(0.05)
            self.assertFalse(future.done())
            os.pwrite(builder["fd"], b"complete", 0)
            self.client.publish(
                key, builder["token"], fingerprint, {"tensor": "fixture"}
            )
            fetched = future.result(timeout=3)
        self.assertEqual(os.pread(fetched["fd"], 8, 0), b"complete")
        self.assertEqual(self.client.status()["used_bytes"], 128)
        another.release(fetched["reader_token"])

    def test_byte_limit_counts_builds_and_recovers_disconnect(self):
        key1, fingerprint1 = self.identity("one")
        key2, fingerprint2 = self.identity("two")
        key3, fingerprint3 = self.identity("three")
        other = self.new_client()
        self.client.acquire(key1, 128, fingerprint1)
        build2 = other.acquire(key2, 128, fingerprint2)
        self.assertEqual(self.client.status()["reserved_bytes"], 256)
        with self.assertRaises(CpuWeightCacheError) as caught:
            self.client.acquire(key3, 1, fingerprint3)
        self.assertEqual(caught.exception.code, "capacity")
        other.release(build2["token"])
        self.assertEqual(self.client.acquire(key3, 128, fingerprint3)["state"], "build")
        with self.assertRaises(CpuWeightCacheError) as caught:
            other.acquire(key2, 257, fingerprint2)
        self.assertEqual(caught.exception.code, "capacity")
        self.client.close()
        self.client = self.new_client()
        self.wait_empty()

    def test_crashed_partial_publication_not_visible_and_fd_reclaimed(self):
        key, fingerprint = self.identity()
        script = """
import json, os, sys
sys.path.insert(0, sys.argv[1])
from cpu_protocol import CpuWeightCacheClient, fingerprint_key
fp=json.loads(sys.argv[3]); client=CpuWeightCacheClient(sys.argv[2])
build=client.acquire(fingerprint_key(fp), 128, fp)
os.pwrite(build['fd'], b'partial', 0)
os._exit(0)
"""
        import json

        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(SOURCE),
                self.socket_path,
                json.dumps(fingerprint),
            ],
            check=True,
            timeout=10,
        )
        self.wait_empty()
        self.assertEqual(self.client.get(key, fingerprint)["state"], "missing")
        targets = []
        for path in Path(f"/proc/{self.proc.pid}/fd").iterdir():
            try:
                targets.append(path.readlink().as_posix())
            except FileNotFoundError:
                pass
        self.assertFalse(
            any("memfd:sglang-cpu-weight-cache" in target for target in targets)
        )

    def test_mutable_publish_and_fingerprint_mismatch_are_rejected(self):
        key, fingerprint = self.identity()
        with self.assertRaises(CpuWeightCacheError) as caught:
            self.client.acquire(key, 128, {**fingerprint, "dtype": "float32"})
        self.assertEqual(caught.exception.code, "fingerprint")
        build = self.client.acquire(key, 128, fingerprint)
        with self.assertRaises(CpuWeightCacheError) as caught:
            self.client._request(
                "publish",
                key=key,
                token=build["token"],
                fingerprint=fingerprint,
                manifest={},
                used_bytes=128,
            )
        self.assertEqual(caught.exception.code, "mutable")
        self.client.release(build["token"])

    def test_dead_owner_is_reaped_even_when_child_inherits_socket(self):
        key, fingerprint = self.identity()
        script = """
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
from cpu_protocol import CpuWeightCacheClient, fingerprint_key
fp=json.loads(sys.argv[3]); client=CpuWeightCacheClient(sys.argv[2])
client.acquire(fingerprint_key(fp), 128, fp)
if os.fork() == 0:
    # A surviving child deliberately keeps both cache socket and memfd open.
    # The daemon must recover the original producer's reserved lease anyway.
    os.close(1); os.close(2)
    time.sleep(2)
    os._exit(0)
os._exit(0)
"""
        import json

        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(SOURCE),
                self.socket_path,
                json.dumps(fingerprint),
            ],
            check=True,
            timeout=10,
        )
        self.wait_empty()
        self.assertEqual(self.client.get(key, fingerprint)["state"], "missing")

    def test_eviction_waits_for_all_readers_and_keeps_live_snapshot(self):
        key, fingerprint = self.publish(self.client)
        first = self.client.get(key, fingerprint)
        other = self.new_client()
        second = other.get(key, fingerprint)
        with self.assertRaises(CpuWeightCacheError) as caught:
            self.client.evict(key)
        self.assertEqual(caught.exception.code, "busy")
        released = self.client.release(first["reader_token"], evict=True)
        self.assertEqual(released["state"], "released")
        self.assertEqual(self.client.get(key, fingerprint)["state"], "retiring")
        self.assertEqual(os.pread(second["fd"], 7, 0), b"correct")
        self.assertEqual(other.release(second["reader_token"])["state"], "evicted")
        self.assertEqual(self.client.status()["current_bytes"], 0)

    def test_lru_only_evicts_ready_unborrowed_entries(self):
        key1, fp1 = self.publish(self.client, "one")
        key2, fp2 = self.publish(self.client, "two")
        first = self.client.get(key1, fp1)
        key3, fp3 = self.identity("three")
        build3 = self.client.acquire(key3, 128, fp3)
        self.assertEqual(self.client.get(key2, fp2)["state"], "missing")
        self.assertEqual(os.pread(first["fd"], 7, 0), b"correct")
        self.client.release(build3["token"])
        self.client.release(first["reader_token"])

    def test_cgroup_headroom_rejects_before_memfd_allocation(self):
        daemon = CpuWeightCacheDaemon(
            self.socket_path + ".unused", 256, headroom_bytes=32
        )
        key, fingerprint = self.identity()
        with patch("cpu_daemon.memory_headroom", return_value={"available_bytes": 100}):
            with patch("cpu_daemon.create_memfd") as allocate:
                with self.assertRaises(CacheFailure) as caught:
                    daemon.handle(
                        {
                            "protocol_version": 1,
                            "op": "acquire",
                            "key": key,
                            "fingerprint": fingerprint,
                            "expected_bytes": 80,
                        },
                        "owner",
                        os.getpid(),
                    )
        self.assertEqual(caught.exception.code, "headroom")
        allocate.assert_not_called()

    def test_singleton_lock_and_private_socket(self):
        self.assertEqual(Path(self.socket_path).stat().st_mode & 0o777, 0o600)
        second = subprocess.run(
            [
                sys.executable,
                str(SOURCE / "cpu_daemon.py"),
                "--socket",
                self.socket_path,
                "--max-bytes",
                "256",
                "--headroom-bytes",
                "0",
            ],
            capture_output=True,
            timeout=10,
        )
        self.assertNotEqual(second.returncode, 0)
        self.assertIn(b"already owns", second.stderr)
        self.assertEqual(self.client.status()["source_commit"], "test-source")

    def test_headroom_does_not_double_count_already_charged_build_pages(self):
        daemon = CpuWeightCacheDaemon(
            self.socket_path + ".unused", 512 * 1024, headroom_bytes=0
        )
        key1, fp1 = self.identity("one")
        key2, fp2 = self.identity("two")
        fds = []
        try:
            with patch(
                "cpu_daemon.memory_headroom", return_value={"available_bytes": 1024**2}
            ):
                _, fd1 = daemon.handle(
                    {
                        "protocol_version": 1,
                        "op": "acquire",
                        "key": key1,
                        "fingerprint": fp1,
                        "expected_bytes": 128 * 1024,
                    },
                    "one",
                    os.getpid(),
                )
                fds.append(fd1)
            os.pwrite(fd1, b"x" * (64 * 1024), 0)
            self.assertGreaterEqual(os.fstat(fd1).st_blocks * 512, 64 * 1024)
            # 64 KiB of the first reservation are already charged. Another
            # 128 KiB build needs 192 KiB remaining, not 256 KiB.
            with patch(
                "cpu_daemon.memory_headroom",
                return_value={"available_bytes": 192 * 1024},
            ):
                second, fd2 = daemon.handle(
                    {
                        "protocol_version": 1,
                        "op": "acquire",
                        "key": key2,
                        "fingerprint": fp2,
                        "expected_bytes": 128 * 1024,
                    },
                    "two",
                    os.getpid(),
                )
                fds.append(fd2)
            self.assertEqual(second["state"], "build")
            self.assertEqual(daemon._total(), 256 * 1024)
        finally:
            for fd in fds:
                os.close(fd)
            with daemon.condition:
                for entry in list(daemon.entries.values()):
                    daemon._remove(entry)


if __name__ == "__main__":
    unittest.main()
