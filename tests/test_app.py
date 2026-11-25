"""Functional tests: upload, whole download, range reads, upgrade, failures.

Run with the project's interpreter:  python -m unittest tests.test_app -v
"""

from __future__ import annotations

import gzip
import http.client
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gzcodec import (  # noqa: E402
    MAX_DECOMPRESS_PER_RANGE,
    MEMBER_SIZE,
    GzipError,
    iter_decompressed,
    read_index,
    read_range_basic,
    read_range_fast,
    rebuild_multimember,
    validate_gzip,
)
from app.main import MAX_RANGE_LENGTH, create_app  # noqa: E402
from app.storage import STATE_BASIC, STATE_FAST, STATE_UPGRADING, Store  # noqa: E402


def make_gzip(path: str, body: bytes, compresslevel: int = 6):
    with gzip.open(path, "wb", compresslevel=compresslevel) as fp:
        fp.write(body)


def make_body(n: int) -> bytes:
    # Deterministic, non-uniform content so compressed size is meaningful.
    return (b"line-%08d: the quick brown fox jumps over the lazy dog\n" % 0) * 0 \
        + b"".join(
            (b"entry-%08d: %s\n" % (i, (b"x" * (i % 64)))).ljust(0)[:80]
            for i in range(n // 80 + 1)
        )[:n]


class GzCodecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.body = make_body(300_000)  # spans several 64 KiB members
        self.src = os.path.join(self.tmp, "src.gz")
        make_gzip(self.src, self.body)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_validate_and_multimember_roundtrip(self):
        self.assertEqual(validate_gzip(self.src), len(self.body))
        out = os.path.join(self.tmp, "multi.gz")
        captured = io.BytesIO()
        index, total = rebuild_multimember(
            self.src, lambda data: captured.write(data)
        )
        with open(out, "wb") as fp:
            fp.write(captured.getvalue())
        self.assertEqual(total, len(self.body))
        # Standard tooling must decode the multi-member file to identical bytes.
        with open(out, "rb") as fp:
            got = b"".join(iter_decompressed(fp))
        self.assertEqual(got, self.body)
        # Index is sparse metadata, not a copy of the body.
        self.assertLess(len(json.dumps(index)), len(self.body) // 10)
        self.assertTrue(all(set(m) == {"uoff", "usize", "coff", "csize"}
                            for m in index["members"]))
        self.assertGreaterEqual(len(index["members"]), 3)

    def test_system_gzip_accepts_multimember(self):
        out = os.path.join(self.tmp, "multi.gz")
        with open(out, "wb") as fp:
            rebuild_multimember(self.src, fp.write)
        proc = subprocess.run(["gzip", "-dc", out], capture_output=True, check=True)
        self.assertEqual(proc.stdout, self.body)

    def test_fast_range_matches_basic_and_obeys_limits(self):
        out = os.path.join(self.tmp, "multi.gz")
        idx_path = os.path.join(self.tmp, "index.json")
        with open(out, "wb") as fp:
            index, _ = rebuild_multimember(self.src, fp.write)
        with open(idx_path, "w") as fp:
            json.dump(index, fp)
        index = read_index(idx_path)

        cases = [(0, 1), (10, 4096), (MEMBER_SIZE - 10, 100),
                 (MEMBER_SIZE * 2 + 333, MAX_RANGE_LENGTH),
                 (len(self.body) - 100, 100)]
        max_decompressed = 0
        for start, length in cases:
            fast, dec = read_range_fast(out, index, start, length)
            basic, _ = read_range_basic(self.src, start, length)
            expected = self.body[start:start + length]
            self.assertEqual(fast, expected)
            self.assertEqual(basic, expected)
            max_decompressed = max(max_decompressed, dec)
        self.assertLessEqual(max_decompressed, MAX_DECOMPRESS_PER_RANGE)
        # A 64 KiB request spans at most 2 members -> <= 128 KiB decompressed.
        self.assertLessEqual(max_decompressed, 2 * MEMBER_SIZE)

    def test_rejects_bad_magic(self):
        bad = os.path.join(self.tmp, "bad.gz")
        with open(bad, "wb") as fp:
            fp.write(b"not gzip at all")
        with self.assertRaises(GzipError):
            validate_gzip(bad)

    def test_empty_body(self):
        src = os.path.join(self.tmp, "empty.gz")
        make_gzip(src, b"")
        with open(os.path.join(self.tmp, "e.gz"), "wb") as fp:
            index, total = rebuild_multimember(src, fp.write)
        self.assertEqual(total, 0)
        self.assertEqual(index["members"][0]["usize"], 0)


class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmp, "data")
        self.app = create_app(self.data_dir)
        self.client = self.app.test_client()
        self.body = make_body(400_000)
        self.gz_path = os.path.join(self.tmp, "log1.gz")
        make_gzip(self.gz_path, self.body)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def upload(self, path=None, name="log1.gz"):
        path = path or self.gz_path
        with open(path, "rb") as fp:
            return self.client.post(
                "/upload", data={"file": (fp, name)},
                content_type="multipart/form-data"
            )

    def test_upload_download_and_byte_equality(self):
        r = self.upload()
        self.assertEqual(r.status_code, 201, r.data)
        log_id = r.get_json()["id"]

        d = self.client.get(f"/logs/{log_id}/download")
        self.assertEqual(d.status_code, 200)
        with open(self.gz_path, "rb") as fp:
            self.assertEqual(d.data, fp.read())  # original gzip kept byte-identical
        d.close()

        c = self.client.get(f"/logs/{log_id}/content")
        self.assertEqual(c.data, self.body)

    def test_upload_rejects_non_gzip(self):
        bad = os.path.join(self.tmp, "bad.gz")
        with open(bad, "wb") as fp:
            fp.write(b"plain text pretending to be gzip")
        r = self.upload(bad)
        self.assertEqual(r.status_code, 400)

    def test_basic_range_reads(self):
        log_id = self.upload().get_json()["id"]
        r = self.client.get(f"/logs/{log_id}/range?start=12345&length=777")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data, self.body[12345:12345 + 777])
        self.assertEqual(r.headers["X-Read-Mode"], "basic")
        self.assertIn("X-Decompressed-Bytes", r.headers)

    def test_range_length_capped_at_64kib(self):
        log_id = self.upload().get_json()["id"]
        r = self.client.get(
            f"/logs/{log_id}/range?start=0&length={MAX_RANGE_LENGTH + 1}")
        self.assertEqual(r.status_code, 400)

    def test_upgrade_then_fast_reads_and_peak(self):
        log_id = self.upload().get_json()["id"]
        cap = 1024 * 1024
        r = self.client.post(f"/logs/{log_id}/upgrade",
                             json={"cap_bytes": cap})
        self.assertEqual(r.status_code, 200, r.data)
        summary = r.get_json()
        self.assertLessEqual(summary["peak_bytes"], cap)
        self.assertGreater(summary["index_bytes"], 0)

        st = self.client.get(f"/logs/{log_id}/status").get_json()
        self.assertEqual(st["state"], STATE_FAST)
        self.assertLessEqual(st["peak_bytes"], cap)
        # Original gzip is excluded from the cap/artifact accounting.
        self.assertGreater(st["original_gzip_bytes"], 0)
        self.assertLessEqual(st["current_artifact_bytes"], cap)

        # Range reads now use the index and decompress far less than 256 KiB.
        r = self.client.get(
            f"/logs/{log_id}/range?start={MEMBER_SIZE + 50}&length=60000")
        self.assertEqual(r.headers["X-Read-Mode"], "fast")
        self.assertEqual(r.data, self.body[MEMBER_SIZE + 50:
                                          MEMBER_SIZE + 50 + 60000])
        dec = int(r.headers["X-Decompressed-Bytes"])
        self.assertLessEqual(dec, MAX_DECOMPRESS_PER_RANGE)

        # Gzip produced after upgrade decompresses to exactly the same bytes.
        d = self.client.get(f"/logs/{log_id}/download")
        with open(self.gz_path, "rb") as fp:
            self.assertEqual(gzip.decompress(d.data), gzip.decompress(fp.read()))
        d.close()

    def test_cap_exceeded_leaves_basic_working(self):
        log_id = self.upload().get_json()["id"]
        r = self.client.post(f"/logs/{log_id}/upgrade",
                             json={"cap_bytes": 32})
        self.assertEqual(r.status_code, 507)
        st = self.client.get(f"/logs/{log_id}/status").get_json()
        self.assertEqual(st["state"], STATE_BASIC)
        # No committed fast artifacts; original number still serves data.
        store: Store = self.app.config["STORE"]
        self.assertFalse(os.path.exists(store.gzip_path(log_id)))
        rr = self.client.get(f"/logs/{log_id}/range?start=0&length=123")
        self.assertEqual(rr.data, self.body[:123])
        rr.close()
        dl = self.client.get(f"/logs/{log_id}/download")
        self.assertEqual(dl.status_code, 200)
        dl.close()

    def test_simulated_write_failure_rolls_back(self):
        log_id = self.upload().get_json()["id"]
        r = self.client.post(f"/logs/{log_id}/upgrade",
                             json={"cap_bytes": 10_000_000,
                                   "fail_after_bytes": 2048})
        self.assertEqual(r.status_code, 500)
        st = self.client.get(f"/logs/{log_id}/status").get_json()
        self.assertEqual(st["state"], STATE_BASIC)
        store: Store = self.app.config["STORE"]
        self.assertFalse(os.path.exists(store.gzip_tmp(log_id)))
        # Retry without fault injection succeeds.
        r2 = self.client.post(f"/logs/{log_id}/upgrade",
                              json={"cap_bytes": 10_000_000})
        self.assertEqual(r2.status_code, 200)

    def test_concurrent_read_during_upgrade(self):
        """Reads stay on the original gzip while the upgrade is in progress."""
        log_id = self.upload().get_json()["id"]
        store: Store = self.app.config["STORE"]
        results = {}

        def slow_write(data):
            # Emulate an in-flight upgrade window.
            time.sleep(0.05)
            results["saw"] = True
            return len(data)

        store.mark_upgrading(log_id, 10_000_000)
        try:
            self.assertEqual(store.get(log_id)["state"], STATE_UPGRADING)
            r = self.client.get(f"/logs/{log_id}/range?start=1&length=99")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers["X-Read-Mode"], "basic")
            self.assertEqual(r.data, self.body[1:100])
            d = self.client.get(f"/logs/{log_id}/download")
            self.assertEqual(d.status_code, 200)
            d.close()
        finally:
            store.rollback(log_id)

    def test_index_is_not_full_body(self):
        log_id = self.upload().get_json()["id"]
        self.client.post(f"/logs/{log_id}/upgrade", json={"cap_bytes": 10_000_000})
        store: Store = self.app.config["STORE"]
        index_size = os.path.getsize(store.index_path(log_id))
        # Sparse per-member metadata must be much smaller than the body.
        self.assertLess(index_size, len(self.body) // 20)

    def test_recovery_adopts_completed_but_uncommitted_build(self):
        """Crash between artifact rename and meta save: complete pair is adopted."""
        log_id = self.upload().get_json()["id"]
        store: Store = self.app.config["STORE"]
        # Build the artifacts, then pretend the process died before meta was saved:
        # state stays "upgrading" while final gzip/index are already in place.
        from app.upgrade import run_upgrade
        run_upgrade(store, log_id, 10_000_000)
        store.meta["logs"][str(log_id)]["state"] = STATE_UPGRADING
        store._save_meta()
        # Leave a stale temp file behind as well.
        with open(store.gzip_tmp(log_id), "wb") as fp:
            fp.write(b"stale")

        recovered = Store(self.data_dir)  # fresh instance = restart
        entry = recovered.get(log_id)
        self.assertEqual(entry["state"], STATE_FAST)
        self.assertTrue(entry.get("recovered"))
        self.assertFalse(os.path.exists(store.gzip_tmp(log_id)))
        # Range reads work against the adopted artifacts.
        app2 = create_app(self.data_dir)
        client2 = app2.test_client()
        r = client2.get(f"/logs/{log_id}/range?start=321&length=456")
        self.assertEqual(r.headers["X-Read-Mode"], "fast")
        self.assertEqual(r.data, self.body[321:777])
        r.close()


class RestartRecoveryTests(unittest.TestCase):
    """Crash (os._exit) mid-upgrade in a real server process, then restart."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmp, "data")
        self.body = make_body(500_000)
        self.gz_path = os.path.join(self.tmp, "log.gz")
        make_gzip(self.gz_path, self.body)
        self.port = self._free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.project = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ, DATA_DIR=self.data_dir, PORT=str(self.port),
                   APP_HOME=self.project)
        self.proc = subprocess.Popen(
            [sys.executable, "-I",
             os.path.join(self.project, "tests", "_server_bootstrap.py")],
            cwd=self.project,
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    @staticmethod
    def _free_port():
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _wait_ready(self, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(self.base + "/healthz", timeout=1)
                return
            except Exception:
                time.sleep(0.2)
        self.fail("server did not start: " +
                  self.proc.stdout.read().decode(errors="replace"))

    def tearDown(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _request(self, path, data=None, headers=None):
        req = urllib.request.Request(self.base + path, data=data, headers=headers
                                     or {}, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            body = e.read()
            e.close()
            return e.code, body, dict(e.headers)

    def test_crash_mid_upgrade_then_restart_recovers(self):
        # 1. upload
        boundary = "----t"
        with open(self.gz_path, "rb") as fp:
            blob = fp.read()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"log.gz\"\r\nContent-Type: application/gzip\r\n\r\n"
                ).encode() + blob + f"\r\n--{boundary}--\r\n".encode()
        code, data, _ = self._request(
            "/upload", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(code, 201, data)
        log_id = json.loads(data)["id"]

        # 2. crash mid-upgrade: os._exit closes the connection without a response
        payload = json.dumps({"cap_bytes": 50_000_000,
                              "crash_after_bytes": 4096}).encode()
        with self.assertRaises((urllib.error.URLError, ConnectionError,
                                http.client.RemoteDisconnected)):
            self._request(
                f"/logs/{log_id}/upgrade", data=payload,
                headers={"Content-Type": "application/json"})
        self.proc.wait(timeout=20)
        self.assertEqual(self.proc.returncode, 87)

        # 3. restart: recovery must roll the unfinished upgrade back
        env = dict(os.environ, DATA_DIR=self.data_dir, PORT=str(self.port),
                   APP_HOME=self.project)
        self.proc = subprocess.Popen(
            [sys.executable, "-I",
             os.path.join(self.project, "tests", "_server_bootstrap.py")],
            cwd=self.project,
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._wait_ready()
        code, data, _ = self._request(f"/logs/{log_id}/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(data)["state"], STATE_BASIC)

        # 4. same log number still serves correct bytes
        code, data, headers = self._request(
            f"/logs/{log_id}/range?start=70000&length=1024")
        self.assertEqual(code, 200)
        self.assertEqual(data, self.body[70000:71024])
        code, data, _ = self._request(f"/logs/{log_id}/download")
        self.assertEqual(code, 200)
        self.assertEqual(gzip.decompress(data), self.body)

        # 5. a clean retry after recovery reaches fast mode
        payload = json.dumps({"cap_bytes": 50_000_000}).encode()
        code, data, _ = self._request(
            f"/logs/{log_id}/upgrade", data=payload,
            headers={"Content-Type": "application/json"})
        self.assertEqual(code, 200, data)
        code, data, headers = self._request(
            f"/logs/{log_id}/range?start=200000&length=65536")
        self.assertEqual(code, 200)
        self.assertEqual(headers["X-Read-Mode"], "fast")
        self.assertEqual(data, self.body[200000:265536])
        self.assertLessEqual(int(headers["X-Decompressed-Bytes"]),
                             MAX_DECOMPRESS_PER_RANGE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
