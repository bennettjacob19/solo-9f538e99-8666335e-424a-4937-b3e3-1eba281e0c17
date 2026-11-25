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
    rebuild_tail_members,
    validate_gzip,
)
from app.main import MAX_RANGE_LENGTH, create_app  # noqa: E402
from app.storage import (  # noqa: E402
    STATE_BASIC,
    STATE_FAST,
    STATE_REPAIRING,
    STATE_UPGRADING,
    Store,
)
from app.upgrade import run_upgrade  # noqa: E402


def make_gzip(path: str, body: bytes, compresslevel: int = 6):
    with gzip.open(path, "wb", compresslevel=compresslevel) as fp:
        fp.write(body)


def gzip_member(body: bytes, pad_to: int | None = None) -> bytes:
    """Build one gzip member, optionally padded to an exact compressed length.

    Padding uses the RFC 1952 FEXTRA header field, so the result stays a valid
    standard gzip member. ``pad_to=65536`` reproduces the historic bug: the
    member's final byte lands exactly on a 64 KiB read boundary, which made the
    old codec stop before the following member.
    """
    import struct
    import zlib
    cobj = zlib.compressobj(6, zlib.DEFLATED, 31)
    data = cobj.compress(body) + cobj.flush()
    if pad_to is not None:
        need = pad_to - len(data)
        assert need >= 2
        header = bytearray(data[:10])
        header[3] |= 0x04  # FLG.FEXTRA
        data = (bytes(header) + struct.pack("<H", need - 2)
                + b"X" * (need - 2) + data[10:])
        assert len(data) == pad_to
    return data


def make_boundary_gzip(path: str):
    """Two-member gzip; member one ends at exactly the 64 KiB read boundary.

    Returns (full_body, truncated_body) where truncated_body is what the old
    buggy codec used to register (everything after the boundary member lost).
    """
    body1 = (b"0123456789ABCDEF" * (MEMBER_SIZE // 16))[:MEMBER_SIZE]
    body2 = b"".join(
        (b"tail-%08d: " % i) + b"z" * 40 + b"\n" for i in range(3000))
    blob = gzip_member(body1, pad_to=MEMBER_SIZE) + gzip_member(body2)
    with open(path, "wb") as fp:
        fp.write(blob)
    return body1 + body2, body1


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


class BoundaryCodecTests(unittest.TestCase):
    """Regression: a member ending exactly at a 64 KiB read boundary."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gz_path = os.path.join(self.tmp, "boundary.gz")
        self.body, self.truncated = make_boundary_gzip(self.gz_path)
        self.old_size = len(self.truncated)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fixture_is_valid_standard_gzip(self):
        # The system gzip tool must see the whole concatenated stream.
        proc = subprocess.run(["gzip", "-dc", self.gz_path],
                              capture_output=True, check=True)
        self.assertEqual(proc.stdout, self.body)

    def test_validate_measures_beyond_boundary_member(self):
        self.assertEqual(validate_gzip(self.gz_path), len(self.body))
        with open(self.gz_path, "rb") as fp:
            self.assertEqual(b"".join(iter_decompressed(fp)), self.body)

    def test_basic_range_reads_reach_the_tail(self):
        data, _ = read_range_basic(
            self.gz_path, self.old_size, min(4096, len(self.body) - self.old_size))
        self.assertEqual(data, self.body[self.old_size:self.old_size + len(data)])

    def test_rebuild_tail_members_covers_only_the_missing_tail(self):
        out = os.path.join(self.tmp, "tail.gz")
        captured = io.BytesIO()
        members, tail_size, total = rebuild_tail_members(
            self.gz_path, self.old_size, captured.write)
        with open(out, "wb") as fp:
            fp.write(captured.getvalue())
        self.assertEqual(total, len(self.body))
        self.assertEqual(tail_size, len(self.body) - self.old_size)
        self.assertTrue(members)
        # Every member respects the 64 KiB uncompressed cap.
        self.assertTrue(all(m["usize"] <= MEMBER_SIZE for m in members))
        # First fragment member starts at the old readable boundary.
        self.assertEqual(members[0]["uoff"], self.old_size)
        with open(out, "rb") as fp:
            tail = b"".join(iter_decompressed(fp))
        self.assertEqual(tail, self.body[self.old_size:])

    def test_truncated_stream_still_rejected(self):
        with open(self.gz_path, "rb") as fp:
            blob = fp.read()
        bad = os.path.join(self.tmp, "trunc.gz")
        with open(bad, "wb") as fp:
            fp.write(blob[:40000])
        with self.assertRaises(GzipError):
            validate_gzip(bad)


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


class RepairTests(unittest.TestCase):
    """Web repair of historic logs whose registered size missed the tail."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmp, "data")
        self.app = create_app(self.data_dir)
        self.client = self.app.test_client()
        self.store: Store = self.app.config["STORE"]
        self.gz_path = os.path.join(self.tmp, "boundary.gz")
        self.body, self.truncated = make_boundary_gzip(self.gz_path)
        self.old_size = len(self.truncated)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed_legacy(self, log_id: int, state: str):
        """Place a valid original gzip with a truncated registered size."""
        os.makedirs(self.store.log_dir(log_id), exist_ok=True)
        shutil.copyfile(self.gz_path, self.store.original_path(log_id))
        self.store.meta["next_id"] = log_id + 1
        self.store.meta["logs"][str(log_id)] = {
            "filename": f"legacy-{log_id}.gz",
            "state": state,
            "uncompressed_size": self.old_size,
            "cap_bytes": None,
        }
        self.store._save_meta()

    def _truncate_fast_artifacts(self, log_id: int):
        """Turn a freshly upgraded log into a legacy truncated-fast log."""
        index = self.store.load_fast_index(log_id)
        keep = [m for m in index["members"]
                if m["uoff"] + m["usize"] <= self.old_size]
        self.assertTrue(
            keep and keep[-1]["uoff"] + keep[-1]["usize"] == self.old_size)
        gz_len = keep[-1]["coff"] + keep[-1]["csize"]
        with open(self.store.gzip_path(log_id), "ab") as fp:
            fp.truncate(gz_len)
        with open(self.store.index_path(log_id), "w") as fp:
            json.dump({"version": 1, "member_size": index["member_size"],
                       "uncompressed_size": self.old_size,
                       "members": keep}, fp)
        self.store.meta["logs"][str(log_id)]["uncompressed_size"] = self.old_size
        self.store._save_meta()

    # ----- new uploads are correct out of the box --------------------------
    def test_new_upload_registers_full_size(self):
        with open(self.gz_path, "rb") as fp:
            r = self.client.post(
                "/upload", data={"file": (fp, "b.gz")},
                content_type="multipart/form-data")
        self.assertEqual(r.status_code, 201, r.data)
        log_id = r.get_json()["id"]
        self.assertEqual(r.get_json()["uncompressed_size"], len(self.body))
        r = self.client.get(
            f"/logs/{log_id}/range?start={self.old_size}&length=256")
        self.assertEqual(r.data, self.body[self.old_size:self.old_size + 256])
        c = self.client.get(f"/logs/{log_id}/content")
        self.assertEqual(c.data, self.body)

    # ----- basic log repair ------------------------------------------------
    def test_repair_basic_log_corrects_size_and_keeps_reads(self):
        lid = 1
        self._seed_legacy(lid, STATE_BASIC)
        # Before repair: tail unreachable, download still complete.
        r = self.client.get(
            f"/logs/{lid}/range?start={self.old_size}&length=100")
        self.assertEqual(r.status_code, 416)
        d = self.client.get(f"/logs/{lid}/download")
        with open(self.gz_path, "rb") as fp:
            self.assertEqual(d.data, fp.read())
        d.close()

        r = self.client.post(f"/logs/{lid}/repair", json={})
        self.assertEqual(r.status_code, 200, r.data)
        j = r.get_json()
        self.assertTrue(j["changed"])
        self.assertEqual(j["uncompressed_size"], len(self.body))
        self.assertEqual(j["tail_bytes"], len(self.body) - self.old_size)

        r = self.client.get(
            f"/logs/{lid}/range?start={self.old_size}&length=100")
        self.assertEqual(r.data, self.body[self.old_size:self.old_size + 100])
        c = self.client.get(f"/logs/{lid}/content")
        self.assertEqual(c.data, self.body)
        # Re-running repair is an idempotent no-op.
        r2 = self.client.post(f"/logs/{lid}/repair", json={})
        self.assertFalse(r2.get_json()["changed"])

    # ----- fast log repair -------------------------------------------------
    def test_repair_fast_log_appends_tail_and_fully_enables_fast_reads(self):
        lid = 2
        self._seed_legacy(lid, STATE_BASIC)
        # Correct size temporarily so the upgrade builds full artifacts...
        self.store.meta["logs"][str(lid)]["uncompressed_size"] = len(self.body)
        run_upgrade(self.store, lid, 50_000_000)
        # ...then emulate the historic truncated registration.
        self._truncate_fast_artifacts(lid)
        before = self.store.fast_artifacts_bytes(lid)

        # Prefix fast reads still work; tail is beyond the registered end.
        r = self.client.get(f"/logs/{lid}/range?start=10&length=90")
        self.assertEqual(r.headers["X-Read-Mode"], "fast")
        self.assertEqual(r.data, self.body[10:100])
        r = self.client.get(
            f"/logs/{lid}/range?start={self.old_size}&length=100")
        self.assertEqual(r.status_code, 416)

        r = self.client.post(f"/logs/{lid}/repair", json={})
        self.assertEqual(r.status_code, 200, r.data)
        j = r.get_json()
        self.assertEqual(j["uncompressed_size"], len(self.body))
        # New disk usage is only the tail gzip + its index fragment.
        self.assertEqual(j["repair_peak_extra_bytes"],
                         j["gzip_tail_bytes"] + j["index_tail_bytes"])
        after = self.store.fast_artifacts_bytes(lid)
        self.assertLessEqual(
            after - before,
            j["gzip_tail_bytes"] + j["index_tail_bytes"] + 64)
        # No staging files remain; base gzip/index were not rewritten.
        for name in ("gzip.repair", "index.tail", "repair.json"):
            self.assertFalse(
                os.path.exists(os.path.join(self.store.fast_dir(lid), name)))

        # Whole log is fast-readable now, tail included, within the limits.
        r = self.client.get(
            f"/logs/{lid}/range?start={self.old_size - 10}&length=200")
        self.assertEqual(r.headers["X-Read-Mode"], "fast")
        self.assertEqual(
            r.data, self.body[self.old_size - 10:self.old_size + 190])
        self.assertLessEqual(
            int(r.headers["X-Decompressed-Bytes"]), MAX_DECOMPRESS_PER_RANGE)
        c = self.client.get(f"/logs/{lid}/content")
        self.assertEqual(c.data, self.body)
        st = self.client.get(f"/logs/{lid}/status").get_json()
        self.assertEqual(st["state"], STATE_FAST)
        self.assertEqual(st["uncompressed_size"], len(self.body))
        self.assertEqual(
            st["repair_peak_extra_bytes"], j["repair_peak_extra_bytes"])
        # The appended multi-member gzip still decodes as one standard stream.
        with open(self.store.gzip_path(lid), "rb") as fp:
            self.assertEqual(b"".join(iter_decompressed(fp)), self.body)

    def test_repair_fast_reads_never_decompress_more_than_limit(self):
        lid = 3
        self._seed_legacy(lid, STATE_BASIC)
        self.store.meta["logs"][str(lid)]["uncompressed_size"] = len(self.body)
        run_upgrade(self.store, lid, 50_000_000)
        self._truncate_fast_artifacts(lid)
        self.client.post(f"/logs/{lid}/repair", json={})
        import random
        random.seed(11)
        max_dec = 0
        for _ in range(20):
            start = random.randint(0, len(self.body) - 1)
            length = random.randint(1, MAX_RANGE_LENGTH)
            r = self.client.get(
                f"/logs/{lid}/range?start={start}&length={length}")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers["X-Read-Mode"], "fast")
            self.assertEqual(
                r.data, self.body[start:min(start + length, len(self.body))])
            max_dec = max(max_dec, int(r.headers["X-Decompressed-Bytes"]))
        self.assertLessEqual(max_dec, MAX_DECOMPRESS_PER_RANGE)
        self.assertLessEqual(max_dec, 2 * MEMBER_SIZE)

    def test_repair_write_failure_rolls_back_and_retries(self):
        lid = 4
        self._seed_legacy(lid, STATE_BASIC)
        self.store.meta["logs"][str(lid)]["uncompressed_size"] = len(self.body)
        run_upgrade(self.store, lid, 50_000_000)
        self._truncate_fast_artifacts(lid)
        prefix_len = os.path.getsize(self.store.gzip_path(lid))

        r = self.client.post(
            f"/logs/{lid}/repair", json={"fail_after_bytes": 256})
        self.assertEqual(r.status_code, 500)
        st = self.client.get(f"/logs/{lid}/status").get_json()
        self.assertEqual(st["state"], STATE_FAST)
        self.assertEqual(st["uncompressed_size"], self.old_size)
        # Everything staged is gone; the old prefix is byte-intact and readable.
        self.assertFalse(os.path.exists(self.store.repair_gzip_tmp(lid)))
        self.assertFalse(os.path.exists(self.store.repair_state_path(lid)))
        self.assertEqual(
            os.path.getsize(self.store.gzip_path(lid)), prefix_len)
        r = self.client.get(f"/logs/{lid}/range?start=0&length=77")
        self.assertEqual(r.headers["X-Read-Mode"], "fast")
        self.assertEqual(r.data, self.body[:77])
        # Retry succeeds.
        r = self.client.post(f"/logs/{lid}/repair", json={})
        self.assertEqual(r.status_code, 200)
        r = self.client.get(
            f"/logs/{lid}/range?start={self.old_size}&length=64")
        self.assertEqual(r.data, self.body[self.old_size:self.old_size + 64])

    def test_reads_and_download_continue_during_repair(self):
        lid = 5
        self._seed_legacy(lid, STATE_BASIC)
        self.store.meta["logs"][str(lid)]["uncompressed_size"] = len(self.body)
        run_upgrade(self.store, lid, 50_000_000)
        self._truncate_fast_artifacts(lid)
        gz_len = os.path.getsize(self.store.gzip_path(lid))
        self.store.mark_repairing(lid, self.old_size, len(self.body), gz_len)
        try:
            r = self.client.get(f"/logs/{lid}/range?start=1&length=99")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers["X-Read-Mode"], "basic")
            self.assertEqual(r.data, self.body[1:100])
            d = self.client.get(f"/logs/{lid}/download")
            self.assertEqual(d.status_code, 200)
            with open(self.gz_path, "rb") as fp:
                self.assertEqual(d.data, fp.read())
            d.close()
        finally:
            self.store.rollback_repair(lid)
        self.assertEqual(self.store.get(lid)["state"], STATE_FAST)

    def test_repair_survives_restart_in_either_outcome(self):
        lid = 6
        self._seed_legacy(lid, STATE_BASIC)
        self.store.meta["logs"][str(lid)]["uncompressed_size"] = len(self.body)
        run_upgrade(self.store, lid, 50_000_000)
        self._truncate_fast_artifacts(lid)
        # A clean repair, then a fresh Store instance simulates a restart.
        r = self.client.post(f"/logs/{lid}/repair", json={})
        self.assertEqual(r.status_code, 200)
        restarted = Store(self.data_dir)
        entry = restarted.get(lid)
        self.assertEqual(entry["state"], STATE_FAST)
        self.assertEqual(entry["uncompressed_size"], len(self.body))
        app2 = create_app(self.data_dir)
        client2 = app2.test_client()
        r = client2.get(
            f"/logs/{lid}/range?start={self.old_size}&length=128")
        self.assertEqual(r.headers["X-Read-Mode"], "fast")
        self.assertEqual(r.data, self.body[self.old_size:self.old_size + 128])

    def test_recovery_adopts_repair_committed_before_meta_save(self):
        """Crash after append + index publish and staging cleanup, pre meta."""
        from app.gzcodec import rebuild_tail_members
        from app.upgrade import ArtifactWriter
        lid = 8
        self._seed_legacy(lid, STATE_BASIC)
        self.store.meta["logs"][str(lid)]["uncompressed_size"] = len(self.body)
        run_upgrade(self.store, lid, 50_000_000)
        index = self.store.load_fast_index(lid)
        self._truncate_fast_artifacts(lid)
        gz_len = os.path.getsize(self.store.gzip_path(lid))
        self.store.mark_repairing(lid, self.old_size, len(self.body), gz_len)
        writer = ArtifactWriter(10**18)
        with open(self.store.repair_gzip_tmp(lid), "wb") as fp:
            members, _, _ = rebuild_tail_members(
                self.store.original_path(lid), self.old_size,
                writer.bind(fp))
        for member in members:
            member["coff"] += gz_len
        # Perform the durable commit steps, then remove staging, but leave the
        # meta state at "repairing" -- exactly the crash window.
        with open(self.store.repair_gzip_tmp(lid), "rb") as src, \
                open(self.store.gzip_path(lid), "ab") as dst:
            dst.write(src.read())
        fragment = {
            "version": 1,
            "member_size": index["member_size"],
            "uncompressed_size": len(self.body),
            "members": members,
        }
        with open(self.store.repair_index_tmp(lid), "wb") as fp:
            fp.write(json.dumps(fragment, indent=2).encode())
        os.replace(self.store.repair_index_tmp(lid),
                   self.store.tail_index_path(lid))
        os.remove(self.store.repair_gzip_tmp(lid))
        self.assertEqual(self.store.get(lid)["state"], STATE_REPAIRING)

        restarted = Store(self.data_dir)
        entry = restarted.get(lid)
        self.assertEqual(entry["state"], STATE_FAST)
        self.assertTrue(entry.get("recovered"))
        self.assertEqual(entry["uncompressed_size"], len(self.body))
        app2 = create_app(self.data_dir)
        client2 = app2.test_client()
        r = client2.get(
            f"/logs/{lid}/range?start={self.old_size}&length=256")
        self.assertEqual(r.headers["X-Read-Mode"], "fast")
        self.assertEqual(r.data, self.body[self.old_size:self.old_size + 256])
        self.assertFalse(
            os.path.exists(self.store.repair_state_path(lid)))

    def test_repair_uses_only_tail_index_not_a_full_index_copy(self):
        lid = 7
        self._seed_legacy(lid, STATE_BASIC)
        self.store.meta["logs"][str(lid)]["uncompressed_size"] = len(self.body)
        run_upgrade(self.store, lid, 50_000_000)
        full_index = self.store.load_fast_index(lid)
        full_member_count = len(full_index["members"])
        self._truncate_fast_artifacts(lid)
        r = self.client.post(f"/logs/{lid}/repair", json={})
        j = r.get_json()
        # Base index is untouched; tail fragment holds only tail members.
        base_index = read_index(self.store.index_path(lid))
        self.assertEqual(
            base_index["uncompressed_size"], self.old_size)
        tail_index = read_index(self.store.tail_index_path(lid))
        self.assertEqual(
            tail_index["uncompressed_size"], len(self.body))
        # The fragment is an index delta, not a rebuilt full-member table.
        self.assertLess(
            len(tail_index["members"]), full_member_count)
        self.assertGreater(j["index_tail_bytes"], 0)
        merged = self.store.load_fast_index(lid)
        self.assertEqual(merged["uncompressed_size"], len(self.body))
        self.assertEqual(
            len(merged["members"]),
            len(base_index["members"]) + len(tail_index["members"]))
        self.assertEqual(len(merged["members"]), full_member_count)


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


class RepairCrashRecoveryTests(unittest.TestCase):
    """Hard crash (os._exit) during a fast-log tail repair, then restart."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmp, "data")
        self.boundary_gz = os.path.join(self.tmp, "boundary.gz")
        self.body, self.truncated = make_boundary_gzip(self.boundary_gz)
        self.old_size = len(self.truncated)
        # Seed the legacy truncated fast log directly on disk first.
        self.app = create_app(self.data_dir)
        store = self.app.config["STORE"]
        lid = 1
        os.makedirs(store.log_dir(lid), exist_ok=True)
        shutil.copyfile(self.boundary_gz, store.original_path(lid))
        store.meta["next_id"] = 2
        store.meta["logs"]["1"] = {
            "filename": "boundary.gz", "state": STATE_BASIC,
            "uncompressed_size": len(self.body), "cap_bytes": None}
        store._save_meta()
        run_upgrade(store, lid, 50_000_000)
        index = store.load_fast_index(lid)
        keep = [m for m in index["members"]
                if m["uoff"] + m["usize"] <= self.old_size]
        gz_len = keep[-1]["coff"] + keep[-1]["csize"]
        with open(store.gzip_path(lid), "ab") as fp:
            fp.truncate(gz_len)
        with open(store.index_path(lid), "w") as fp:
            json.dump({"version": 1, "member_size": index["member_size"],
                       "uncompressed_size": self.old_size,
                       "members": keep}, fp)
        store.meta["logs"]["1"]["uncompressed_size"] = self.old_size
        store._save_meta()
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
                  self.proc.stdout.read().decode("utf-8", "replace"))

    def tearDown(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _request(self, path, data=None, headers=None):
        req = urllib.request.Request(self.base + path, data=data,
                                     headers=headers or {},
                                     method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            body = e.read()
            e.close()
            return e.code, body, dict(e.headers)

    def _restart(self):
        env = dict(os.environ, DATA_DIR=self.data_dir, PORT=str(self.port),
                   APP_HOME=self.project)
        self.proc = subprocess.Popen(
            [sys.executable, "-I",
             os.path.join(self.project, "tests", "_server_bootstrap.py")],
            cwd=self.project,
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    def test_crash_during_repair_then_restart_rolls_back_then_repairs(self):
        log_id = 1
        # 1. hard crash mid-repair
        payload = json.dumps({"crash_after_bytes": 2048}).encode()
        with self.assertRaises((urllib.error.URLError, ConnectionError,
                                http.client.RemoteDisconnected)):
            self._request(
                f"/logs/{log_id}/repair", data=payload,
                headers={"Content-Type": "application/json"})
        self.proc.wait(timeout=20)
        self.assertEqual(self.proc.returncode, 87)

        # 2. restart: repair rolled back, old fast prefix still serves reads
        self._restart()
        code, data, _ = self._request(f"/logs/{log_id}/status")
        self.assertEqual(code, 200)
        st = json.loads(data)
        self.assertEqual(st["state"], STATE_FAST)
        self.assertEqual(st["uncompressed_size"], self.old_size)
        code, data, headers = self._request(
            f"/logs/{log_id}/range?start=0&length=1024")
        self.assertEqual(code, 200)
        self.assertEqual(headers["X-Read-Mode"], "fast")
        self.assertEqual(data, self.body[:1024])
        # download of the original is unaffected
        code, data, _ = self._request(f"/logs/{log_id}/download")
        self.assertEqual(code, 200)
        self.assertEqual(gzip.decompress(data), self.body)

        # 3. clean retry repairs the full log, including the tail
        payload = json.dumps({}).encode()
        code, data, _ = self._request(
            f"/logs/{log_id}/repair", data=payload,
            headers={"Content-Type": "application/json"})
        self.assertEqual(code, 200, data)
        self.assertEqual(json.loads(data)["uncompressed_size"], len(self.body))

        # 4. another restart keeps the repaired tail fast-readable
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self._restart()
        code, data, headers = self._request(
            f"/logs/{log_id}/range?start={self.old_size}&length=4096")
        self.assertEqual(code, 200)
        self.assertEqual(headers["X-Read-Mode"], "fast")
        self.assertEqual(
            data, self.body[self.old_size:self.old_size + 4096])
        self.assertLessEqual(int(headers["X-Decompressed-Bytes"]),
                             MAX_DECOMPRESS_PER_RANGE)
        code, data, _ = self._request(f"/logs/{log_id}/content")
        self.assertEqual(data, self.body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
