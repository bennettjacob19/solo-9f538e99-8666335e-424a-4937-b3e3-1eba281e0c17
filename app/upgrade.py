"""Fast-mode upgrade transaction.

The upgrade rebuilds the original gzip as a standard multi-member gzip plus a sparse
index. Reads continue to be served from the original gzip while the build runs, and
the new artifacts become visible only via an atomic rename at commit time. Any error
(quota overflow, simulated failure, process restart mid-upgrade) leaves the original
gzip and its log number fully usable; unfinished .new files are discarded.
"""

from __future__ import annotations

import json
import os

from .gzcodec import (
    QuotaExceeded,
    SimulatedWriteFailure,
    rebuild_multimember,
)
from .storage import Store, STATE_BASIC, STATE_FAST


class UpgradeError(RuntimeError):
    pass


class ArtifactWriter:
    """Single accounting point for every byte written to fast-mode artifacts.

    Both the rebuilt gzip and the index go through here, so the configured cap and
    the measured peak are exact. The original gzip is never written here and never
    counts against the cap.
    """

    def __init__(self, cap_bytes: int, fault: dict | None = None):
        self.cap_bytes = cap_bytes
        self.bytes_written = 0
        self.peak = 0
        self.fault = fault or {}

    def write(self, data: bytes, fp=None):
        if not data:
            return
        fail_after = self.fault.get("fail_after_bytes")
        crash_after = self.fault.get("crash_after_bytes")
        if fail_after is not None and self.bytes_written >= fail_after:
            raise SimulatedWriteFailure(
                f"simulated write failure after {self.bytes_written} bytes"
            )
        if crash_after is not None and self.bytes_written >= crash_after:
            os._exit(87)  # hard restart simulation: no flush, no rename
        if self.bytes_written + len(data) > self.cap_bytes:
            raise QuotaExceeded(
                f"new-file cap {self.cap_bytes} bytes exceeded "
                f"(would reach {self.bytes_written + len(data)})"
            )
        target = fp if fp is not None else self._fp
        target.write(data)
        self.bytes_written += len(data)
        self.peak = max(self.peak, self.bytes_written)

    def bind(self, fp):
        self._fp = fp
        return lambda data: self.write(data, fp)


def run_upgrade(store: Store, log_id: int, cap_bytes: int, fault: dict | None = None):
    """Build fast-mode artifacts for one log. Returns summary dict."""
    entry = store.get(log_id)
    if entry is None:
        raise UpgradeError(f"log {log_id} not found")
    if entry["state"] not in (STATE_BASIC, STATE_FAST):
        raise UpgradeError(f"log {log_id} is already being upgraded")
    if not isinstance(cap_bytes, int) or cap_bytes <= 0:
        raise UpgradeError("cap must be a positive number of bytes")

    # Clean leftovers from a previous failed attempt (idempotent retry).
    store.rollback(log_id)
    store.mark_upgrading(log_id, cap_bytes)
    accounting = ArtifactWriter(cap_bytes, fault)

    try:
        g_tmp, i_tmp = store.gzip_tmp(log_id), store.index_tmp(log_id)
        with open(g_tmp, "wb") as raw_fp:
            index, uncompressed = rebuild_multimember(
                store.original_path(log_id), accounting.bind(raw_fp)
            )
            raw_fp.flush()
            os.fsync(raw_fp.fileno())
        gzip_bytes = accounting.bytes_written

        index_blob = json.dumps(index).encode("utf-8")
        i_tmp2 = i_tmp + ".tmp"
        with open(i_tmp2, "wb") as fp:
            accounting.write(index_blob, fp)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(i_tmp2, i_tmp)
        index_bytes = accounting.bytes_written - gzip_bytes

        store.commit_fast(log_id, accounting.peak)
        return {
            "log_id": log_id,
            "state": STATE_FAST,
            "gzip_bytes": gzip_bytes,
            "index_bytes": index_bytes,
            "peak_bytes": accounting.peak,
            "cap_bytes": cap_bytes,
            "uncompressed_size": uncompressed,
        }
    except BaseException:
        # QuotaExceeded / SimulatedWriteFailure / any IO error: nothing takes effect.
        store.rollback(log_id)
        raise
