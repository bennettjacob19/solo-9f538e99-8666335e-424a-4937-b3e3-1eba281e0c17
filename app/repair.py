"""Tail-repair transaction for logs registered with a truncated size.

Historic builds of the codec stopped decompressing a multi-member gzip when a
member ended exactly at a 64 KiB read boundary, so the registered
``uncompressed_size`` (and, for fast logs, the rebuilt artifacts) missed every
later member. The original gzip was always stored byte-for-byte, so a repair
needs no re-upload: the true size is re-measured from the original file and the
missing tail is brought into service.

Two cases:

* basic log  -> the registered size is corrected with a single atomic
  meta.json replace; the original gzip already serves every byte once the size
  is right, so there is no intermediate state and nothing to roll back.
* fast log   -> the missing tail is rebuilt into *additional* gzip members
  written to a staging file, together with a tail index fragment. On commit the
  staging bytes are appended to the existing fast gzip (its already built
  prefix is never rewritten) and the fragment is published atomically. New disk
  usage therefore grows only with the missing tail and its index. Staging
  files, the appended bytes and the repair bookkeeping file are all counted in
  the reported peak.

At every point of the repair the original gzip stays downloadable and the
previously readable range keeps working: while the state is "repairing" the web
range/content endpoints fall back to full decompression of the original gzip.
A save failure, a simulated write failure or a hard crash before commit leaves
the old state intact and the repair retryable.
"""

from __future__ import annotations

import json
import os

from .gzcodec import (
    GzipError,
    QuotaExceeded,
    SimulatedWriteFailure,
    rebuild_tail_members,
    validate_gzip,
)
from .storage import STATE_BASIC, STATE_FAST, STATE_REPAIRING, Store
from .upgrade import ArtifactWriter, UpgradeError


def run_repair(store: Store, log_id: int, fault: dict | None = None):
    """Repair one log's truncated registered size. Returns a summary dict."""
    entry = store.get(log_id)
    if entry is None:
        raise UpgradeError(f"log {log_id} not found")
    state = entry["state"]
    if state not in (STATE_BASIC, STATE_FAST):
        raise UpgradeError(
            f"log {log_id} is not repairable in state {state}")

    original = store.original_path(log_id)
    true_size = validate_gzip(original)
    old_size = int(entry["uncompressed_size"])
    if true_size == old_size and state == STATE_BASIC:
        return {
            "log_id": log_id,
            "state": state,
            "changed": False,
            "old_uncompressed_size": old_size,
            "uncompressed_size": true_size,
            "tail_bytes": 0,
            "repair_peak_extra_bytes": 0,
        }
    if true_size < old_size:
        # The original cannot be shorter than what the artifacts already cover.
        raise GzipError(
            f"original gzip decompresses to {true_size} bytes but the log "
            f"registers {old_size}; refusing repair")

    if state == STATE_BASIC:
        # Single atomic metadata publish; reads were already correct against the
        # original gzip except for the registered end-of-log cutoff.
        with store.lock():
            cur = store.meta["logs"][str(log_id)]
            cur["uncompressed_size"] = true_size
            cur["repaired"] = True
            store._save_meta()
        return {
            "log_id": log_id,
            "state": STATE_BASIC,
            "changed": True,
            "old_uncompressed_size": old_size,
            "uncompressed_size": true_size,
            "tail_bytes": true_size - old_size,
            "repair_peak_extra_bytes": 0,
        }

    return _repair_fast(store, log_id, old_size, true_size, fault)


def _repair_fast(store: Store, log_id: int, old_size: int, true_size: int,
                 fault: dict | None):
    # Fast log: make sure the committed artifacts really cover the old size, and
    # find where the existing fast gzip ends so the tail can be appended.
    index = store.load_fast_index(log_id)
    covered = index["uncompressed_size"]
    if covered != old_size:
        raise UpgradeError(
            f"log {log_id} index covers {covered} bytes but meta registers "
            f"{old_size}; inconsistent state, refusing repair")
    gzip_len = os.path.getsize(store.gzip_path(log_id))
    last = index["members"][-1]
    if last["coff"] + last["csize"] != gzip_len:
        raise UpgradeError(
            f"log {log_id} fast gzip length {gzip_len} does not match its index")

    if true_size == old_size:
        return {
            "log_id": log_id,
            "state": STATE_FAST,
            "changed": False,
            "old_uncompressed_size": old_size,
            "uncompressed_size": true_size,
            "tail_bytes": 0,
            "repair_peak_extra_bytes": 0,
        }

    # No cap on repairs (the original gzip is not re-created); the writer still
    # enforces fault injection and counts every new byte for the peak.
    accounting = ArtifactWriter(cap_bytes=10**18, fault=fault)
    store.mark_repairing(log_id, old_size, true_size, gzip_len)
    g_tmp = store.repair_gzip_tmp(log_id)
    i_tmp = store.repair_index_tmp(log_id)
    try:
        with open(g_tmp, "wb") as raw_fp:
            tail_members, tail_size, total = rebuild_tail_members(
                store.original_path(log_id), old_size,
                accounting.bind(raw_fp))
            raw_fp.flush()
            os.fsync(raw_fp.fileno())
        if tail_size != true_size - old_size or total != true_size:
            raise GzipError("tail rebuild size mismatch; aborting repair")
        # Shift compressed offsets to be absolute inside the appended gzip.
        for m in tail_members:
            m["coff"] += gzip_len
        gzip_tail_bytes = accounting.bytes_written

        fragment = {
            "version": 1,
            "member_size": index.get("member_size"),
            "uncompressed_size": true_size,
            "members": tail_members,
        }
        blob = json.dumps(fragment, indent=2).encode("utf-8")
        i_pub_tmp = i_tmp + ".tmp"
        with open(i_pub_tmp, "wb") as fp:
            accounting.write(blob, fp)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(i_pub_tmp, i_tmp)
        index_tail_bytes = accounting.bytes_written - gzip_tail_bytes

        store.commit_repair(log_id, true_size, accounting.peak)
        return {
            "log_id": log_id,
            "state": STATE_FAST,
            "changed": True,
            "old_uncompressed_size": old_size,
            "uncompressed_size": true_size,
            "tail_bytes": true_size - old_size,
            "gzip_tail_bytes": gzip_tail_bytes,
            "index_tail_bytes": index_tail_bytes,
            "repair_peak_extra_bytes": accounting.peak,
        }
    except BaseException:
        # Simulated write failure, quota, IO error, crash-detected inconsistency:
        # nothing takes effect, the old fast artifacts and range remain usable.
        store.rollback_repair(log_id)
        raise
