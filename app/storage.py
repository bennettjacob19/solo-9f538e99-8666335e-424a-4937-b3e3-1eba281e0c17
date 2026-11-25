"""On-disk storage layout, atomic metadata, crash recovery, peak accounting.

Layout under DATA_DIR (per log id N, the original standard gzip is kept untouched
and is never counted against the fast-mode cap):

  meta.json                         global registry (atomic write)
  logs/1/original.gz                original upload (basic mode data source)
  logs/1/fast/                      fast-mode artifacts (count toward cap)
      gzip.new / index.new          upgrade in progress (visible only after commit)
      gzip.repair                   repair staging: gzip members covering the tail
      index.tail                    repair staging: index fragment for the tail
      repair.json                   repair bookkeeping (prefix gzip length, sizes)
      gzip / index                  committed fast-mode data
      peak.json                     persistent peak / artifact accounting

meta.json states:
  basic          - only original.gz exists; reads use full decompression
  upgrading      - fast build in progress; reads still use original.gz
  fast           - fast artifacts committed; range reads use the index
  repairing      - tail repair in progress on a fast log; reads use original.gz
A crash while "upgrading"/"repairing" is detected at startup (or on any
inconsistent state) and rolled back; the original gzip always remains readable
under the same log number.
"""

from __future__ import annotations

import json
import os
import threading

from .gzcodec import GzipError, read_index, validate_gzip

STATE_BASIC = "basic"
STATE_UPGRADING = "upgrading"
STATE_FAST = "fast"
STATE_REPAIRING = "repairing"


def atomic_write_text(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        fp.write(text)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


class Store:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.logs_dir = os.path.join(data_dir, "logs")
        self.meta_path = os.path.join(data_dir, "meta.json")
        self._lock = threading.RLock()
        os.makedirs(self.logs_dir, exist_ok=True)
        self.meta = self._load_meta()
        self._recover()

    # ----- paths -----------------------------------------------------------
    def log_dir(self, log_id: int) -> str:
        return os.path.join(self.logs_dir, str(log_id))

    def original_path(self, log_id: int) -> str:
        return os.path.join(self.log_dir(log_id), "original.gz")

    def fast_dir(self, log_id: int) -> str:
        return os.path.join(self.log_dir(log_id), "fast")

    def gzip_path(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "gzip")

    def index_path(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "index")

    def gzip_tmp(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "gzip.new")

    def index_tmp(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "index.new")

    # Tail-repair staging paths (repair appends to existing fast artifacts).
    def repair_gzip_tmp(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "gzip.repair")

    def repair_index_tmp(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "index.tail")

    def repair_state_path(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "repair.json")

    def tail_index_path(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "index.tail.committed")

    def peak_path(self, log_id: int) -> str:
        return os.path.join(self.fast_dir(log_id), "peak.json")

    # ----- meta ------------------------------------------------------------
    def _load_meta(self) -> dict:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            meta.setdefault("logs", {})
            return meta
        return {"next_id": 1, "logs": {}}

    def _save_meta(self):
        atomic_write_text(self.meta_path, json.dumps(self.meta, indent=2))

    def list_logs(self):
        with self._lock:
            return [dict(v, id=int(k)) for k, v in sorted(self.meta["logs"].items(),
                                                          key=lambda kv: int(kv[0]))]

    def get(self, log_id: int) -> dict | None:
        with self._lock:
            entry = self.meta["logs"].get(str(log_id))
            return dict(entry, id=log_id) if entry else None

    # ----- upload ----------------------------------------------------------
    def add_upload(self, filename: str, src_stream) -> dict:
        """Stream an upload to original.gz, validating gzip as it lands.

        The registered uncompressed size comes from a full member-aware
        decompression, so a multi-member gzip whose member ends exactly at a
        64 KiB read boundary is measured correctly (new uploads never inherit
        the historic truncated-size bug).
        """
        with self._lock:
            log_id = self.meta["next_id"]
            self.meta["next_id"] = log_id + 1
            log_dir = self.log_dir(log_id)
            os.makedirs(log_dir, exist_ok=True)
            dest = self.original_path(log_id)
            tmp = dest + ".tmp"
            try:
                with open(tmp, "wb") as out:
                    magic = src_stream.read(2)
                    if magic != b"\x1f\x8b":
                        raise GzipError("not a standard gzip file (bad magic)")
                    out.write(magic)
                    while True:
                        chunk = src_stream.read(65536)
                        if not chunk:
                            break
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
                size = validate_gzip(tmp)
                os.replace(tmp, dest)
                entry = {
                    "filename": filename,
                    "state": STATE_BASIC,
                    "uncompressed_size": size,
                    "cap_bytes": None,
                }
                self.meta["logs"][str(log_id)] = entry
                self._save_meta()
                return dict(entry, id=log_id)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                self.meta["next_id"] = log_id
                self._save_meta()
                raise

    # ----- peak accounting -------------------------------------------------
    def load_peak(self, log_id: int) -> dict:
        path = self.peak_path(log_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            data.setdefault("repair_peak_extra_bytes", 0)
            return data
        return {"peak_bytes": 0, "current_bytes": 0,
                "repair_peak_extra_bytes": 0}

    def save_peak(self, log_id: int, peak: int, current: int,
                  repair_extra: int | None = None):
        os.makedirs(self.fast_dir(log_id), exist_ok=True)
        if repair_extra is None:
            repair_extra = self.load_peak(log_id).get(
                "repair_peak_extra_bytes", 0)
        atomic_write_text(self.peak_path(log_id), json.dumps({
            "peak_bytes": int(peak),
            "current_bytes": int(current),
            "repair_peak_extra_bytes": int(repair_extra),
        }, indent=2))

    def fast_artifacts_bytes(self, log_id: int) -> int:
        total = 0
        for name in ("gzip", "index", "peak.json",
                     "gzip.new", "index.new",
                     "gzip.repair", "index.tail", "repair.json",
                     "index.tail.committed"):
            p = os.path.join(self.fast_dir(log_id), name)
            if os.path.exists(p):
                total += os.path.getsize(p)
        return total

    def load_fast_index(self, log_id: int) -> dict:
        """Load the committed index, including any tail-repair fragment.

        The repair appends new members without rewriting the original index:
        the base index and the committed tail fragment are merged here, so the
        extra disk usage of a repair grows only with the missing tail itself.
        """
        index = read_index(self.index_path(log_id))
        tail_path = self.tail_index_path(log_id)
        if os.path.exists(tail_path):
            with open(tail_path, "r", encoding="utf-8") as fp:
                tail = json.load(fp)
            index = dict(index)
            index["members"] = list(index["members"]) + list(tail["members"])
            index["uncompressed_size"] = tail.get(
                "uncompressed_size", index["uncompressed_size"])
        return index

    # ----- state transitions -----------------------------------------------
    def mark_upgrading(self, log_id: int, cap_bytes: int):
        with self._lock:
            entry = self.meta["logs"][str(log_id)]
            entry["state"] = STATE_UPGRADING
            entry["cap_bytes"] = int(cap_bytes)
            self._save_meta()
            os.makedirs(self.fast_dir(log_id), exist_ok=True)

    def commit_fast(self, log_id: int, peak_bytes: int):
        """Publish fast artifacts atomically and switch state to fast."""
        with self._lock:
            g_tmp, i_tmp = self.gzip_tmp(log_id), self.index_tmp(log_id)
            if not (os.path.exists(g_tmp) and os.path.exists(i_tmp)):
                raise RuntimeError("cannot commit: upgrade artifacts missing")
            g_final, i_final = self.gzip_path(log_id), self.index_path(log_id)
            os.replace(g_tmp, g_final)
            os.replace(i_tmp, i_final)
            current = self.fast_artifacts_bytes(log_id)
            self.save_peak(log_id, max(peak_bytes, current), current)
            entry = self.meta["logs"][str(log_id)]
            entry["state"] = STATE_FAST
            entry["peak_bytes"] = max(peak_bytes, current)
            self._save_meta()

    def rollback(self, log_id: int):
        """Discard any unfinished upgrade; the original gzip stays authoritative."""
        with self._lock:
            for p in (self.gzip_tmp(log_id), self.index_tmp(log_id),
                      self.gzip_path(log_id), self.index_path(log_id)):
                if os.path.exists(p):
                    os.remove(p)
            entry = self.meta["logs"].get(str(log_id))
            if entry is None:
                return
            if entry["state"] == STATE_UPGRADING:
                entry["state"] = STATE_BASIC
            self._save_meta()

    # ----- tail repair (historic truncated-size logs) ----------------------
    def mark_repairing(self, log_id: int, old_size: int, true_size: int,
                       gzip_prefix_len: int):
        """Enter the repairing state and persist enough bookkeeping to recover.

        Reads keep flowing from the original gzip while this state is active.
        A repair never rewrites the already-committed gzip prefix; only staging
        files for the missing tail are created.
        """
        with self._lock:
            entry = self.meta["logs"][str(log_id)]
            entry["state"] = STATE_REPAIRING
            self._save_meta()
            os.makedirs(self.fast_dir(log_id), exist_ok=True)
            atomic_write_text(self.repair_state_path(log_id), json.dumps({
                "old_uncompressed_size": int(old_size),
                "true_uncompressed_size": int(true_size),
                "gzip_prefix_len": int(gzip_prefix_len),
            }, indent=2))

    def commit_repair(self, log_id: int, true_size: int,
                      repair_written_peak: int):
        """Publish tail artifacts and the corrected size, fully crash-safe.

        Sequence (every metadata step is an atomic replace; the only append
        happens before any state change, and repair.json keeps the prefix length
        so a crash at any point is resolvable on restart):

          1. append gzip.repair bytes to the committed gzip
          2. atomically publish the tail index fragment
          3. update meta.json size + state to fast
          4. remove staging files
        """
        with self._lock:
            g_tmp = self.repair_gzip_tmp(log_id)
            i_tmp = self.repair_index_tmp(log_id)
            state_path = self.repair_state_path(log_id)
            if not (os.path.exists(g_tmp) and os.path.exists(i_tmp)
                    and os.path.exists(state_path)):
                raise RuntimeError("cannot commit repair: staging missing")
            with open(state_path, "r", encoding="utf-8") as fp:
                rstate = json.load(fp)
            g_final = self.gzip_path(log_id)
            # 1. append the tail members (idempotent under recovery: the prefix
            # length tells us whether the append already happened).
            with open(g_tmp, "rb") as src, open(g_final, "ab") as dst:
                dst.write(src.read())
                dst.flush()
                os.fsync(dst.fileno())
            # 2. publish the tail index fragment atomically.
            os.replace(i_tmp, self.tail_index_path(log_id))
            # 3. correct the registered size and return to fast.
            entry = self.meta["logs"][str(log_id)]
            entry["state"] = STATE_FAST
            entry["uncompressed_size"] = int(true_size)
            entry["repaired"] = True
            self._save_meta()
            # 4. staging cleanup + peak accounting (repair extra peak includes
            # every staging byte and every byte appended to the old fast gzip).
            if os.path.exists(g_tmp):
                os.remove(g_tmp)
            if os.path.exists(state_path):
                os.remove(state_path)
            current = self.fast_artifacts_bytes(log_id)
            prior_peak = self.load_peak(log_id)
            repair_extra = max(
                prior_peak.get("repair_peak_extra_bytes", 0),
                int(repair_written_peak),
            )
            self.save_peak(
                log_id,
                max(prior_peak.get("peak_bytes", 0), current + repair_extra),
                current,
                repair_extra,
            )

    def rollback_repair(self, log_id: int):
        """Undo an unfinished repair; the previously readable range stays intact."""
        with self._lock:
            state_path = self.repair_state_path(log_id)
            rstate = None
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as fp:
                    rstate = json.load(fp)
            g_final = self.gzip_path(log_id)
            # Truncate the gzip back to its pre-repair prefix if the append had
            # already happened before the failure.
            if rstate is not None and os.path.exists(g_final):
                prefix = int(rstate["gzip_prefix_len"])
                if os.path.getsize(g_final) > prefix:
                    with open(g_final, "ab") as fp:
                        fp.truncate(prefix)
                        fp.flush()
                        os.fsync(fp.fileno())
            for p in (self.repair_gzip_tmp(log_id),
                      self.repair_index_tmp(log_id), state_path,
                      self.tail_index_path(log_id)):
                if os.path.exists(p):
                    os.remove(p)
            entry = self.meta["logs"].get(str(log_id))
            if entry is None:
                return
            if entry.get("state") == STATE_REPAIRING:
                entry["state"] = STATE_FAST if os.path.exists(g_final) \
                    else STATE_BASIC
                entry["recovered"] = True
            self._save_meta()

    # ----- crash recovery --------------------------------------------------
    def _recover(self):
        """At startup: resolve any upgrade/repair left unfinished by a crash.

        - complete artifacts present (gzip + index) -> adopt them as committed,
          even if meta still says "upgrading" (crash between rename and meta save)
        - state == upgrading with partial/missing artifacts -> roll back to basic
        - state == fast but artifacts gone -> inconsistent -> back to basic
        - state == repairing -> either adopt a fully committed tail repair or
          roll the repair back; either way the previously readable prefix and
          the original gzip remain available
        A committed fast state with both artifacts present is left untouched.
        """
        for log_id in list(self.meta["logs"]):
            lid = int(log_id)
            entry = self.meta["logs"][log_id]
            state = entry.get("state")
            has_final = os.path.exists(self.gzip_path(lid)) and \
                os.path.exists(self.index_path(lid))
            has_tmp = os.path.exists(self.gzip_tmp(lid)) or \
                os.path.exists(self.index_tmp(lid))
            has_repair_tmp = os.path.exists(self.repair_gzip_tmp(lid)) or \
                os.path.exists(self.repair_index_tmp(lid)) or \
                os.path.exists(self.repair_state_path(lid))

            if state == STATE_REPAIRING:
                self._recover_repair(lid, entry)
                continue

            if has_final and state in (STATE_UPGRADING, STATE_FAST):
                # Complete build: discard stale temp files and adopt the result.
                for p in (self.gzip_tmp(lid), self.index_tmp(lid)):
                    if os.path.exists(p):
                        os.remove(p)
                if state == STATE_UPGRADING:
                    current = self.fast_artifacts_bytes(lid)
                    self.save_peak(lid, max(entry.get("peak_bytes", 0), current),
                                   current)
                    entry["state"] = STATE_FAST
                    entry["peak_bytes"] = max(entry.get("peak_bytes", 0), current)
                    entry["recovered"] = True
                    self._save_meta()
            elif state == STATE_UPGRADING or has_tmp or \
                    (state == STATE_FAST and not has_final):
                self.rollback(lid)
                entry = self.meta["logs"][log_id]
                entry["recovered"] = True
                self._save_meta()
            elif has_repair_tmp:
                # Orphaned repair staging outside an explicit repairing state:
                # drop it; committed artifacts (if any) are authoritative.
                self.rollback_repair(lid)

    def _recover_repair(self, lid: int, entry: dict):
        """Resolve a tail repair interrupted by a hard process exit."""
        state_path = self.repair_state_path(lid)
        if not os.path.exists(state_path):
            # No bookkeeping: nothing was durably started.
            self.rollback_repair(lid)
            return
        with open(state_path, "r", encoding="utf-8") as fp:
            rstate = json.load(fp)
        g_final = self.gzip_path(lid)
        tail_index = self.tail_index_path(lid)
        g_tmp = self.repair_gzip_tmp(lid)
        prefix = int(rstate["gzip_prefix_len"])
        true_size = int(rstate["true_uncompressed_size"])

        # The append is durable exactly when the gzip grew past its prefix. The
        # staging file may already have been removed (crash between cleanup and
        # the meta.json save), so do not rely on its presence.
        gzip_grew = os.path.exists(g_final) and \
            os.path.getsize(g_final) > prefix
        committed = gzip_grew and os.path.exists(tail_index)
        if committed:
            # Append and index both landed; finish the metadata publish. If the
            # staging copy is still around it is now redundant.
            for p in (g_tmp, self.repair_index_tmp(lid), state_path):
                if os.path.exists(p):
                    os.remove(p)
            entry["state"] = STATE_FAST
            entry["uncompressed_size"] = true_size
            entry["repaired"] = True
            entry["recovered"] = True
            self._save_meta()
            current = self.fast_artifacts_bytes(lid)
            prior_peak = self.load_peak(lid)
            self.save_peak(
                lid, max(prior_peak.get("peak_bytes", 0), current), current,
                prior_peak.get("repair_peak_extra_bytes", 0))
            return
        # Anything short of a full commit: undo and keep the old readable range.
        self.rollback_repair(lid)

    # ----- helpers ---------------------------------------------------------
    def lock(self):
        return self._lock
