"""Flask web app: gzip log archive with byte-offset range reads and fast-mode upgrade.

Endpoints
  GET  /                      UI
  POST /upload                upload a standard gzip -> assigned a log number
  GET  /logs/<id>/download    download the whole original gzip by log number
  GET  /logs/<id>/content     download the full decompressed body (convenience)
  GET  /logs/<id>/range?start=&length=   raw bytes of the decompressed log
  POST /logs/<id>/upgrade     enable fast mode (cap_bytes, optional fault injection)
  GET  /logs/<id>/status      JSON status incl. cap and measured peak

Reads work in every state (basic / upgrading / fast). During an upgrade they fall
back to full decompression of the original gzip, so both read modes stay available.
"""

from __future__ import annotations

import os
import threading

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

from .gzcodec import (
    MAX_DECOMPRESS_PER_RANGE,
    GzipError,
    QuotaExceeded,
    SimulatedWriteFailure,
    read_index,
    read_range_basic,
    read_range_fast,
)
from .storage import Store, STATE_FAST
from .upgrade import UpgradeError, run_upgrade

MAX_RANGE_LENGTH = 64 * 1024


def create_app(data_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    data_dir = data_dir or os.environ.get("DATA_DIR", "/data")
    store = Store(data_dir)
    # Per-log upgrade serialization; reads never block on it.
    upgrade_locks: dict[int, threading.Lock] = {}
    locks_guard = threading.Lock()

    def get_lock(log_id: int) -> threading.Lock:
        with locks_guard:
            return upgrade_locks.setdefault(log_id, threading.Lock())

    def entry_or_404(log_id: int) -> dict:
        entry = store.get(log_id)
        if entry is None:
            abort(404, description=f"log {log_id} not found")
        return entry

    # ----- pages / API -----------------------------------------------------
    @app.get("/")
    def index():
        logs = []
        for log in store.list_logs():
            peak = store.load_peak(log["id"])
            log["peak_bytes"] = max(log.get("peak_bytes", 0), peak["peak_bytes"])
            log["current_artifact_bytes"] = peak["current_bytes"]
            logs.append(log)
        return render_template(
            "index.html",
            logs=logs,
            max_range=MAX_RANGE_LENGTH,
            max_decompress=MAX_DECOMPRESS_PER_RANGE,
        )

    @app.post("/upload")
    def upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            abort(400, description="missing upload file")
        try:
            entry = store.add_upload(f.filename, f.stream)
        except GzipError as exc:
            abort(400, description=f"invalid gzip: {exc}")
        return jsonify(entry), 201

    @app.get("/logs/<int:log_id>/download")
    def download(log_id: int):
        entry = entry_or_404(log_id)
        # The original standard gzip is always retained and downloadable by number.
        return send_file(
            store.original_path(log_id),
            mimetype="application/gzip",
            as_attachment=True,
            download_name=entry["filename"],
        )

    @app.get("/logs/<int:log_id>/content")
    def content(log_id: int):
        entry_or_404(log_id)
        data, decompressed = read_range_basic(
            store.original_path(log_id), 0, entry_size(log_id)
        )
        return Response(
            data,
            headers={
                "X-Decompressed-Bytes": str(decompressed),
                "X-Read-Mode": "basic",
            },
            content_type="application/octet-stream",
        )

    def entry_size(log_id: int) -> int:
        return store.get(log_id)["uncompressed_size"]

    @app.get("/logs/<int:log_id>/range")
    def range_read(log_id: int):
        entry = entry_or_404(log_id)
        try:
            start = int(request.args.get("start", ""))
            length = int(request.args.get("length", ""))
        except (TypeError, ValueError):
            abort(400, description="start and length must be integers")
        if start < 0 or length <= 0:
            abort(400, description="start >= 0 and length > 0 required")
        if length > MAX_RANGE_LENGTH:
            abort(
                400,
                description=f"length exceeds maximum of {MAX_RANGE_LENGTH} bytes "
                            f"(64 KiB)",
            )
        if start >= entry["uncompressed_size"]:
            abort(416, description="start is beyond end of log")

        # Fast only when committed. While upgrading (or in basic mode) the original
        # gzip serves the read; both modes therefore keep working during an upgrade.
        if entry["state"] == STATE_FAST and os.path.exists(store.gzip_path(log_id)):
            try:
                index = read_index(store.index_path(log_id))
                data, decompressed = read_range_fast(
                    store.gzip_path(log_id), index, start, length
                )
                mode = "fast"
            except (GzipError, FileNotFoundError):
                # Corrupted/missing artifacts must not break availability.
                data, decompressed = read_range_basic(
                    store.original_path(log_id), start, length
                )
                mode = "basic-fallback"
        else:
            data, decompressed = read_range_basic(
                store.original_path(log_id), start, length
            )
            mode = "basic"

        return Response(
            data,
            headers={
                "X-Read-Mode": mode,
                "X-Start": str(start),
                "X-Length": str(len(data)),
                "X-Decompressed-Bytes": str(decompressed),
                "X-Max-Decompress-Limit": str(MAX_DECOMPRESS_PER_RANGE),
                "Cache-Control": "no-store",
            },
            content_type="application/octet-stream",
        )

    @app.post("/logs/<int:log_id>/upgrade")
    def upgrade(log_id: int):
        entry_or_404(log_id)
        payload = request.get_json(silent=True) or request.form
        try:
            cap_bytes = int(payload.get("cap_bytes", ""))
        except (TypeError, ValueError):
            abort(400, description="cap_bytes must be an integer")
        if cap_bytes <= 0:
            abort(400, description="cap_bytes must be positive")
        fault = {}
        for key in ("fail_after_bytes", "crash_after_bytes"):
            raw = payload.get(key)
            if raw not in (None, ""):
                try:
                    fault[key] = int(raw)
                except (TypeError, ValueError):
                    abort(400, description=f"{key} must be an integer")

        lock = get_lock(log_id)
        if not lock.acquire(blocking=False):
            abort(409, description="an upgrade for this log is already running")
        try:
            result = run_upgrade(store, log_id, cap_bytes, fault=fault or None)
            return jsonify(result)
        except QuotaExceeded as exc:
            return jsonify({"error": "cap_exceeded", "message": str(exc),
                            "state": store.get(log_id)["state"]}), 507
        except SimulatedWriteFailure as exc:
            return jsonify({"error": "simulated_write_failure", "message": str(exc),
                            "state": store.get(log_id)["state"]}), 500
        except UpgradeError as exc:
            abort(400, description=str(exc))
        finally:
            lock.release()

    @app.get("/logs/<int:log_id>/status")
    def status(log_id: int):
        entry = entry_or_404(log_id)
        info = dict(entry)
        peak = store.load_peak(log_id)
        info["peak_bytes"] = max(entry.get("peak_bytes", 0), peak["peak_bytes"])
        info["current_artifact_bytes"] = store.fast_artifacts_bytes(log_id)
        info["original_gzip_bytes"] = os.path.getsize(store.original_path(log_id))
        return jsonify(info)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(416)
    @app.errorhandler(500)
    def json_errors(err):
        return jsonify({"error": err.name, "message": err.description}), err.code

    # Validate everything still present after a restart (called by tests/CLI).
    app.config["STORE"] = store
    return app


def main():
    app = create_app()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
