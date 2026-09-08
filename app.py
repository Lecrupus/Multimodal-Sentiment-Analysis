"""Web layer for the multimodal sentiment analyser.

The browser talks to a small JSON API. Short jobs (text, image, audio) are
answered inline; video is queued as a background job the client polls, because
a long clip can easily outlive an HTTP timeout.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import analysis_logic
from config import Config

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("app")

app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# --------------------------------------------------------------------------
# Background job registry
# --------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "progress": 0.0,
            "message": "Queued",
            "result": None,
            "created": time.time(),
        }
    return job_id


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job.update(fields)


def _purge_jobs() -> None:
    cutoff = time.time() - Config.JOB_TTL_SECONDS
    with _jobs_lock:
        for job_id in [k for k, v in _jobs.items() if v["created"] < cutoff]:
            _jobs.pop(job_id, None)


# --------------------------------------------------------------------------
# Upload helpers
# --------------------------------------------------------------------------
def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _save_upload(field: str, allowed: set[str]) -> tuple[str | None, str | None]:
    """Validate and store an upload. Returns ``(path, error)``."""
    if field not in request.files:
        return None, "No file was included in the request."

    file = request.files[field]
    if not file or not file.filename:
        return None, "No file was selected."

    ext = _extension(file.filename)
    if ext not in allowed:
        return None, f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(allowed))}."

    safe = secure_filename(file.filename) or f"upload.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{safe}")
    file.save(path)

    if os.path.getsize(path) == 0:
        os.remove(path)
        return None, "The uploaded file is empty."
    return path, None


def _cleanup(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:  # pragma: no cover
            log.warning("could not delete %s: %s", path, exc)


def _respond(result: dict):
    return jsonify(result), (200 if result.get("ok") else 422)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        max_upload_mb=Config.MAX_CONTENT_LENGTH // (1024 * 1024),
        asr_enabled=Config.ENABLE_ASR,
        ffmpeg=analysis_logic.ffmpeg_available(),
    )


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "ffmpeg": analysis_logic.ffmpeg_available(),
            "asr_enabled": Config.ENABLE_ASR,
            "models_loaded": sorted(analysis_logic._models.keys()),
        }
    )


# --------------------------------------------------------------------------
# Analysis endpoints
# --------------------------------------------------------------------------
@app.post("/api/analyze/text")
def api_text():
    payload = request.get_json(silent=True) or request.form
    return _respond(analysis_logic.analyze_text(payload.get("text", "")))


@app.post("/api/analyze/image")
def api_image():
    path, error = _save_upload("file", Config.ALLOWED_IMAGE)
    if error:
        return jsonify({"ok": False, "error": error, "modality": "image"}), 400
    try:
        return _respond(analysis_logic.analyze_image(path))
    finally:
        _cleanup(path)


@app.post("/api/analyze/audio")
def api_audio():
    path, error = _save_upload("file", Config.ALLOWED_AUDIO)
    if error:
        return jsonify({"ok": False, "error": error, "modality": "audio"}), 400
    try:
        return _respond(analysis_logic.analyze_audio(path))
    finally:
        _cleanup(path)


@app.post("/api/analyze/video")
def api_video():
    """Queue a video job and return its id immediately."""
    path, error = _save_upload("file", Config.ALLOWED_VIDEO)
    if error:
        return jsonify({"ok": False, "error": error, "modality": "video"}), 400

    _purge_jobs()
    job_id = _new_job()

    def run():
        _update_job(job_id, state="running", message="Starting")
        try:
            def progress(pct: float, message: str) -> None:
                _update_job(job_id, progress=round(pct, 3), message=message)

            result = analysis_logic.analyze_video(path, progress=progress)
            _update_job(
                job_id,
                state="done",
                progress=1.0,
                message="Complete",
                result=result,
            )
        except Exception as exc:  # pragma: no cover
            log.exception("video job failed")
            _update_job(
                job_id,
                state="error",
                message=str(exc),
                result={"ok": False, "modality": "video", "error": str(exc)},
            )
        finally:
            _cleanup(path)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        snapshot = dict(job) if job else None
    if not snapshot:
        return jsonify({"ok": False, "error": "Unknown or expired job."}), 404
    return jsonify(snapshot)


@app.post("/api/fuse")
def api_fuse():
    """Combine already-computed modality results into one verdict."""
    payload = request.get_json(silent=True) or {}
    results = payload.get("results") or {}
    if not isinstance(results, dict) or not results:
        return jsonify({"ok": False, "error": "Provide a 'results' object."}), 400
    return _respond(analysis_logic.fuse(results))


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    limit = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({"ok": False, "error": f"File exceeds the {limit} MB limit."}), 413


@app.errorhandler(404)
def not_found(_exc):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "No such endpoint."}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(_exc):  # pragma: no cover
    return jsonify({"ok": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    if os.environ.get("WARMUP", "1") == "1":
        analysis_logic.warmup()
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
