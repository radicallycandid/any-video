#!/usr/bin/env python3
"""
Flask server that wraps any_video functionality for the Chrome extension.

Run with: uv run python server.py
The server listens on http://localhost:8765
"""

import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from any_video import (
    APIError,
    DownloadError,
    TranscriptionError,
    process_video,
    setup_logging,
    slugify,
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(
    app,
    origins=["chrome-extension://*", "http://localhost:*", "http://127.0.0.1:*"],
    methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)
limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])

# Set up logging
setup_logging(verbose=True)

# In-memory store for async job progress and results
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _run_job(request_id: str, url: str, model: str, keep_audio: bool) -> None:
    """Run process_video in a background thread, updating progress."""

    def on_progress(stage: str, pct: int) -> None:
        with _jobs_lock:
            _jobs[request_id]["stage"] = stage
            _jobs[request_id]["progress"] = pct

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = process_video(url, model, work_dir=Path(temp_dir), on_progress=on_progress)

            response = {
                "success": True,
                "video_id": result.video_id,
                "video_title": result.video_title,
                "transcript_raw": result.transcript_raw,
                "transcript": result.transcript,
                "summary": result.summary,
                "quiz": result.quiz,
            }

            if keep_audio and result.audio_path:
                output_dir = Path(__file__).parent / "output"
                folder_name = f"{result.video_id}_{slugify(result.video_title)}"
                video_output = output_dir / folder_name
                video_output.mkdir(parents=True, exist_ok=True)

                persistent_audio = video_output / "audio.mp3"
                result.audio_path.rename(persistent_audio)
                response["audio_path"] = str(persistent_audio)

            with _jobs_lock:
                _jobs[request_id]["result"] = response
                _jobs[request_id]["stage"] = "Complete"
                _jobs[request_id]["progress"] = 100

    except ValueError as e:
        with _jobs_lock:
            _jobs[request_id]["result"] = {"success": False, "error": str(e)}
            _jobs[request_id]["error_code"] = 400
    except (DownloadError, TranscriptionError, APIError) as e:
        with _jobs_lock:
            _jobs[request_id]["result"] = {"success": False, "error": str(e)}
            _jobs[request_id]["error_code"] = 500
    except (OSError, RuntimeError) as e:
        with _jobs_lock:
            _jobs[request_id]["result"] = {
                "success": False,
                "error": f"Unexpected error: {e}",
            }
            _jobs[request_id]["error_code"] = 500


@app.route("/health", methods=["GET"])
@limiter.limit("30 per minute")
def health():
    """Health check endpoint."""
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    return jsonify(
        {
            "status": "ok",
            "openai_configured": has_api_key,
        }
    )


@app.route("/process", methods=["POST"])
@limiter.limit("5 per minute")
def process_video_endpoint():
    """
    Start processing a YouTube video asynchronously.

    Request body:
        {
            "url": "https://www.youtube.com/watch?v=...",
            "model": "small",  // optional, defaults to "small"
            "keep_audio": false  // optional, defaults to false
        }

    Returns:
        {"request_id": "..."} — use GET /progress/<id> and /result/<id> to poll.
    """
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"success": False, "error": "Missing 'url' in request body"}), 400

    url = data["url"]
    model = data.get("model", "small")
    keep_audio = data.get("keep_audio", False)

    if model not in ("tiny", "small", "large-v3"):
        return jsonify(
            {
                "success": False,
                "error": f"Invalid model '{model}'. Choose from: tiny, small, large-v3",
            }
        ), 400

    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({"success": False, "error": "OPENAI_API_KEY not set on server"}), 500

    request_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[request_id] = {"stage": "Starting", "progress": 0, "result": None}

    thread = threading.Thread(
        target=_run_job, args=(request_id, url, model, keep_audio), daemon=True
    )
    thread.start()

    return jsonify({"request_id": request_id})


@app.route("/progress/<request_id>", methods=["GET"])
@limiter.limit("60 per minute")
def get_progress(request_id):
    """Return current progress for a processing job."""
    with _jobs_lock:
        job = _jobs.get(request_id)

    if not job:
        return jsonify({"error": "Unknown request_id"}), 404

    return jsonify({"stage": job["stage"], "progress": job["progress"]})


@app.route("/result/<request_id>", methods=["GET"])
@limiter.limit("30 per minute")
def get_result(request_id):
    """Return the result of a completed processing job."""
    with _jobs_lock:
        job = _jobs.get(request_id)

    if not job:
        return jsonify({"error": "Unknown request_id"}), 404

    if job["result"] is None:
        return jsonify({"status": "pending", "stage": job["stage"], "progress": job["progress"]})

    error_code = job.get("error_code")
    result = job["result"]

    # Clean up completed job
    with _jobs_lock:
        _jobs.pop(request_id, None)

    if error_code:
        return jsonify(result), error_code

    return jsonify(result)


if __name__ == "__main__":
    print("Starting any-video server on http://localhost:8765")
    print("Press Ctrl+C to stop")
    app.run(host="127.0.0.1", port=8765, debug=False)
