#!/usr/bin/env python3
"""
Flask server that wraps any_video functionality for the Chrome extension.

Run with: uv run python server.py
The server listens on http://localhost:8765
"""

import logging
import os
import tempfile
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


@app.route("/health", methods=["GET"])
@limiter.limit("30 per minute")
def health():
    """Health check endpoint."""
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    return jsonify({
        "status": "ok",
        "openai_configured": has_api_key,
    })


@app.route("/process", methods=["POST"])
@limiter.limit("5 per minute")
def process_video_endpoint():
    """
    Process a YouTube video: download, transcribe, beautify, summarize, and generate quiz.

    Request body:
        {
            "url": "https://www.youtube.com/watch?v=...",
            "model": "small",  // optional, defaults to "small"
            "keep_audio": false  // optional, defaults to false
        }

    Returns:
        {
            "success": true,
            "video_id": "...",
            "video_title": "...",
            "transcript_raw": "...",
            "transcript": "...",
            "summary": "...",
            "quiz": "...",
            "audio_path": "..."  // only if keep_audio is true
        }
    """
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"success": False, "error": "Missing 'url' in request body"}), 400

    url = data["url"]
    model = data.get("model", "small")
    keep_audio = data.get("keep_audio", False)

    # Validate model choice
    if model not in ("tiny", "small", "large-v3"):
        return jsonify({
            "success": False,
            "error": f"Invalid model '{model}'. Choose from: tiny, small, large-v3"
        }), 400

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({
            "success": False,
            "error": "OPENAI_API_KEY not set on server"
        }), 500

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = process_video(url, model, work_dir=Path(temp_dir))

            response = {
                "success": True,
                "video_id": result.video_id,
                "video_title": result.video_title,
                "transcript_raw": result.transcript_raw,
                "transcript": result.transcript,
                "summary": result.summary,
                "quiz": result.quiz,
            }

            # Handle audio file
            if keep_audio and result.audio_path:
                # Move audio to a persistent location
                output_dir = Path(__file__).parent / "output"
                folder_name = f"{result.video_id}_{slugify(result.video_title)}"
                video_output = output_dir / folder_name
                video_output.mkdir(parents=True, exist_ok=True)

                persistent_audio = video_output / "audio.mp3"
                result.audio_path.rename(persistent_audio)
                response["audio_path"] = str(persistent_audio)

            return jsonify(response)

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except (DownloadError, TranscriptionError, APIError) as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500


if __name__ == "__main__":
    print("Starting any-video server on http://localhost:8765")
    print("Press Ctrl+C to stop")
    app.run(host="127.0.0.1", port=8765, debug=False)
