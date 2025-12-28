#!/usr/bin/env python3
"""
Flask server that wraps any_video functionality for the Chrome extension.

Run with: uv run python server.py
The server listens on http://localhost:8765
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from any_video import (
    APIError,
    DownloadError,
    TranscriptionError,
    beautify_transcript,
    download_audio,
    extract_video_id,
    generate_summary_and_quiz,
    get_video_title,
    setup_logging,
    slugify,
    transcribe_audio,
)

app = Flask(__name__)
CORS(app)  # Allow requests from Chrome extension

# Set up logging
setup_logging(verbose=True)

# Store active jobs for progress tracking
jobs: dict[str, dict] = {}


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    return jsonify({
        "status": "ok",
        "openai_configured": has_api_key,
    })


@app.route("/process", methods=["POST"])
def process_video():
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
        # Extract video info
        video_id = extract_video_id(url)
        video_title = get_video_title(url)

        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download audio
            audio_file = download_audio(url, temp_path)

            # Transcribe
            raw_transcript = transcribe_audio(audio_file, model)

            # Beautify transcript
            transcript = beautify_transcript(raw_transcript, video_title)

            # Generate summary and quiz
            summary, quiz = generate_summary_and_quiz(transcript, video_title)

            # Build response
            result = {
                "success": True,
                "video_id": video_id,
                "video_title": video_title,
                "transcript_raw": raw_transcript,
                "transcript": transcript,
                "summary": summary,
                "quiz": quiz,
            }

            # Handle audio file
            if keep_audio:
                # Move audio to a persistent location
                output_dir = Path(__file__).parent / "output"
                output_dir.mkdir(exist_ok=True)
                folder_name = f"{video_id}_{slugify(video_title)}"
                video_output = output_dir / folder_name
                video_output.mkdir(exist_ok=True)

                persistent_audio = video_output / "audio.mp3"
                audio_file.rename(persistent_audio)
                result["audio_path"] = str(persistent_audio)

            return jsonify(result)

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except DownloadError as e:
        return jsonify({"success": False, "error": f"Download failed: {e}"}), 500
    except TranscriptionError as e:
        return jsonify({"success": False, "error": f"Transcription failed: {e}"}), 500
    except APIError as e:
        return jsonify({"success": False, "error": f"API error: {e}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {e}"}), 500


if __name__ == "__main__":
    print("Starting any-video server on http://localhost:8765")
    print("Press Ctrl+C to stop")
    app.run(host="127.0.0.1", port=8765, debug=False)
