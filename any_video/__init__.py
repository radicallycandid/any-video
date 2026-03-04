"""
any-video: YouTube video transcriber with AI-generated summaries and quizzes.

This tool downloads audio from YouTube videos, transcribes them using OpenAI's
Whisper model (running locally), and generates summaries and quizzes using
OpenAI's GPT API.

Usage:
    python -m any_video <youtube_url> [--model tiny|small|large-v3] [--output-dir PATH]

Examples:
    python -m any_video "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python -m any_video "https://youtu.be/dQw4w9WgXcQ" --model large-v3
    python -m any_video "https://youtube.com/shorts/abc123" --output-dir ./my-output
"""

from any_video.cli import main
from any_video.config import (
    BEAUTIFY_CHUNK_SIZE,
    DEFAULT_OUTPUT_DIR,
    GPT_MODEL,
    GPT_MODEL_ADVANCED,
    MAX_RETRIES,
    MAX_TRANSCRIPT_CHARS,
    RETRY_DELAY,
    WHISPER_MODELS_DIR,
    logger,
    retry_with_backoff,
    setup_logging,
)
from any_video.downloader import (
    download_audio,
    extract_video_id,
    get_video_title,
    get_yt_dlp_path,
    slugify,
)
from any_video.exceptions import APIError, DownloadError, TranscriptionError
from any_video.openai_client import (
    BEAUTIFY_SYSTEM_PROMPT,
    _call_openai_api,
    _get_openai_client,
    _raw_openai_call,
    _split_into_chunks,
    beautify_transcript,
    generate_summary_and_quiz,
    truncate_transcript,
)
from any_video.pipeline import ProcessingResult, process_video
from any_video.transcriber import transcribe_audio

__all__ = [
    # Exceptions
    "APIError",
    "DownloadError",
    "TranscriptionError",
    # Config
    "BEAUTIFY_CHUNK_SIZE",
    "DEFAULT_OUTPUT_DIR",
    "GPT_MODEL",
    "GPT_MODEL_ADVANCED",
    "MAX_RETRIES",
    "MAX_TRANSCRIPT_CHARS",
    "RETRY_DELAY",
    "WHISPER_MODELS_DIR",
    "logger",
    "retry_with_backoff",
    "setup_logging",
    # Downloader
    "download_audio",
    "extract_video_id",
    "get_video_title",
    "get_yt_dlp_path",
    "slugify",
    # Transcriber
    "transcribe_audio",
    # OpenAI
    "BEAUTIFY_SYSTEM_PROMPT",
    "_call_openai_api",
    "_get_openai_client",
    "_raw_openai_call",
    "_split_into_chunks",
    "beautify_transcript",
    "generate_summary_and_quiz",
    "truncate_transcript",
    # Pipeline
    "ProcessingResult",
    "process_video",
    # CLI
    "main",
]
