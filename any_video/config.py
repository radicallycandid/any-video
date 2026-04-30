"""Constants, logging setup, shared types and dataclasses."""

import logging
import re
import sys
from dataclasses import dataclass

# --- Constants ---

WHISPER_MODELS = ("tiny", "small", "medium", "large-v3")
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_OUTPUT_DIR = "./output"
CLAUDE_MODEL = "claude-sonnet-4-6"
OUTPUT_FILES = {
    "raw_transcript": "transcript_raw.md",
    "transcript": "transcript.md",
    "summary": "summary.md",
    "gems": "gems.md",
    "quiz": "quiz.md",
    "audio": "audio.mp3",
}
MAX_SLUG_LENGTH = 50


# --- Dataclasses ---


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    title: str
    slug_title: str


# --- Exceptions ---


class AnyVideoError(Exception):
    """Base exception for any-video."""


class DownloadError(AnyVideoError):
    """Error during video download or URL validation."""


class TranscriptionError(AnyVideoError):
    """Error during Whisper transcription."""


class AnthropicError(AnyVideoError):
    """Error during Anthropic API calls."""


# --- Helpers ---


def slugify(title: str) -> str:
    """Convert a title to a URL-safe slug."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:MAX_SLUG_LENGTH]


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    logger = logging.getLogger("any_video")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
