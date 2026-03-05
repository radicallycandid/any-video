"""Whisper model loading and transcription."""

import logging
from pathlib import Path

import whisper

from any_video.config import TranscriptionError

logger = logging.getLogger("any_video")


def load_model(model_name: str) -> whisper.Whisper:
    """Load a Whisper model by name."""
    logger.debug("Loading Whisper model '%s'...", model_name)
    try:
        return whisper.load_model(model_name)
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model '{model_name}': {e}") from e


def transcribe(model: whisper.Whisper, audio_path: Path) -> str:
    """Transcribe an audio file using a loaded Whisper model.

    Returns the raw transcript text.
    """
    logger.debug("Transcribing audio...")
    try:
        result = model.transcribe(str(audio_path))
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e
    return result["text"].strip()
