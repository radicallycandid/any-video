"""Whisper model loading and transcription via faster-whisper (local, CTranslate2)."""

import logging
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

from any_video.config import TranscriptionError

logger = logging.getLogger("any_video")


def load_model(model_name: str) -> WhisperModel:
    """Load a faster-whisper model by name.

    Models are downloaded from HuggingFace on first use and cached locally
    under ~/.cache/huggingface/hub/. Inference is fully local.
    """
    logger.debug("Loading Whisper model '%s'...", model_name)
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as e:
        raise TranscriptionError(f"Failed to load Whisper model '{model_name}': {e}") from e


def transcribe(model: WhisperModel, audio_path: Path) -> str:
    """Transcribe an audio file, showing a progress bar tied to audio duration.

    Returns the raw transcript text.
    """
    logger.debug("Transcribing audio...")
    try:
        segments, info = model.transcribe(str(audio_path))
        text_parts: list[str] = []
        with tqdm(
            total=info.duration,
            unit="s",
            desc="Transcribing",
            bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
        ) as pbar:
            for segment in segments:
                text_parts.append(segment.text)
                pbar.update(segment.end - pbar.n)
        return "".join(text_parts).strip()
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e
