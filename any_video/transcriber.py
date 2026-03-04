"""Audio transcription using OpenAI Whisper."""

from pathlib import Path

import whisper

from any_video.config import WHISPER_MODELS_DIR, logger
from any_video.exceptions import TranscriptionError


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    """
    Transcribe audio using Whisper.

    Uses locally stored Whisper model files from ~/whisper/.

    Args:
        audio_path: Path to the audio file.
        model_name: Name of the Whisper model (tiny, small, large-v3).

    Returns:
        The transcribed text.

    Raises:
        TranscriptionError: If transcription fails.
    """
    # Ensure the whisper models directory exists
    WHISPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model_path = WHISPER_MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            logger.info(
                f"Downloading Whisper model '{model_name}' (this may take a few minutes)..."
            )
        else:
            logger.info(f"Loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name, download_root=str(WHISPER_MODELS_DIR))

        logger.info("Transcribing audio (this may take a while)...")
        result = model.transcribe(str(audio_path), verbose=False)
        logger.info("Transcription complete.")
        return result["text"]
    except (RuntimeError, OSError, ValueError) as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e
