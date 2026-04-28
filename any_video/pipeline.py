"""Processing orchestrator — ties all modules together."""

import logging
import shutil
import tempfile
from pathlib import Path

from any_video.anthropic_client import beautify_transcript, generate_quiz, generate_summary
from any_video.config import OUTPUT_FILES
from any_video.downloader import download_audio, get_video_metadata
from any_video.transcriber import load_model, transcribe

logger = logging.getLogger("any_video")


def find_existing_output(output_dir: Path, video_id: str) -> Path | None:
    """Check if output for this video ID already exists.

    Returns the most recently modified match if multiple exist.
    """
    matches = list(output_dir.glob(f"{video_id}_*"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _is_output_complete(output_path: Path) -> bool:
    """Check if all expected text output files exist."""
    required = ["raw_transcript", "transcript", "summary", "quiz"]
    return all((output_path / OUTPUT_FILES[key]).exists() for key in required)


def process(
    url: str,
    model_name: str,
    output_dir: Path,
    keep_audio: bool,
    force: bool,
) -> Path:
    """Run the full processing pipeline.

    Returns the path to the output directory. Intermediate results are persisted
    so that a failed Claude step can be resumed without re-downloading or re-transcribing.
    """
    # 1. Get metadata (validates URL)
    metadata = get_video_metadata(url)
    logger.info("Video: %s (%s)", metadata.title, metadata.video_id)

    # 2. Check cache
    existing = find_existing_output(output_dir, metadata.video_id)
    if existing and not force:
        if _is_output_complete(existing):
            logger.info("Output already exists: %s", existing)
            return existing
        logger.info("Resuming incomplete output: %s", existing)
        dest = existing
    else:
        if existing and force:
            logger.info("Removing existing output (--force): %s", existing)
            shutil.rmtree(existing)
        dir_name = f"{metadata.video_id}_{metadata.slug_title}"
        dest = output_dir / dir_name
        dest.mkdir(parents=True, exist_ok=True)

    raw_transcript_path = dest / OUTPUT_FILES["raw_transcript"]

    # 3. Download and transcribe (skip if raw transcript already persisted)
    if raw_transcript_path.exists():
        logger.info("Found existing raw transcript, skipping download and transcription")
        raw_transcript = raw_transcript_path.read_text(encoding="utf-8")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            logger.info("[1/5] Downloading audio...")
            audio_path = download_audio(url, tmp_path)

            logger.info("[2/5] Transcribing audio...")
            whisper_model = load_model(model_name)
            raw_transcript = transcribe(whisper_model, audio_path)

            # Persist raw transcript immediately so it survives Claude failures
            raw_transcript_path.write_text(raw_transcript, encoding="utf-8")

            if keep_audio:
                shutil.copy2(audio_path, dest / OUTPUT_FILES["audio"])

    # 4. Claude processing — each result written immediately, skip if already done
    transcript_path = dest / OUTPUT_FILES["transcript"]
    if transcript_path.exists():
        logger.info("[3/5] Beautified transcript already exists, skipping")
        beautified = transcript_path.read_text(encoding="utf-8")
    else:
        logger.info("[3/5] Beautifying transcript...")
        beautified = beautify_transcript(raw_transcript)
        transcript_path.write_text(beautified, encoding="utf-8")

    summary_path = dest / OUTPUT_FILES["summary"]
    if summary_path.exists():
        logger.info("[4/5] Summary already exists, skipping")
    else:
        logger.info("[4/5] Generating summary...")
        summary = generate_summary(beautified)
        summary_path.write_text(summary, encoding="utf-8")

    quiz_path = dest / OUTPUT_FILES["quiz"]
    if quiz_path.exists():
        logger.info("[5/5] Quiz already exists, skipping")
    else:
        logger.info("[5/5] Generating quiz...")
        quiz = generate_quiz(beautified)
        quiz_path.write_text(quiz, encoding="utf-8")

    logger.info("Output written to: %s", dest)
    return dest
