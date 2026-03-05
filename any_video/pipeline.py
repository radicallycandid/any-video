"""Processing orchestrator — ties all modules together."""

import logging
import shutil
import tempfile
from pathlib import Path

from any_video.config import OUTPUT_FILES, ProcessingResult, VideoMetadata
from any_video.downloader import download_audio, get_video_metadata
from any_video.openai_client import beautify_transcript, generate_quiz, generate_summary
from any_video.transcriber import load_model, transcribe

logger = logging.getLogger("any_video")


def find_existing_output(output_dir: Path, video_id: str) -> Path | None:
    """Check if output for this video ID already exists."""
    matches = list(output_dir.glob(f"{video_id}_*"))
    if matches:
        return matches[0]
    return None


def write_output(
    output_dir: Path,
    metadata: VideoMetadata,
    result: ProcessingResult,
    audio_path: Path | None,
    keep_audio: bool,
) -> Path:
    """Write all processing results to the output directory."""
    dir_name = f"{metadata.video_id}_{metadata.slug_title}"
    dest = output_dir / dir_name
    dest.mkdir(parents=True, exist_ok=True)

    (dest / OUTPUT_FILES["raw_transcript"]).write_text(result.raw_transcript, encoding="utf-8")
    (dest / OUTPUT_FILES["transcript"]).write_text(result.beautified_transcript, encoding="utf-8")
    (dest / OUTPUT_FILES["summary"]).write_text(result.summary, encoding="utf-8")
    (dest / OUTPUT_FILES["quiz"]).write_text(result.quiz, encoding="utf-8")

    if keep_audio and audio_path and audio_path.exists():
        shutil.copy2(audio_path, dest / OUTPUT_FILES["audio"])

    return dest


def process(
    url: str,
    model_name: str,
    output_dir: Path,
    keep_audio: bool,
    force: bool,
) -> Path:
    """Run the full processing pipeline.

    Returns the path to the output directory.
    """
    # 1. Get metadata (validates URL)
    metadata = get_video_metadata(url)
    logger.info("Video: %s (%s)", metadata.title, metadata.video_id)

    # 2. Check cache
    existing = find_existing_output(output_dir, metadata.video_id)
    if existing and not force:
        logger.info("Output already exists: %s", existing)
        return existing

    if existing and force:
        logger.info("Removing existing output (--force): %s", existing)
        shutil.rmtree(existing)

    # 3. Download audio to temp dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        audio_path = download_audio(url, tmp_path)

        # 4. Transcribe
        whisper_model = load_model(model_name)
        raw_transcript = transcribe(whisper_model, audio_path)

        # 5. GPT processing
        beautified = beautify_transcript(raw_transcript)
        summary = generate_summary(beautified)
        quiz = generate_quiz(beautified)

        result = ProcessingResult(
            raw_transcript=raw_transcript,
            beautified_transcript=beautified,
            summary=summary,
            quiz=quiz,
        )

        # 6. Write output
        output_path = write_output(output_dir, metadata, result, audio_path, keep_audio)

    logger.info("Output written to: %s", output_path)
    return output_path
