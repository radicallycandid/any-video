"""Video processing pipeline."""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from any_video.config import logger
from any_video.downloader import download_audio, extract_video_id, get_video_title
from any_video.openai_client import beautify_transcript, generate_summary_and_quiz
from any_video.transcriber import transcribe_audio


@dataclass
class ProcessingResult:
    """Result of processing a video through the full pipeline."""

    video_id: str
    video_title: str
    transcript_raw: str
    transcript: str
    summary: str
    quiz: str
    audio_path: Path | None = None


def process_video(
    url: str,
    model: str = "small",
    work_dir: Path | None = None,
    gpt_model: str | None = None,
    gpt_model_advanced: str | None = None,
    video_id: str | None = None,
    video_title: str | None = None,
    on_progress: Callable[[str, int], None] | None = None,
) -> ProcessingResult:
    """
    Run the full video processing pipeline.

    Downloads audio, transcribes locally with Whisper, beautifies the transcript,
    and generates a summary and quiz using GPT.

    Args:
        url: YouTube video URL.
        model: Whisper model name (tiny, small, large-v3).
        work_dir: Directory for intermediate files. Uses a temp dir if None.
        gpt_model: Override GPT model for beautification.
        gpt_model_advanced: Override GPT model for summary/quiz.
        video_id: Pre-computed video ID (avoids re-extraction).
        video_title: Pre-computed video title (avoids re-fetching).
        on_progress: Optional callback(stage, percent) for progress updates.

    Returns:
        ProcessingResult with all generated content.
    """

    def report(stage: str, pct: int) -> None:
        if on_progress:
            on_progress(stage, pct)

    if video_id is None:
        video_id = extract_video_id(url)
    if video_title is None:
        video_title = get_video_title(url)
    logger.info(f"Video: {video_title}")

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    work_dir.mkdir(parents=True, exist_ok=True)

    report("Downloading audio", 0)
    audio_file = download_audio(url, work_dir)
    report("Transcribing audio", 25)
    raw_transcript = transcribe_audio(audio_file, model)
    report("Beautifying transcript", 50)
    transcript = beautify_transcript(raw_transcript, video_title, model=gpt_model)
    report("Generating summary and quiz", 75)
    summary, quiz = generate_summary_and_quiz(transcript, video_title, model=gpt_model_advanced)
    report("Complete", 100)

    return ProcessingResult(
        video_id=video_id,
        video_title=video_title,
        transcript_raw=raw_transcript,
        transcript=transcript,
        summary=summary,
        quiz=quiz,
        audio_path=audio_file,
    )
