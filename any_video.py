#!/usr/bin/env python3
"""
any-video: YouTube video transcriber with AI-generated summaries and quizzes.

This tool downloads audio from YouTube videos, transcribes them using OpenAI's
Whisper model (running locally), and generates summaries and quizzes using
OpenAI's GPT API.

Usage:
    python any_video.py <youtube_url> [--model tiny|small|large-v3] [--output-dir PATH]

Examples:
    python any_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python any_video.py "https://youtu.be/dQw4w9WgXcQ" --model large-v3
    python any_video.py "https://youtube.com/shorts/abc123" --output-dir ./my-output
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache, wraps
from pathlib import Path
from typing import Callable, TypeVar

import openai
import whisper

# Configuration
WHISPER_MODELS_DIR = Path.home() / "whisper"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
GPT_MODEL = os.environ.get("ANY_VIDEO_GPT_MODEL", "gpt-4.1")
GPT_MODEL_ADVANCED = os.environ.get("ANY_VIDEO_GPT_MODEL_ADVANCED", "gpt-5.2")
MAX_TRANSCRIPT_CHARS = 100_000
BEAUTIFY_CHUNK_SIZE = 50_000

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Base delay in seconds (will be multiplied for exponential backoff)

# Set up logging
logger = logging.getLogger("any-video")

T = TypeVar("T")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    if logger.handlers:
        return
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_DELAY,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries (doubles each attempt).
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed.")
            raise last_exception

        return wrapper

    return decorator


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""

    pass


class APIError(Exception):
    """Raised when OpenAI API calls fail."""

    pass


class DownloadError(Exception):
    """Raised when video download fails."""

    pass


@cache
def get_yt_dlp_path() -> str:
    """
    Find the yt-dlp executable, preferring the one in the same directory as Python.

    This ensures yt-dlp is found when running from a virtual environment,
    even if the venv's bin directory isn't in the system PATH.

    Returns:
        Path to the yt-dlp executable.

    Raises:
        DownloadError: If yt-dlp is not found.
    """
    import shutil

    # First, check in the same directory as the Python interpreter (venv bin)
    python_dir = Path(sys.executable).parent
    venv_yt_dlp = python_dir / "yt-dlp"
    if venv_yt_dlp.exists():
        return str(venv_yt_dlp)

    # Fall back to system PATH
    system_yt_dlp = shutil.which("yt-dlp")
    if system_yt_dlp:
        return system_yt_dlp

    raise DownloadError("yt-dlp not found. Install it with: pip install yt-dlp")


def extract_video_id(url: str) -> str:
    """
    Extract the video ID from a YouTube URL.

    Supports standard URLs, shortened URLs, embeds, and shorts.

    Args:
        url: A YouTube URL in any common format.

    Returns:
        The 11-character video ID.

    Raises:
        ValueError: If the URL format is not recognized.
    """
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_video_title(url: str) -> str:
    """
    Get the video title using yt-dlp.

    Args:
        url: A YouTube URL.

    Returns:
        The video title as a string.

    Raises:
        DownloadError: If yt-dlp fails to fetch the title.
    """
    yt_dlp = get_yt_dlp_path()
    try:
        result = subprocess.run(
            [yt_dlp, "--get-title", url],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise DownloadError(f"Failed to get video title: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise DownloadError("Timed out while fetching video title")


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to a URL-friendly slug.

    Args:
        text: The text to slugify.
        max_length: Maximum length of the resulting slug.

    Returns:
        A lowercase, hyphenated string safe for filenames.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")[:max_length]


def download_audio(url: str, output_path: Path) -> Path:
    """
    Download audio from a YouTube video.

    Args:
        url: A YouTube URL.
        output_path: Directory to save the audio file.

    Returns:
        Path to the downloaded MP3 file.

    Raises:
        DownloadError: If the download fails.
    """
    audio_file = output_path / "audio.mp3"
    logger.info("Downloading audio...")
    logger.debug(f"Output path: {audio_file}")

    yt_dlp = get_yt_dlp_path()
    try:
        subprocess.run(
            [
                yt_dlp,
                "-x",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "0",
                "-o",
                str(audio_file),
                "--progress",
                url,
            ],
            check=True,
            timeout=600,  # 10 minute timeout for long videos
        )
    except subprocess.CalledProcessError as e:
        raise DownloadError(f"Failed to download audio: {e}") from e
    except subprocess.TimeoutExpired:
        raise DownloadError("Download timed out (exceeded 10 minutes)")

    if not audio_file.exists():
        raise DownloadError("Audio file was not created")

    logger.debug(f"Downloaded: {audio_file} ({audio_file.stat().st_size / 1024 / 1024:.1f} MB)")
    return audio_file


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
            logger.info(f"Downloading Whisper model '{model_name}' (this may take a few minutes)...")
        else:
            logger.info(f"Loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name, download_root=str(WHISPER_MODELS_DIR))

        logger.info("Transcribing audio (this may take a while)...")
        result = model.transcribe(str(audio_path), verbose=False)
        logger.info("Transcription complete.")
        return result["text"]
    except (RuntimeError, OSError, ValueError) as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e


def truncate_transcript(transcript: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> tuple[str, bool]:
    """
    Truncate transcript if it exceeds the maximum length.

    Args:
        transcript: The full transcript text.
        max_chars: Maximum number of characters allowed.

    Returns:
        A tuple of (possibly truncated transcript, was_truncated boolean).
    """
    if len(transcript) <= max_chars:
        return transcript, False

    # Try to truncate at a sentence boundary
    truncated = transcript[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.8:  # Only use if we're not losing too much
        truncated = truncated[: last_period + 1]

    return truncated, True


# --- OpenAI API layer ---


def _get_openai_client() -> openai.OpenAI:
    """Create an OpenAI client, validating the API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise APIError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it with: export OPENAI_API_KEY='your-key-here'"
        )
    return openai.OpenAI()


@retry_with_backoff(
    max_retries=MAX_RETRIES,
    base_delay=RETRY_DELAY,
    exceptions=(openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
)
def _raw_openai_call(
    client: openai.OpenAI, messages: list, max_tokens: int, model: str
) -> str:
    """Make a raw OpenAI API call with automatic retry on transient errors."""
    use_new_param = any(model.startswith(p) for p in ("gpt-4.1", "gpt-5", "o1", "o3"))
    token_param = (
        {"max_completion_tokens": max_tokens} if use_new_param else {"max_tokens": max_tokens}
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **token_param,
    )
    return response.choices[0].message.content


def _call_openai_api(
    messages: list, max_tokens: int, model: str | None = None
) -> str:
    """
    Make an OpenAI API call with retry logic and error translation.

    All OpenAI-specific exceptions are caught and re-raised as APIError,
    providing a single place for error handling across the entire module.
    """
    client = _get_openai_client()
    actual_model = model or GPT_MODEL
    try:
        return _raw_openai_call(client, messages, max_tokens, actual_model)
    except openai.AuthenticationError:
        raise APIError("Invalid OpenAI API key. Please check your OPENAI_API_KEY.")
    except openai.RateLimitError:
        raise APIError(
            "OpenAI API rate limit exceeded after multiple retries. Please try again later."
        )
    except openai.APIConnectionError:
        raise APIError(
            "Failed to connect to OpenAI API after multiple retries. Check your internet connection."
        )
    except openai.APITimeoutError:
        raise APIError(
            "OpenAI API request timed out after multiple retries. Please try again later."
        )
    except openai.APIError as e:
        raise APIError(f"OpenAI API error: {e}") from e


# --- Transcript processing ---

BEAUTIFY_SYSTEM_PROMPT = """You are an expert transcript editor. Your job is to clean up raw speech-to-text transcripts while preserving the original meaning exactly.

Your tasks:
1. Fix obvious transcription errors and typos
2. Correct proper nouns (people's names, company names, technical terms) based on context
3. Add appropriate paragraph breaks for readability (every 3-5 sentences or at topic changes)
4. Fix punctuation and capitalization
5. Remove filler words like "um", "uh", "you know" (but keep natural speech patterns)
6. Do NOT add, remove, or change the actual content or meaning
7. Do NOT add summaries, headers, or commentary
8. Do NOT use markdown formatting except for paragraph breaks

Return ONLY the cleaned transcript text, nothing else."""


def _split_into_chunks(text: str, chunk_size: int = BEAUTIFY_CHUNK_SIZE) -> list[str]:
    """Split text into chunks at sentence boundaries for processing."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break
        candidate = remaining[:chunk_size]
        last_period = candidate.rfind(". ")
        split_at = (last_period + 2) if last_period > chunk_size * 0.8 else chunk_size
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return chunks


def beautify_transcript(raw_transcript: str, video_title: str, model: str | None = None) -> str:
    """
    Clean up and format raw Whisper transcript using an LLM.

    Processes long transcripts in chunks to avoid content loss from truncation.

    Args:
        raw_transcript: The raw transcript from Whisper.
        video_title: The video title for context.
        model: Optional model override. Defaults to GPT_MODEL.

    Returns:
        The beautified transcript text.

    Raises:
        APIError: If the API call fails.
    """
    chunks = _split_into_chunks(raw_transcript)
    if len(chunks) > 1:
        logger.info(f"Processing transcript in {len(chunks)} chunks...")

    beautified_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(f"Beautifying chunk {i + 1}/{len(chunks)}...")
        else:
            logger.info("Beautifying transcript...")

        beautified = _call_openai_api(
            [
                {"role": "system", "content": BEAUTIFY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f'Please clean up this transcript from the video "{video_title}":'
                        f"\n\n{chunk}"
                    ),
                },
            ],
            max_tokens=16000,
            model=model or GPT_MODEL,
        )
        beautified_chunks.append(beautified)

    return "\n\n".join(beautified_chunks)


def generate_summary_and_quiz(
    transcript: str, video_title: str, model: str | None = None
) -> tuple[str, str]:
    """
    Generate summary and quiz from a transcript, in parallel.

    Args:
        transcript: The video transcript text.
        video_title: The title of the video.
        model: Optional model override. Defaults to GPT_MODEL_ADVANCED.

    Returns:
        A tuple of (summary, quiz) as strings.

    Raises:
        APIError: If the API calls fail.
    """
    # Truncate transcript if needed
    transcript_for_api, was_truncated = truncate_transcript(transcript)
    if was_truncated:
        logger.warning(
            f"Transcript was truncated from {len(transcript):,} to "
            f"{len(transcript_for_api):,} characters to fit API limits."
        )

    actual_model = model or GPT_MODEL_ADVANCED

    summary_messages = [
        {
            "role": "system",
            "content": """You are an expert content summarizer. Your summaries should be direct and opinionated, capturing the actual claims, arguments, and opinions expressed in the content.

Key guidelines:
- Use direct, assertive language: "X argues that..." or "X states that..." rather than "X reflects on..." or "X discusses..."
- Capture the speaker's actual positions and opinions, not just topics covered
- Include specific claims, arguments, and conclusions made by the speaker
- Present the substance of what was said, not a meta-description of the video
- Be concise but substantive - every sentence should convey meaningful content
- If the speaker expresses strong opinions or makes bold claims, include them directly""",
        },
        {
            "role": "user",
            "content": f"""Summarize the key points, arguments, and opinions from this video transcript.
Video title: "{video_title}"

Transcript:
{transcript_for_api}""",
        },
    ]

    quiz_messages = [
        {
            "role": "system",
            "content": """You are an expert quiz creator. Create high-quality multiple choice questions that genuinely test comprehension.

Critical requirements for answer options:
1. ALL four options (A, B, C, D) must be similar in length, tone, and level of detail
2. Wrong answers must sound equally plausible and professional as the correct answer
3. Avoid making the correct answer longer, more detailed, or more "polished" than wrong answers
4. Distribute correct answers roughly evenly across A, B, C, and D (not clustering on any letter)

Question quality guidelines:
- Target MEDIUM difficulty - not trivially obvious, but answerable if one paid attention
- Questions should require actual comprehension, not just keyword matching
- Wrong answers should be reasonable interpretations that someone might believe if they misunderstood
- Avoid "all of the above" or "none of the above" options
- Each question should test a distinct concept or claim from the content""",
        },
        {
            "role": "user",
            "content": f"""Create a 10-question multiple choice quiz based on this video transcript.
Video title: "{video_title}"

Transcript:
{transcript_for_api}

Format each question exactly like this:

## Question 1
[Question text]

- A) [Option - similar length and tone to others]
- B) [Option - similar length and tone to others]
- C) [Option - similar length and tone to others]
- D) [Option - similar length and tone to others]

**Correct Answer: [Letter]**

---

Remember: Vary which letter is correct across questions, and ensure all options look equally plausible.""",
        },
    ]

    logger.info("Generating summary and quiz...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        summary_future = executor.submit(_call_openai_api, summary_messages, 4000, actual_model)
        quiz_future = executor.submit(_call_openai_api, quiz_messages, 3000, actual_model)
        summary = summary_future.result()
        quiz = quiz_future.result()

    return summary, quiz


# --- Pipeline ---


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

    Returns:
        ProcessingResult with all generated content.
    """
    if video_id is None:
        video_id = extract_video_id(url)
    if video_title is None:
        video_title = get_video_title(url)
    logger.info(f"Video: {video_title}")

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_file = download_audio(url, work_dir)
    raw_transcript = transcribe_audio(audio_file, model)
    transcript = beautify_transcript(raw_transcript, video_title, model=gpt_model)
    summary, quiz = generate_summary_and_quiz(transcript, video_title, model=gpt_model_advanced)

    return ProcessingResult(
        video_id=video_id,
        video_title=video_title,
        transcript_raw=raw_transcript,
        transcript=transcript,
        summary=summary,
        quiz=quiz,
        audio_path=audio_file,
    )


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos and generate summaries with quizzes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"
  %(prog)s "https://youtu.be/VIDEO_ID" --model large-v3
  %(prog)s "https://youtube.com/shorts/VIDEO_ID" --output-dir ./results
        """,
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--model",
        choices=["tiny", "small", "large-v3"],
        default="small",
        help="Whisper model to use (default: small)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with debug information",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the downloaded audio file instead of deleting it",
    )
    parser.add_argument(
        "--gpt-model",
        default=None,
        help=f"GPT model for transcript beautification (default: {GPT_MODEL})",
    )
    parser.add_argument(
        "--gpt-model-advanced",
        default=None,
        help=f"GPT model for summary/quiz generation (default: {GPT_MODEL_ADVANCED})",
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging(verbose=args.verbose)

    try:
        # Check for API key early
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set.")
            logger.error("Export it with: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)

        logger.info(f"Processing: {args.url}")

        # Extract video info for output directory naming
        video_id = extract_video_id(args.url)
        video_title = get_video_title(args.url)
        folder_name = f"{video_id}_{slugify(video_title)}"
        output_path = args.output_dir / folder_name

        # Run the pipeline, passing pre-computed info to avoid duplicate work
        result = process_video(
            args.url,
            args.model,
            work_dir=output_path,
            gpt_model=args.gpt_model,
            gpt_model_advanced=args.gpt_model_advanced,
            video_id=video_id,
            video_title=video_title,
        )

        # Save raw transcript for reference
        raw_transcript_file = output_path / "transcript_raw.md"
        raw_transcript_file.write_text(
            f"# Raw Transcript: {result.video_title}\n\n{result.transcript_raw}\n"
        )
        logger.debug(f"Saved raw transcript: {raw_transcript_file}")

        # Save beautified transcript
        transcript_file = output_path / "transcript.md"
        transcript_file.write_text(
            f"# Transcript: {result.video_title}\n\n{result.transcript}\n"
        )
        logger.info(f"Saved: {transcript_file}")

        # Save summary
        summary_file = output_path / "summary.md"
        summary_file.write_text(f"# Summary: {result.video_title}\n\n{result.summary}\n")
        logger.info(f"Saved: {summary_file}")

        # Save quiz
        quiz_file = output_path / "quiz.md"
        quiz_file.write_text(f"# Quiz: {result.video_title}\n\n{result.quiz}\n")
        logger.info(f"Saved: {quiz_file}")

        # Clean up audio file (unless user wants to keep it)
        if args.keep_audio:
            logger.info(f"Audio file kept: {result.audio_path}")
        elif result.audio_path and result.audio_path.exists():
            result.audio_path.unlink()
            logger.debug("Cleaned up temporary audio file")

        logger.info(f"\nDone! Files saved to: {output_path}")

    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except DownloadError as e:
        logger.error(f"Download error: {e}")
        sys.exit(1)
    except TranscriptionError as e:
        logger.error(f"Transcription error: {e}")
        sys.exit(1)
    except APIError as e:
        logger.error(f"API error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
