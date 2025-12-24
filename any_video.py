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
import os
import re
import subprocess
import sys
from pathlib import Path

import openai
import whisper

# Configuration
WHISPER_MODELS_DIR = Path.home() / "whisper"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
GPT_MODEL = "gpt-4.1"
MAX_TRANSCRIPT_CHARS = 100000  # Approximate limit to avoid token overflow


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""

    pass


class APIError(Exception):
    """Raised when OpenAI API calls fail."""

    pass


class DownloadError(Exception):
    """Raised when video download fails."""

    pass


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
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", url],
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
    except FileNotFoundError:
        raise DownloadError("yt-dlp not found. Install it with: pip install yt-dlp")


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
    try:
        subprocess.run(
            [
                "yt-dlp",
                "-x",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "0",
                "-o",
                str(audio_file),
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for long videos
        )
    except subprocess.CalledProcessError as e:
        raise DownloadError(f"Failed to download audio: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise DownloadError("Download timed out (exceeded 10 minutes)")
    except FileNotFoundError:
        raise DownloadError("yt-dlp not found. Install it with: pip install yt-dlp")

    if not audio_file.exists():
        raise DownloadError("Audio file was not created")

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
    model_path = WHISPER_MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        available = list(WHISPER_MODELS_DIR.glob("*.pt"))
        available_names = [p.stem for p in available] if available else ["none found"]
        raise TranscriptionError(
            f"Model '{model_name}' not found at {model_path}. "
            f"Available models: {', '.join(available_names)}"
        )

    try:
        print(f"Loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name, download_root=str(WHISPER_MODELS_DIR))

        print("Transcribing audio...")
        result = model.transcribe(str(audio_path))
        return result["text"]
    except Exception as e:
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


def generate_summary_and_quiz(transcript: str, video_title: str) -> tuple[str, str]:
    """
    Generate summary and quiz using OpenAI API.

    Args:
        transcript: The video transcript text.
        video_title: The title of the video.

    Returns:
        A tuple of (summary, quiz) as strings.

    Raises:
        APIError: If the API calls fail.
    """
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise APIError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it with: export OPENAI_API_KEY='your-key-here'"
        )

    # Truncate transcript if needed
    transcript_for_api, was_truncated = truncate_transcript(transcript)
    if was_truncated:
        print(f"Note: Transcript was truncated from {len(transcript):,} to "
              f"{len(transcript_for_api):,} characters to fit API limits.")

    try:
        client = openai.OpenAI()

        # Generate summary
        print("Generating summary...")
        summary_response = client.chat.completions.create(
            model=GPT_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""Please provide a concise summary of the following video transcript.
The video is titled: "{video_title}"

Transcript:
{transcript_for_api}

Write a clear, well-structured summary that captures the main points and key takeaways.""",
                }
            ],
        )
        summary = summary_response.choices[0].message.content

        # Generate quiz
        print("Generating quiz...")
        quiz_response = client.chat.completions.create(
            model=GPT_MODEL,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Based on the following video transcript, create a 10-question multiple choice quiz.
The video is titled: "{video_title}"

Transcript:
{transcript_for_api}

Format each question exactly like this:

## Question 1
What is the main topic discussed in the video?

- A) Option A
- B) Option B
- C) Option C
- D) Option D

**Correct Answer: A**

---

Make sure:
- Questions test comprehension of the key concepts
- All 4 options are plausible
- Include the correct answer after each question
- Separate questions with ---""",
                }
            ],
        )
        quiz = quiz_response.choices[0].message.content

        return summary, quiz

    except openai.AuthenticationError:
        raise APIError("Invalid OpenAI API key. Please check your OPENAI_API_KEY.")
    except openai.RateLimitError:
        raise APIError("OpenAI API rate limit exceeded. Please try again later.")
    except openai.APIConnectionError:
        raise APIError("Failed to connect to OpenAI API. Check your internet connection.")
    except openai.APIError as e:
        raise APIError(f"OpenAI API error: {e}") from e


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
    args = parser.parse_args()

    try:
        # Check for API key early
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set.")
            print("Export it with: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)

        # Extract video info
        print(f"Processing: {args.url}")
        video_id = extract_video_id(args.url)
        video_title = get_video_title(args.url)
        print(f"Video: {video_title}")

        # Create output directory
        folder_name = f"{video_id}_{slugify(video_title)}"
        output_path = args.output_dir / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {output_path}")

        # Download audio
        print("Downloading audio...")
        audio_file = download_audio(args.url, output_path)

        # Transcribe
        transcript = transcribe_audio(audio_file, args.model)

        # Save transcript
        transcript_file = output_path / "transcript.md"
        transcript_file.write_text(f"# Transcript: {video_title}\n\n{transcript}\n")
        print(f"Saved: {transcript_file}")

        # Generate summary and quiz
        summary, quiz = generate_summary_and_quiz(transcript, video_title)

        # Save summary
        summary_file = output_path / "summary.md"
        summary_file.write_text(f"# Summary: {video_title}\n\n{summary}\n")
        print(f"Saved: {summary_file}")

        # Save quiz
        quiz_file = output_path / "quiz.md"
        quiz_file.write_text(f"# Quiz: {video_title}\n\n{quiz}\n")
        print(f"Saved: {quiz_file}")

        # Clean up audio file
        audio_file.unlink()

        print(f"\nDone! Files saved to: {output_path}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except DownloadError as e:
        print(f"Download error: {e}")
        sys.exit(1)
    except TranscriptionError as e:
        print(f"Transcription error: {e}")
        sys.exit(1)
    except APIError as e:
        print(f"API error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
