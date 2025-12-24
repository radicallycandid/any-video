#!/usr/bin/env python3
"""
any-video: YouTube video transcriber with summaries and quizzes.

Usage:
    python any_video.py <youtube_url> [--model tiny|small|large-v3]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import openai
import whisper


WHISPER_MODELS_DIR = Path.home() / "whisper"
OUTPUT_DIR = Path.home() / "git" / "any-video" / "output"


def extract_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL."""
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
    """Get the video title using yt-dlp."""
    result = subprocess.run(
        ["yt-dlp", "--get-title", url],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")[:50]


def download_audio(url: str, output_path: Path) -> Path:
    """Download audio from YouTube video."""
    audio_file = output_path / "audio.mp3"
    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", str(audio_file),
            url,
        ],
        check=True,
    )
    return audio_file


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    """Transcribe audio using Whisper."""
    model_path = WHISPER_MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        print(f"Model {model_name} not found at {model_path}")
        print("Available models:", list(WHISPER_MODELS_DIR.glob("*.pt")))
        sys.exit(1)

    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name, download_root=str(WHISPER_MODELS_DIR))

    print("Transcribing audio...")
    result = model.transcribe(str(audio_path))
    return result["text"]


def generate_summary_and_quiz(transcript: str, video_title: str) -> tuple[str, str]:
    """Generate summary and quiz using OpenAI API."""
    client = openai.OpenAI()

    # Generate summary
    print("Generating summary...")
    summary_response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Please provide a concise summary of the following video transcript.
The video is titled: "{video_title}"

Transcript:
{transcript}

Write a clear, well-structured summary that captures the main points and key takeaways.""",
            }
        ],
    )
    summary = summary_response.choices[0].message.content

    # Generate quiz
    print("Generating quiz...")
    quiz_response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"""Based on the following video transcript, create a 10-question multiple choice quiz.
The video is titled: "{video_title}"

Transcript:
{transcript}

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


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos and generate summaries with quizzes."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--model",
        choices=["tiny", "small", "large-v3"],
        default="small",
        help="Whisper model to use (default: small)",
    )
    args = parser.parse_args()

    # Check for API key
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
    output_path = OUTPUT_DIR / folder_name
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


if __name__ == "__main__":
    main()
