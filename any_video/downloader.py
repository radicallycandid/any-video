"""YouTube URL validation, video ID extraction, metadata fetching, and audio download."""

import logging
import re
from pathlib import Path

import yt_dlp

from any_video.config import DownloadError, VideoMetadata, slugify

logger = logging.getLogger("any_video")

# Patterns for YouTube video IDs (11 characters, alphanumeric + _ -)
_YOUTUBE_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?.*v=)([\w-]{11})"),
    re.compile(r"(?:youtu\.be/)([\w-]{11})"),
    re.compile(r"(?:youtube\.com/embed/)([\w-]{11})"),
]


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL.

    Raises DownloadError if the URL doesn't match any known YouTube format.
    """
    for pattern in _YOUTUBE_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    raise DownloadError(f"Invalid YouTube URL: {url}")


def get_video_metadata(url: str) -> VideoMetadata:
    """Fetch video metadata (title, ID) without downloading."""
    video_id = extract_video_id(url)
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"Failed to fetch video metadata: {e}") from e

    title = info.get("title", "untitled")
    return VideoMetadata(video_id=video_id, title=title, slug_title=slugify(title))


def download_audio(url: str, output_path: Path) -> Path:
    """Download audio from a YouTube URL as MP3.

    Returns the path to the downloaded MP3 file.
    """
    audio_file = output_path / "audio.mp3"
    opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": str(output_path / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    logger.info("Downloading audio...")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"Failed to download audio: {e}") from e

    if not audio_file.exists():
        raise DownloadError("Audio download completed but MP3 file not found")
    return audio_file
