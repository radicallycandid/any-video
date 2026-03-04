"""Video downloading and URL handling."""

import re
import subprocess
import sys
from functools import cache
from pathlib import Path

from any_video.config import logger
from any_video.exceptions import DownloadError


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
    except subprocess.TimeoutExpired as e:
        raise DownloadError("Timed out while fetching video title") from e
    except FileNotFoundError as e:
        raise DownloadError("yt-dlp not found. Install it with: pip install yt-dlp") from e


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
    except subprocess.TimeoutExpired as e:
        raise DownloadError("Download timed out (exceeded 10 minutes)") from e

    if not audio_file.exists():
        raise DownloadError("Audio file was not created")

    logger.debug(f"Downloaded: {audio_file} ({audio_file.stat().st_size / 1024 / 1024:.1f} MB)")
    return audio_file
