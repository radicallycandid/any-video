"""Tests for downloader module."""

from unittest.mock import MagicMock, patch

import pytest

from any_video.config import DownloadError
from any_video.downloader import download_audio, extract_video_id, get_video_metadata


class TestExtractVideoId:
    def test_standard_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url_with_www(self):
        assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        assert (
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120") == "dQw4w9WgXcQ"
        )

    def test_invalid_url_raises(self):
        with pytest.raises(DownloadError, match="Invalid YouTube URL"):
            extract_video_id("https://example.com/video")

    def test_empty_string_raises(self):
        with pytest.raises(DownloadError):
            extract_video_id("")


class TestGetVideoMetadata:
    @patch("any_video.downloader.yt_dlp.YoutubeDL")
    def test_returns_metadata(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {"title": "Test Video Title"}
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        metadata = get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert metadata.video_id == "dQw4w9WgXcQ"
        assert metadata.title == "Test Video Title"
        assert metadata.slug_title == "test-video-title"

    @patch("any_video.downloader.yt_dlp.YoutubeDL")
    def test_download_error_wraps(self, mock_ydl_class):
        import yt_dlp

        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("not found")
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        with pytest.raises(DownloadError, match="Failed to fetch video metadata"):
            get_video_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


class TestDownloadAudio:
    @patch("any_video.downloader.yt_dlp.YoutubeDL")
    def test_downloads_and_returns_path(self, mock_ydl_class, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        # Simulate yt-dlp creating the file
        audio_file = tmp_path / "audio.mp3"
        mock_ydl.download.side_effect = lambda _: audio_file.write_text("fake audio")

        result = download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)
        assert result == audio_file

    @patch("any_video.downloader.yt_dlp.YoutubeDL")
    def test_raises_when_file_not_created(self, mock_ydl_class, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl_class.return_value = mock_ydl

        with pytest.raises(DownloadError, match="MP3 file not found"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)
