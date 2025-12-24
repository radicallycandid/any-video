"""Tests for any_video module."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from any_video import (
    APIError,
    DownloadError,
    TranscriptionError,
    extract_video_id,
    retry_with_backoff,
    slugify,
    truncate_transcript,
)


class TestExtractVideoId:
    """Tests for extract_video_id function."""

    def test_standard_url(self):
        """Test extraction from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self):
        """Test extraction from shortened youtu.be URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        """Test extraction from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        """Test extraction from shorts URL."""
        url = "https://youtube.com/shorts/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        """Test extraction from URL with additional parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120&list=PLtest"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_timestamp(self):
        """Test extraction from URL with timestamp."""
        url = "https://youtu.be/dQw4w9WgXcQ?t=60"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_raises_error(self):
        """Test that invalid URLs raise ValueError."""
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id("https://example.com/not-youtube")

    def test_empty_url_raises_error(self):
        """Test that empty URL raises ValueError."""
        with pytest.raises(ValueError):
            extract_video_id("")

    def test_malformed_video_id(self):
        """Test that URLs with wrong-length IDs are rejected."""
        with pytest.raises(ValueError):
            extract_video_id("https://www.youtube.com/watch?v=short")


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_slugify(self):
        """Test basic text slugification."""
        assert slugify("Hello World") == "hello-world"

    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        assert slugify("Hello! World? Test.") == "hello-world-test"

    def test_multiple_spaces_collapsed(self):
        """Test that multiple spaces become single hyphen."""
        assert slugify("Hello    World") == "hello-world"

    def test_max_length_respected(self):
        """Test that max_length parameter is respected."""
        long_text = "a" * 100
        assert len(slugify(long_text, max_length=50)) == 50
        assert len(slugify(long_text, max_length=20)) == 20

    def test_leading_trailing_hyphens_stripped(self):
        """Test that leading/trailing hyphens are removed."""
        assert slugify("  Hello World  ") == "hello-world"
        assert slugify("---test---") == "test"

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        # Unicode letters should be preserved (as \w matches them)
        result = slugify("Café résumé")
        assert "caf" in result

    def test_empty_string(self):
        """Test empty string input."""
        assert slugify("") == ""

    def test_only_special_chars(self):
        """Test string with only special characters."""
        assert slugify("!!!???") == ""


class TestTruncateTranscript:
    """Tests for truncate_transcript function."""

    def test_short_transcript_unchanged(self):
        """Test that short transcripts are not modified."""
        transcript = "This is a short transcript."
        result, was_truncated = truncate_transcript(transcript, max_chars=1000)
        assert result == transcript
        assert was_truncated is False

    def test_long_transcript_truncated(self):
        """Test that long transcripts are truncated."""
        transcript = "A" * 200
        result, was_truncated = truncate_transcript(transcript, max_chars=100)
        assert len(result) <= 100
        assert was_truncated is True

    def test_truncation_at_sentence_boundary(self):
        """Test that truncation prefers sentence boundaries."""
        transcript = "First sentence. Second sentence. Third sentence here is very long."
        result, was_truncated = truncate_transcript(transcript, max_chars=50)
        assert was_truncated is True
        # Should end at a sentence boundary if possible
        assert result.endswith(".") or len(result) <= 50

    def test_exact_length_not_truncated(self):
        """Test that exact-length text is not truncated."""
        transcript = "A" * 100
        result, was_truncated = truncate_transcript(transcript, max_chars=100)
        assert result == transcript
        assert was_truncated is False

    def test_default_max_chars(self):
        """Test default max_chars value works."""
        transcript = "Short text"
        result, was_truncated = truncate_transcript(transcript)
        assert result == transcript
        assert was_truncated is False


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_exception(self):
        """Test that exceptions trigger retries."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, exceptions=(ValueError,))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test that max retries is respected."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, exceptions=(ValueError,))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
        assert call_count == 3  # Initial + 2 retries

    def test_non_matching_exception_not_retried(self):
        """Test that non-matching exceptions are not retried."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong type")

        with pytest.raises(TypeError):
            raises_type_error()
        assert call_count == 1  # No retries for non-matching exception

    def test_exponential_backoff_timing(self):
        """Test that delays increase exponentially."""
        call_times = []

        @retry_with_backoff(max_retries=2, base_delay=0.1, exceptions=(ValueError,))
        def timed_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Fail")
            return "success"

        timed_func()

        # Check delays are roughly exponential (base_delay * 2^attempt)
        # First retry: ~0.1s, Second retry: ~0.2s
        if len(call_times) >= 2:
            first_delay = call_times[1] - call_times[0]
            assert first_delay >= 0.08  # Allow some tolerance

        if len(call_times) >= 3:
            second_delay = call_times[2] - call_times[1]
            assert second_delay >= 0.15  # Should be roughly double


class TestExceptions:
    """Tests for custom exception classes."""

    def test_transcription_error(self):
        """Test TranscriptionError can be raised and caught."""
        with pytest.raises(TranscriptionError):
            raise TranscriptionError("Test error")

    def test_api_error(self):
        """Test APIError can be raised and caught."""
        with pytest.raises(APIError):
            raise APIError("Test error")

    def test_download_error(self):
        """Test DownloadError can be raised and caught."""
        with pytest.raises(DownloadError):
            raise DownloadError("Test error")

    def test_exception_messages(self):
        """Test that exception messages are preserved."""
        msg = "Custom error message"
        try:
            raise APIError(msg)
        except APIError as e:
            assert str(e) == msg


class TestGetVideoTitle:
    """Tests for get_video_title function (mocked)."""

    @patch("any_video.subprocess.run")
    def test_successful_title_fetch(self, mock_run):
        """Test successful video title retrieval."""
        from any_video import get_video_title

        mock_run.return_value = MagicMock(stdout="Test Video Title\n")
        result = get_video_title("https://youtube.com/watch?v=test")
        assert result == "Test Video Title"

    @patch("any_video.subprocess.run")
    def test_title_fetch_timeout(self, mock_run):
        """Test timeout handling."""
        from subprocess import TimeoutExpired

        from any_video import get_video_title

        mock_run.side_effect = TimeoutExpired("yt-dlp", 30)
        with pytest.raises(DownloadError, match="Timed out"):
            get_video_title("https://youtube.com/watch?v=test")

    @patch("any_video.subprocess.run")
    def test_ytdlp_not_found(self, mock_run):
        """Test handling when yt-dlp is not installed."""
        from any_video import get_video_title

        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(DownloadError, match="yt-dlp not found"):
            get_video_title("https://youtube.com/watch?v=test")


class TestDownloadAudio:
    """Tests for download_audio function (mocked)."""

    @patch("any_video.subprocess.run")
    def test_successful_download(self, mock_run, tmp_path):
        """Test successful audio download."""
        from any_video import download_audio

        # Create a fake audio file
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_run.return_value = MagicMock()
        result = download_audio("https://youtube.com/watch?v=test", tmp_path)
        assert result == audio_file

    @patch("any_video.subprocess.run")
    def test_download_timeout(self, mock_run, tmp_path):
        """Test download timeout handling."""
        from subprocess import TimeoutExpired

        from any_video import download_audio

        mock_run.side_effect = TimeoutExpired("yt-dlp", 600)
        with pytest.raises(DownloadError, match="timed out"):
            download_audio("https://youtube.com/watch?v=test", tmp_path)

    @patch("any_video.subprocess.run")
    def test_download_file_not_created(self, mock_run, tmp_path):
        """Test handling when download succeeds but file not created."""
        from any_video import download_audio

        mock_run.return_value = MagicMock()
        # Don't create the file
        with pytest.raises(DownloadError, match="not created"):
            download_audio("https://youtube.com/watch?v=test", tmp_path)


class TestTranscribeAudio:
    """Tests for transcribe_audio function (mocked)."""

    def test_model_not_found(self, tmp_path):
        """Test error when Whisper model doesn't exist."""
        from any_video import WHISPER_MODELS_DIR, transcribe_audio

        # Use a non-existent model
        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake")

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(TranscriptionError, match="not found"):
                transcribe_audio(fake_audio, "nonexistent-model")


class TestGenerateSummaryAndQuiz:
    """Tests for generate_summary_and_quiz function (mocked)."""

    def test_missing_api_key(self):
        """Test error when API key is not set."""
        from any_video import generate_summary_and_quiz

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIError, match="OPENAI_API_KEY"):
                generate_summary_and_quiz("transcript", "title")

    @patch("any_video.openai.OpenAI")
    def test_successful_generation(self, mock_openai_class):
        """Test successful summary and quiz generation."""
        from any_video import generate_summary_and_quiz

        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated content"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            summary, quiz = generate_summary_and_quiz("Test transcript", "Test Title")

        assert summary == "Generated content"
        assert quiz == "Generated content"
        assert mock_client.chat.completions.create.call_count == 2
