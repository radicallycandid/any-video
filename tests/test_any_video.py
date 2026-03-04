"""Tests for any_video module."""

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from any_video import (
    APIError,
    DownloadError,
    ProcessingResult,
    TranscriptionError,
    _split_into_chunks,
    beautify_transcript,
    extract_video_id,
    process_video,
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


class TestSplitIntoChunks:
    """Tests for _split_into_chunks function."""

    def test_short_text_single_chunk(self):
        """Test that short text returns a single chunk."""
        text = "Short text."
        chunks = _split_into_chunks(text, chunk_size=100)
        assert chunks == [text]

    def test_exact_size_single_chunk(self):
        """Test that text at exactly chunk_size returns a single chunk."""
        text = "A" * 100
        chunks = _split_into_chunks(text, chunk_size=100)
        assert chunks == [text]

    def test_splits_at_sentence_boundary(self):
        """Test that chunks split at sentence boundaries when possible."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = _split_into_chunks(text, chunk_size=35)
        assert len(chunks) >= 2
        # First chunk should end at a sentence boundary
        assert chunks[0].rstrip().endswith(".")

    def test_all_content_preserved(self):
        """Test that joining chunks reproduces the original text."""
        text = "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50
        chunks = _split_into_chunks(text, chunk_size=60)
        assert len(chunks) >= 2
        rejoined = "".join(chunks)
        assert rejoined == text

    def test_empty_string(self):
        """Test that empty string returns single empty chunk."""
        chunks = _split_into_chunks("", chunk_size=100)
        assert chunks == [""]

    def test_splits_at_word_boundary_when_no_sentence_boundary(self):
        """Test fallback to word boundary when no period exists."""
        text = "word " * 20  # 100 chars of words with spaces but no periods
        chunks = _split_into_chunks(text.strip(), chunk_size=40)
        assert len(chunks) >= 2
        # First chunk should end at a word boundary (no trailing partial word)
        assert not chunks[0].endswith("wor")
        # All content preserved
        assert "".join(chunks) == text.strip()

    def test_hard_cuts_when_no_spaces(self):
        """Test hard cut when text has no whitespace at all."""
        text = "a" * 100
        chunks = _split_into_chunks(text, chunk_size=40)
        assert len(chunks) == 3
        assert chunks[0] == "a" * 40
        assert chunks[1] == "a" * 40
        assert chunks[2] == "a" * 20
        assert "".join(chunks) == text


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_creation(self):
        """Test basic dataclass creation."""
        result = ProcessingResult(
            video_id="abc123",
            video_title="Test Video",
            transcript_raw="raw text",
            transcript="clean text",
            summary="summary text",
            quiz="quiz text",
        )
        assert result.video_id == "abc123"
        assert result.audio_path is None

    def test_with_audio_path(self):
        """Test creation with audio_path."""
        result = ProcessingResult(
            video_id="abc123",
            video_title="Test",
            transcript_raw="raw",
            transcript="clean",
            summary="summary",
            quiz="quiz",
            audio_path=Path("/tmp/audio.mp3"),
        )
        assert result.audio_path == Path("/tmp/audio.mp3")


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

    @patch("any_video.downloader.subprocess.run")
    def test_successful_title_fetch(self, mock_run):
        """Test successful video title retrieval."""
        from any_video.downloader import get_video_title

        mock_run.return_value = MagicMock(stdout="Test Video Title\n")
        result = get_video_title("https://youtube.com/watch?v=test")
        assert result == "Test Video Title"

    @patch("any_video.downloader.subprocess.run")
    def test_title_fetch_timeout(self, mock_run):
        """Test timeout handling."""
        from subprocess import TimeoutExpired

        from any_video.downloader import get_video_title

        mock_run.side_effect = TimeoutExpired("yt-dlp", 30)
        with pytest.raises(DownloadError, match="Timed out"):
            get_video_title("https://youtube.com/watch?v=test")

    @patch("any_video.downloader.get_yt_dlp_path")
    def test_ytdlp_not_found(self, mock_get_path):
        """Test handling when yt-dlp is not installed."""
        from any_video.downloader import get_video_title

        mock_get_path.side_effect = DownloadError(
            "yt-dlp not found. Install it with: pip install yt-dlp"
        )
        with pytest.raises(DownloadError, match="yt-dlp not found"):
            get_video_title("https://youtube.com/watch?v=test")


class TestDownloadAudio:
    """Tests for download_audio function (mocked)."""

    @patch("any_video.downloader.subprocess.run")
    def test_successful_download(self, mock_run, tmp_path):
        """Test successful audio download."""
        from any_video.downloader import download_audio

        # Create a fake audio file
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_run.return_value = MagicMock()
        result = download_audio("https://youtube.com/watch?v=test", tmp_path)
        assert result == audio_file

    @patch("any_video.downloader.subprocess.run")
    def test_download_timeout(self, mock_run, tmp_path):
        """Test download timeout handling."""
        from subprocess import TimeoutExpired

        from any_video.downloader import download_audio

        mock_run.side_effect = TimeoutExpired("yt-dlp", 600)
        with pytest.raises(DownloadError, match="timed out"):
            download_audio("https://youtube.com/watch?v=test", tmp_path)

    @patch("any_video.downloader.subprocess.run")
    def test_download_file_not_created(self, mock_run, tmp_path):
        """Test handling when download succeeds but file not created."""
        from any_video.downloader import download_audio

        mock_run.return_value = MagicMock()
        # Don't create the file
        with pytest.raises(DownloadError, match="not created"):
            download_audio("https://youtube.com/watch?v=test", tmp_path)


class TestTranscribeAudio:
    """Tests for transcribe_audio function (mocked)."""

    def test_transcription_failure(self, tmp_path):
        """Test error when Whisper transcription fails."""
        from any_video.transcriber import transcribe_audio

        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake")

        with patch("any_video.transcriber.whisper.load_model") as mock_load:
            mock_load.side_effect = RuntimeError("Model loading failed")
            with pytest.raises(TranscriptionError, match="Transcription failed"):
                transcribe_audio(fake_audio, "small")

    def test_successful_transcription(self, tmp_path):
        """Test successful audio transcription returns text."""
        from any_video.transcriber import transcribe_audio

        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world"}

        with patch("any_video.transcriber.whisper.load_model", return_value=mock_model):
            result = transcribe_audio(fake_audio, "small")

        assert result == "hello world"
        mock_model.transcribe.assert_called_once_with(str(fake_audio), verbose=False)


class TestGenerateSummaryAndQuiz:
    """Tests for generate_summary_and_quiz function (mocked)."""

    def test_missing_api_key(self):
        """Test error when API key is not set."""
        from any_video.openai_client import generate_summary_and_quiz

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(APIError, match="OPENAI_API_KEY"),
        ):
            generate_summary_and_quiz("transcript", "title")

    @patch("any_video.openai_client.openai.OpenAI")
    def test_successful_generation(self, mock_openai_class):
        """Test successful summary and quiz generation."""
        from any_video.openai_client import generate_summary_and_quiz

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


class TestBeautifyTranscript:
    """Tests for beautify_transcript function (mocked)."""

    def test_missing_api_key(self):
        """Test error when API key is not set."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(APIError, match="OPENAI_API_KEY"),
        ):
            beautify_transcript("raw transcript", "title")

    @patch("any_video.openai_client.openai.OpenAI")
    def test_successful_beautification(self, mock_openai_class):
        """Test successful transcript beautification."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Beautified transcript"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = beautify_transcript("raw um transcript with uh errors", "Test Title")

        assert result == "Beautified transcript"
        assert mock_client.chat.completions.create.call_count == 1

    @patch("any_video.openai_client.openai.OpenAI")
    def test_beautification_uses_system_prompt(self, mock_openai_class):
        """Test that beautification uses a system prompt for instructions."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Clean text"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            beautify_transcript("raw text", "Title")

        # Check that a system message was included
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "transcript editor" in messages[0]["content"].lower()

    @patch("any_video.openai_client.openai.OpenAI")
    def test_multi_chunk_beautification(self, mock_openai_class):
        """Test that long transcripts are split into chunks and results joined."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Return different content for each chunk call
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="Chunk 1 result"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Chunk 2 result"))]),
        ]
        mock_client.chat.completions.create.side_effect = responses

        # Create text that exceeds chunk_size to force splitting
        long_text = "word " * 20000  # ~100K chars

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = beautify_transcript(long_text.strip(), "Title", model="gpt-4.1")

        assert mock_client.chat.completions.create.call_count == 2
        assert result == "Chunk 1 result\n\nChunk 2 result"


class TestProcessVideo:
    """Tests for the process_video pipeline function."""

    @patch("any_video.pipeline.generate_summary_and_quiz")
    @patch("any_video.pipeline.beautify_transcript")
    @patch("any_video.pipeline.transcribe_audio")
    @patch("any_video.pipeline.download_audio")
    @patch("any_video.pipeline.get_video_title")
    @patch("any_video.pipeline.extract_video_id")
    def test_full_pipeline(
        self,
        mock_extract,
        mock_title,
        mock_download,
        mock_transcribe,
        mock_beautify,
        mock_summary_quiz,
        tmp_path,
    ):
        """Test that process_video orchestrates all steps and returns ProcessingResult."""
        mock_extract.return_value = "abc123"
        mock_title.return_value = "Test Video"
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "raw transcript"
        mock_beautify.return_value = "clean transcript"
        mock_summary_quiz.return_value = ("the summary", "the quiz")

        result = process_video("https://youtube.com/watch?v=abc123", work_dir=tmp_path)

        assert isinstance(result, ProcessingResult)
        assert result.video_id == "abc123"
        assert result.video_title == "Test Video"
        assert result.transcript_raw == "raw transcript"
        assert result.transcript == "clean transcript"
        assert result.summary == "the summary"
        assert result.quiz == "the quiz"
        assert result.audio_path == tmp_path / "audio.mp3"

        mock_download.assert_called_once_with("https://youtube.com/watch?v=abc123", tmp_path)
        mock_transcribe.assert_called_once_with(tmp_path / "audio.mp3", "small")
        mock_beautify.assert_called_once_with("raw transcript", "Test Video", model=None)
        mock_summary_quiz.assert_called_once_with("clean transcript", "Test Video", model=None)

    @patch("any_video.pipeline.generate_summary_and_quiz")
    @patch("any_video.pipeline.beautify_transcript")
    @patch("any_video.pipeline.transcribe_audio")
    @patch("any_video.pipeline.download_audio")
    def test_uses_precomputed_video_info(
        self, mock_download, mock_transcribe, mock_beautify, mock_summary_quiz, tmp_path
    ):
        """Test that pre-computed video_id and video_title skip extraction."""
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "raw"
        mock_beautify.return_value = "clean"
        mock_summary_quiz.return_value = ("s", "q")

        result = process_video(
            "https://youtube.com/watch?v=abc123",
            work_dir=tmp_path,
            video_id="pre_id",
            video_title="Pre Title",
        )

        assert result.video_id == "pre_id"
        assert result.video_title == "Pre Title"

    @patch("any_video.pipeline.generate_summary_and_quiz")
    @patch("any_video.pipeline.beautify_transcript")
    @patch("any_video.pipeline.transcribe_audio")
    @patch("any_video.pipeline.download_audio")
    @patch("any_video.pipeline.get_video_title")
    @patch("any_video.pipeline.extract_video_id")
    def test_passes_gpt_model_overrides(
        self,
        mock_extract,
        mock_title,
        mock_download,
        mock_transcribe,
        mock_beautify,
        mock_summary_quiz,
        tmp_path,
    ):
        """Test that gpt_model and gpt_model_advanced are forwarded."""
        mock_extract.return_value = "id"
        mock_title.return_value = "title"
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "raw"
        mock_beautify.return_value = "clean"
        mock_summary_quiz.return_value = ("s", "q")

        process_video(
            "https://youtube.com/watch?v=id",
            work_dir=tmp_path,
            gpt_model="gpt-4o",
            gpt_model_advanced="o3",
        )

        mock_beautify.assert_called_once_with("raw", "title", model="gpt-4o")
        mock_summary_quiz.assert_called_once_with("clean", "title", model="o3")

    @patch("any_video.pipeline.generate_summary_and_quiz")
    @patch("any_video.pipeline.beautify_transcript")
    @patch("any_video.pipeline.transcribe_audio")
    @patch("any_video.pipeline.download_audio")
    @patch("any_video.pipeline.get_video_title")
    @patch("any_video.pipeline.extract_video_id")
    def test_creates_work_dir_when_none(
        self,
        mock_extract,
        mock_title,
        mock_download,
        mock_transcribe,
        mock_beautify,
        mock_summary_quiz,
        tmp_path,
    ):
        """Test that a temp dir is created when work_dir is None."""
        mock_extract.return_value = "id"
        mock_title.return_value = "title"
        mock_download.return_value = tmp_path / "audio.mp3"
        mock_transcribe.return_value = "raw"
        mock_beautify.return_value = "clean"
        mock_summary_quiz.return_value = ("s", "q")

        result = process_video("https://youtube.com/watch?v=id")

        assert result.video_id == "id"
        # download_audio was called with some directory (the auto-created temp dir)
        call_args = mock_download.call_args
        work_dir_used = call_args[0][1]
        assert work_dir_used.exists()

    @patch("any_video.pipeline.download_audio")
    @patch("any_video.pipeline.get_video_title")
    @patch("any_video.pipeline.extract_video_id")
    def test_propagates_download_error(self, mock_extract, mock_title, mock_download, tmp_path):
        """Test that DownloadError propagates from the pipeline."""
        mock_extract.return_value = "id"
        mock_title.return_value = "title"
        mock_download.side_effect = DownloadError("download failed")

        with pytest.raises(DownloadError, match="download failed"):
            process_video("https://youtube.com/watch?v=id", work_dir=tmp_path)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def setup_method(self):
        """Clear logger handlers before each test."""
        logger = logging.getLogger("any-video")
        logger.handlers.clear()

    def test_sets_info_level_by_default(self):
        """Test that default logging level is INFO."""
        from any_video.config import setup_logging

        setup_logging()
        logger = logging.getLogger("any-video")
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1

    def test_sets_debug_level_when_verbose(self):
        """Test that verbose=True sets DEBUG level."""
        from any_video.config import setup_logging

        setup_logging(verbose=True)
        logger = logging.getLogger("any-video")
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers(self):
        """Test that repeated calls don't add duplicate handlers."""
        from any_video.config import setup_logging

        setup_logging()
        setup_logging()
        setup_logging()
        logger = logging.getLogger("any-video")
        assert len(logger.handlers) == 1


class TestGetYtDlpPath:
    """Tests for get_yt_dlp_path function."""

    def setup_method(self):
        """Clear the cache before each test."""
        from any_video.downloader import get_yt_dlp_path

        get_yt_dlp_path.cache_clear()

    @patch("any_video.downloader.sys")
    def test_finds_in_venv_bin(self, mock_sys, tmp_path):
        """Test finding yt-dlp in the Python interpreter's directory."""
        from any_video.downloader import get_yt_dlp_path

        fake_yt_dlp = tmp_path / "yt-dlp"
        fake_yt_dlp.touch()
        mock_sys.executable = str(tmp_path / "python")

        result = get_yt_dlp_path()
        assert result == str(fake_yt_dlp)

    @patch("shutil.which", return_value="/usr/local/bin/yt-dlp")
    @patch("any_video.downloader.sys")
    def test_falls_back_to_system_path(self, mock_sys, mock_which, tmp_path):
        """Test falling back to PATH when not in venv bin."""
        from any_video.downloader import get_yt_dlp_path

        mock_sys.executable = str(tmp_path / "python")

        result = get_yt_dlp_path()
        assert result == "/usr/local/bin/yt-dlp"

    @patch("shutil.which", return_value=None)
    @patch("any_video.downloader.sys")
    def test_raises_when_not_found(self, mock_sys, mock_which, tmp_path):
        """Test DownloadError when yt-dlp is not found anywhere."""
        from any_video.downloader import get_yt_dlp_path

        mock_sys.executable = str(tmp_path / "python")

        with pytest.raises(DownloadError, match="yt-dlp not found"):
            get_yt_dlp_path()


class TestMain:
    """Tests for the main CLI entry point."""

    @patch("any_video.cli.process_video")
    @patch("any_video.cli.get_video_title")
    @patch("any_video.cli.extract_video_id")
    def test_successful_run(self, mock_extract, mock_title, mock_process, tmp_path):
        """Test successful CLI execution creates output files."""
        from any_video.cli import main

        mock_extract.return_value = "abc123"
        mock_title.return_value = "Test Video"

        output_dir = tmp_path / "output"

        def mock_process_fn(url, model, work_dir=None, **kwargs):
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True)
            return ProcessingResult(
                video_id="abc123",
                video_title="Test Video",
                transcript_raw="raw text",
                transcript="clean text",
                summary="summary text",
                quiz="quiz text",
                audio_path=work_dir / "audio.mp3" if work_dir else None,
            )

        mock_process.side_effect = mock_process_fn

        with (
            patch(
                "sys.argv",
                [
                    "any-video",
                    "https://youtube.com/watch?v=abc123",
                    "--output-dir",
                    str(output_dir),
                ],
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            main()

        expected_dir = output_dir / "abc123_test-video"
        assert (expected_dir / "transcript_raw.md").exists()
        assert (expected_dir / "transcript.md").exists()
        assert (expected_dir / "summary.md").exists()
        assert (expected_dir / "quiz.md").exists()

        # Verify content format
        content = (expected_dir / "summary.md").read_text()
        assert content.startswith("# Summary: Test Video")
        assert "summary text" in content

    def test_missing_api_key_exits(self):
        """Test that missing API key causes exit with code 1."""
        from any_video.cli import main

        with (
            patch("sys.argv", ["any-video", "https://youtube.com/watch?v=abc123"]),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    @patch("any_video.cli.extract_video_id")
    def test_invalid_url_exits(self, mock_extract):
        """Test that invalid URL causes exit with code 1."""
        from any_video.cli import main

        mock_extract.side_effect = ValueError("Could not extract video ID")
        with (
            patch("sys.argv", ["any-video", "not-a-url"]),
            patch.dict("os.environ", {"OPENAI_API_KEY": "key"}),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    @patch("any_video.cli.get_video_title")
    @patch("any_video.cli.extract_video_id")
    def test_download_error_exits(self, mock_extract, mock_title):
        """Test that DownloadError causes exit with code 1."""
        from any_video.cli import main

        mock_extract.return_value = "abc123"
        mock_title.side_effect = DownloadError("Video not found")
        with (
            patch("sys.argv", ["any-video", "https://youtube.com/watch?v=abc123"]),
            patch.dict("os.environ", {"OPENAI_API_KEY": "key"}),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    @patch("any_video.cli.process_video")
    @patch("any_video.cli.get_video_title")
    @patch("any_video.cli.extract_video_id")
    def test_api_error_exits(self, mock_extract, mock_title, mock_process):
        """Test that APIError causes exit with code 1."""
        from any_video.cli import main

        mock_extract.return_value = "abc123"
        mock_title.return_value = "Test"
        mock_process.side_effect = APIError("Rate limited")
        with (
            patch(
                "sys.argv",
                [
                    "any-video",
                    "https://youtube.com/watch?v=abc123",
                    "--output-dir",
                    "/tmp/test-output",
                ],
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "key"}),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    @patch("any_video.cli.process_video")
    @patch("any_video.cli.get_video_title")
    @patch("any_video.cli.extract_video_id")
    def test_keep_audio_preserves_file(self, mock_extract, mock_title, mock_process, tmp_path):
        """Test that --keep-audio prevents audio file deletion."""
        from any_video.cli import main

        mock_extract.return_value = "abc123"
        mock_title.return_value = "Test Video"

        output_dir = tmp_path / "output"

        def mock_process_fn(url, model, work_dir=None, **kwargs):
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True)
            audio_path = work_dir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            return ProcessingResult(
                video_id="abc123",
                video_title="Test Video",
                transcript_raw="raw",
                transcript="clean",
                summary="summary",
                quiz="quiz",
                audio_path=audio_path,
            )

        mock_process.side_effect = mock_process_fn

        with (
            patch(
                "sys.argv",
                [
                    "any-video",
                    "https://youtube.com/watch?v=abc123",
                    "--output-dir",
                    str(output_dir),
                    "--keep-audio",
                ],
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            main()

        # Audio file should still exist
        expected_dir = output_dir / "abc123_test-video"
        assert (expected_dir / "audio.mp3").exists()

    @patch("any_video.cli.process_video")
    @patch("any_video.cli.get_video_title")
    @patch("any_video.cli.extract_video_id")
    def test_audio_cleaned_up_by_default(self, mock_extract, mock_title, mock_process, tmp_path):
        """Test that audio file is deleted by default."""
        from any_video.cli import main

        mock_extract.return_value = "abc123"
        mock_title.return_value = "Test Video"

        output_dir = tmp_path / "output"

        def mock_process_fn(url, model, work_dir=None, **kwargs):
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True)
            audio_path = work_dir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            return ProcessingResult(
                video_id="abc123",
                video_title="Test Video",
                transcript_raw="raw",
                transcript="clean",
                summary="summary",
                quiz="quiz",
                audio_path=audio_path,
            )

        mock_process.side_effect = mock_process_fn

        with (
            patch(
                "sys.argv",
                [
                    "any-video",
                    "https://youtube.com/watch?v=abc123",
                    "--output-dir",
                    str(output_dir),
                ],
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            main()

        # Audio file should be cleaned up
        expected_dir = output_dir / "abc123_test-video"
        assert not (expected_dir / "audio.mp3").exists()

    @patch("any_video.cli.process_video")
    @patch("any_video.cli.get_video_title")
    @patch("any_video.cli.extract_video_id")
    def test_keyboard_interrupt_exits_130(self, mock_extract, mock_title, mock_process):
        """Test that KeyboardInterrupt causes exit with code 130."""
        from any_video.cli import main

        mock_extract.side_effect = KeyboardInterrupt()
        with (
            patch("sys.argv", ["any-video", "https://youtube.com/watch?v=abc123"]),
            patch.dict("os.environ", {"OPENAI_API_KEY": "key"}),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 130

    @patch("any_video.cli.process_video")
    @patch("any_video.cli.get_video_title")
    @patch("any_video.cli.extract_video_id")
    def test_skips_processing_when_output_exists(
        self, mock_extract, mock_title, mock_process, tmp_path
    ):
        """Test that existing output causes early exit without re-processing."""
        from any_video.cli import main

        mock_extract.return_value = "abc123"
        mock_title.return_value = "Test Video"

        # Pre-create output files
        output_dir = tmp_path / "output"
        expected_dir = output_dir / "abc123_test-video"
        expected_dir.mkdir(parents=True)
        for f in ["transcript_raw.md", "transcript.md", "summary.md", "quiz.md"]:
            (expected_dir / f).write_text("existing content")

        with (
            patch(
                "sys.argv",
                [
                    "any-video",
                    "https://youtube.com/watch?v=abc123",
                    "--output-dir",
                    str(output_dir),
                ],
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
        mock_process.assert_not_called()

    @patch("any_video.cli.process_video")
    @patch("any_video.cli.get_video_title")
    @patch("any_video.cli.extract_video_id")
    def test_force_flag_reprocesses(self, mock_extract, mock_title, mock_process, tmp_path):
        """Test that --force re-processes even when output exists."""
        from any_video.cli import main

        mock_extract.return_value = "abc123"
        mock_title.return_value = "Test Video"

        output_dir = tmp_path / "output"
        expected_dir = output_dir / "abc123_test-video"
        expected_dir.mkdir(parents=True)
        for f in ["transcript_raw.md", "transcript.md", "summary.md", "quiz.md"]:
            (expected_dir / f).write_text("existing content")

        def mock_process_fn(url, model, work_dir=None, **kwargs):
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True)
            return ProcessingResult(
                video_id="abc123",
                video_title="Test Video",
                transcript_raw="new raw",
                transcript="new clean",
                summary="new summary",
                quiz="new quiz",
                audio_path=None,
            )

        mock_process.side_effect = mock_process_fn

        with (
            patch(
                "sys.argv",
                [
                    "any-video",
                    "https://youtube.com/watch?v=abc123",
                    "--output-dir",
                    str(output_dir),
                    "--force",
                ],
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            main()

        mock_process.assert_called_once()
