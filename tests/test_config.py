"""Tests for config module."""

from any_video.config import AnyVideoError, DownloadError, OpenAIError, TranscriptionError, slugify


class TestSlugify:
    def test_simple_title(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        assert slugify("Hello! World? #1") == "hello-world-1"

    def test_consecutive_hyphens(self):
        assert slugify("Hello --- World") == "hello-world"

    def test_leading_trailing_hyphens(self):
        assert slugify("---Hello World---") == "hello-world"

    def test_truncates_long_titles(self):
        long_title = "a" * 100
        assert len(slugify(long_title)) == 50

    def test_empty_string(self):
        assert slugify("") == ""

    def test_unicode_characters(self):
        assert slugify("Café résumé") == "caf-r-sum"


class TestExceptionHierarchy:
    def test_download_error_is_any_video_error(self):
        assert issubclass(DownloadError, AnyVideoError)

    def test_transcription_error_is_any_video_error(self):
        assert issubclass(TranscriptionError, AnyVideoError)

    def test_openai_error_is_any_video_error(self):
        assert issubclass(OpenAIError, AnyVideoError)
