"""Tests for openai_client module."""

from unittest.mock import MagicMock, patch

import openai
import pytest

from any_video.config import MAX_CHUNK_CHARS, OpenAIError
from any_video.openai_client import (
    _chunk_text,
    beautify_transcript,
    generate_quiz,
    generate_summary,
)


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world."
        assert _chunk_text(text) == [text]

    def test_splits_at_sentence_boundary(self):
        # Create text that exceeds max_chars
        sentence = "This is a sentence. "
        text = sentence * 100
        chunks = _chunk_text(text, max_chars=100)
        assert len(chunks) > 1
        # Each chunk should end at a sentence boundary (except possibly the last)
        for chunk in chunks[:-1]:
            assert chunk.rstrip().endswith(".")

    def test_handles_no_sentence_boundary(self):
        text = "a" * 200
        chunks = _chunk_text(text, max_chars=100)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 100
        assert chunks[1] == "a" * 100

    def test_exact_max_chars(self):
        text = "a" * MAX_CHUNK_CHARS
        assert _chunk_text(text) == [text]


def _mock_openai_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestBeautifyTranscript:
    @patch("any_video.openai_client.openai.OpenAI")
    def test_beautifies_short_text(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response("Clean text.")

        result = beautify_transcript("messy text")

        assert result == "Clean text."
        mock_client.chat.completions.create.assert_called_once()

    @patch("any_video.openai_client.openai.OpenAI")
    def test_chunks_long_text(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response("Clean chunk.")

        long_text = "This is a sentence. " * 3000  # Well over MAX_CHUNK_CHARS
        result = beautify_transcript(long_text)

        assert mock_client.chat.completions.create.call_count > 1
        assert "Clean chunk." in result


class TestGenerateSummary:
    @patch("any_video.openai_client.openai.OpenAI")
    def test_generates_summary(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "## Summary\nKey points."
        )

        result = generate_summary("Some transcript text.")

        assert result == "## Summary\nKey points."


class TestGenerateQuiz:
    @patch("any_video.openai_client.openai.OpenAI")
    def test_generates_quiz(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "## Question 1\nQ: What?"
        )

        result = generate_quiz("Some transcript text.")

        assert result == "## Question 1\nQ: What?"


class TestRetry:
    @patch("any_video.openai_client.time.sleep")
    @patch("any_video.openai_client.openai.OpenAI")
    def test_retries_on_rate_limit(self, mock_openai_class, mock_sleep):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            openai.RateLimitError("rate limited", response=MagicMock(), body=None),
            _mock_openai_response("Success after retry."),
        ]

        result = generate_summary("Some text.")

        assert result == "Success after retry."
        assert mock_sleep.call_count == 1

    @patch("any_video.openai_client.time.sleep")
    @patch("any_video.openai_client.openai.OpenAI")
    def test_raises_after_max_retries(self, mock_openai_class, mock_sleep):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            "rate limited", response=MagicMock(), body=None
        )

        with pytest.raises(OpenAIError, match="failed after 3 retries"):
            generate_summary("Some text.")

    @patch("any_video.openai_client.openai.OpenAI")
    def test_raises_on_non_retryable_error(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            "bad key", response=MagicMock(), body=None
        )

        with pytest.raises(OpenAIError, match="OpenAI API error"):
            generate_summary("Some text.")
