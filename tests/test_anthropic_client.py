"""Tests for anthropic_client module."""

from unittest.mock import MagicMock, patch

import anthropic
import pytest

from any_video.anthropic_client import (
    beautify_transcript,
    generate_quiz,
    generate_summary,
)
from any_video.config import AnthropicError


def _mock_message(content: str) -> MagicMock:
    """Create a mock Anthropic Message response with a single text block."""
    block = MagicMock()
    block.type = "text"
    block.text = content
    response = MagicMock()
    response.content = [block]
    return response


def _stream_returning(message: MagicMock) -> MagicMock:
    """Create a mock context manager that yields a stream returning `message`."""
    stream = MagicMock()
    stream.get_final_message.return_value = message
    cm = MagicMock()
    cm.__enter__.return_value = stream
    cm.__exit__.return_value = False
    return cm


class TestBeautifyTranscript:
    @patch("any_video.anthropic_client._get_client")
    def test_beautifies_text(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.return_value = _stream_returning(_mock_message("Clean text."))

        assert beautify_transcript("messy text") == "Clean text."
        mock_client.messages.stream.assert_called_once()


class TestGenerateSummary:
    @patch("any_video.anthropic_client._get_client")
    def test_generates_summary(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.return_value = _stream_returning(
            _mock_message("## Summary\nKey points.")
        )

        assert generate_summary("Some transcript text.") == "## Summary\nKey points."


class TestGenerateQuiz:
    @patch("any_video.anthropic_client._get_client")
    def test_generates_quiz(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.return_value = _stream_returning(
            _mock_message("## Question 1\nQ: What?")
        )

        assert generate_quiz("Some transcript text.") == "## Question 1\nQ: What?"


class TestErrorHandling:
    @patch("any_video.anthropic_client._get_client")
    def test_wraps_api_error(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.side_effect = anthropic.APIError(
            "boom", request=MagicMock(), body=None
        )

        with pytest.raises(AnthropicError, match="Anthropic API error"):
            generate_summary("Some text.")

    @patch("any_video.anthropic_client._get_client")
    def test_raises_on_no_text_blocks(self, mock_get_client):
        mock_client = mock_get_client.return_value
        empty = MagicMock()
        empty.content = []
        mock_client.messages.stream.return_value = _stream_returning(empty)

        with pytest.raises(AnthropicError, match="no text content"):
            generate_summary("Some text.")
