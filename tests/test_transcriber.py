"""Tests for transcriber module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from any_video.config import TranscriptionError
from any_video.transcriber import load_model, transcribe


class TestLoadModel:
    @patch("any_video.transcriber.whisper.load_model")
    def test_loads_model(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        result = load_model("small")

        mock_load.assert_called_once_with("small")
        assert result is mock_model

    @patch("any_video.transcriber.whisper.load_model")
    def test_wraps_error(self, mock_load):
        mock_load.side_effect = RuntimeError("model not found")

        with pytest.raises(TranscriptionError, match="Failed to load Whisper model"):
            load_model("nonexistent")


class TestTranscribe:
    @patch("any_video.transcriber.whisper.load_model")
    def test_returns_text(self, _mock_load):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello world  "}

        result = transcribe(mock_model, Path("/fake/audio.mp3"))

        assert result == "Hello world"
        mock_model.transcribe.assert_called_once_with("/fake/audio.mp3")

    @patch("any_video.transcriber.whisper.load_model")
    def test_wraps_error(self, _mock_load):
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("transcription failed")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            transcribe(mock_model, Path("/fake/audio.mp3"))
