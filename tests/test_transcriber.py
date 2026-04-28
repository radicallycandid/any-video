"""Tests for transcriber module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from any_video.config import TranscriptionError
from any_video.transcriber import load_model, transcribe


class TestLoadModel:
    @patch("any_video.transcriber.WhisperModel")
    def test_loads_model(self, mock_class):
        mock_model = MagicMock()
        mock_class.return_value = mock_model

        result = load_model("small")

        mock_class.assert_called_once_with("small", device="cpu", compute_type="int8")
        assert result is mock_model

    @patch("any_video.transcriber.WhisperModel")
    def test_wraps_error(self, mock_class):
        mock_class.side_effect = RuntimeError("model not found")

        with pytest.raises(TranscriptionError, match="Failed to load Whisper model"):
            load_model("nonexistent")


class TestTranscribe:
    def test_concatenates_segments(self):
        seg1 = MagicMock(text="Hello ", end=2.5)
        seg2 = MagicMock(text="world.", end=5.0)
        info = MagicMock(duration=5.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg1, seg2]), info)

        result = transcribe(mock_model, Path("/fake/audio.mp3"))

        assert result == "Hello world."
        mock_model.transcribe.assert_called_once_with("/fake/audio.mp3")

    def test_strips_whitespace(self):
        seg = MagicMock(text="  padded text  ", end=1.0)
        info = MagicMock(duration=1.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)

        assert transcribe(mock_model, Path("/fake/audio.mp3")) == "padded text"

    def test_wraps_error(self):
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("transcription failed")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            transcribe(mock_model, Path("/fake/audio.mp3"))
