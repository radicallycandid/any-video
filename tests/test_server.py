"""Tests for the Flask server."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from any_video import APIError, DownloadError, ProcessingResult, TranscriptionError


@pytest.fixture
def client():
    """Create a Flask test client."""
    from server import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_with_api_key(self, client):
        """Test health check when API key is configured."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            response = client.get("/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["openai_configured"] is True

    def test_health_without_api_key(self, client):
        """Test health check when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            response = client.get("/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["openai_configured"] is False


class TestProcessEndpoint:
    """Tests for POST /process."""

    def test_missing_body(self, client):
        """Test request with no JSON body."""
        response = client.post(
            "/process",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "url" in data["error"].lower()

    def test_missing_url(self, client):
        """Test request with JSON body but no url field."""
        response = client.post(
            "/process",
            data=json.dumps({"model": "small"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False

    def test_invalid_model(self, client):
        """Test request with invalid model choice."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc", "model": "huge"}),
                content_type="application/json",
            )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "huge" in data["error"]

    def test_missing_api_key(self, client):
        """Test request when OPENAI_API_KEY is not set."""
        with patch.dict("os.environ", {}, clear=True):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )
        assert response.status_code == 500
        data = response.get_json()
        assert data["success"] is False
        assert "OPENAI_API_KEY" in data["error"]

    @patch("server.process_video")
    def test_successful_processing(self, mock_process, client):
        """Test successful video processing returns all fields."""
        mock_process.return_value = ProcessingResult(
            video_id="abc123",
            video_title="Test Video",
            transcript_raw="raw text",
            transcript="clean text",
            summary="summary text",
            quiz="quiz text",
            audio_path=None,
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc123"}),
                content_type="application/json",
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["video_id"] == "abc123"
        assert data["video_title"] == "Test Video"
        assert data["transcript_raw"] == "raw text"
        assert data["transcript"] == "clean text"
        assert data["summary"] == "summary text"
        assert data["quiz"] == "quiz text"
        assert "audio_path" not in data

    @patch("server.process_video")
    def test_download_error_returns_500(self, mock_process, client):
        """Test that DownloadError maps to 500."""
        mock_process.side_effect = DownloadError("Video not found")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        assert response.status_code == 500
        data = response.get_json()
        assert data["success"] is False
        assert "Video not found" in data["error"]

    @patch("server.process_video")
    def test_transcription_error_returns_500(self, mock_process, client):
        """Test that TranscriptionError maps to 500."""
        mock_process.side_effect = TranscriptionError("Whisper crashed")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        assert response.status_code == 500
        data = response.get_json()
        assert data["success"] is False
        assert "Whisper crashed" in data["error"]

    @patch("server.process_video")
    def test_api_error_returns_500(self, mock_process, client):
        """Test that APIError maps to 500."""
        mock_process.side_effect = APIError("Rate limited")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        assert response.status_code == 500
        data = response.get_json()
        assert data["success"] is False
        assert "Rate limited" in data["error"]

    @patch("server.process_video")
    def test_value_error_returns_400(self, mock_process, client):
        """Test that ValueError (bad URL) maps to 400."""
        mock_process.side_effect = ValueError("Could not extract video ID")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://example.com/not-youtube"}),
                content_type="application/json",
            )

        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False

    @patch("server.process_video")
    def test_unexpected_error_returns_500(self, mock_process, client):
        """Test that unexpected exceptions map to 500."""
        mock_process.side_effect = RuntimeError("Something weird")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        assert response.status_code == 500
        data = response.get_json()
        assert data["success"] is False
        assert "Unexpected error" in data["error"]

    @patch("server.process_video")
    def test_default_model_is_small(self, mock_process, client):
        """Test that model defaults to 'small' when not specified."""
        mock_process.return_value = ProcessingResult(
            video_id="id", video_title="t", transcript_raw="r",
            transcript="c", summary="s", quiz="q",
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        call_args = mock_process.call_args
        assert call_args[0][1] == "small"  # second positional arg is model
