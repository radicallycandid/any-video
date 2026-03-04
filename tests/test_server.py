"""Tests for the Flask server."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from any_video import APIError, DownloadError, ProcessingResult, TranscriptionError


@pytest.fixture
def client():
    """Create a Flask test client with rate limiting disabled."""
    from server import app, limiter

    app.config["TESTING"] = True
    limiter.enabled = False
    with app.test_client() as client:
        yield client
    limiter.enabled = True


@pytest.fixture(autouse=True)
def clear_jobs():
    """Clear the jobs store before each test."""
    from server import _jobs, _jobs_lock

    with _jobs_lock:
        _jobs.clear()
    yield


@pytest.fixture
def rate_limited_client():
    """Create a Flask test client with rate limiting enabled."""
    from server import app, limiter

    app.config["TESTING"] = True
    limiter.enabled = True
    limiter.reset()
    with app.test_client() as client:
        yield client


def _wait_for_result(client, request_id, timeout=5):
    """Poll /result/<id> until the job completes or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/result/{request_id}")
        data = response.get_json()
        if data.get("status") != "pending":
            return response
        time.sleep(0.1)
    raise TimeoutError(f"Job {request_id} did not complete within {timeout}s")


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
    def test_process_returns_request_id(self, mock_process, client):
        """Test that /process returns a request_id for async polling."""
        mock_process.return_value = ProcessingResult(
            video_id="id",
            video_title="t",
            transcript_raw="r",
            transcript="c",
            summary="s",
            quiz="q",
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        assert response.status_code == 200
        data = response.get_json()
        assert "request_id" in data

    @patch("server.process_video")
    def test_successful_processing(self, mock_process, client):
        """Test successful video processing returns all fields via /result."""
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

        request_id = response.get_json()["request_id"]
        result_response = _wait_for_result(client, request_id)

        assert result_response.status_code == 200
        data = result_response.get_json()
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
        """Test that DownloadError maps to 500 via /result."""
        mock_process.side_effect = DownloadError("Video not found")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        result_response = _wait_for_result(client, request_id)

        assert result_response.status_code == 500
        data = result_response.get_json()
        assert data["success"] is False
        assert "Video not found" in data["error"]

    @patch("server.process_video")
    def test_transcription_error_returns_500(self, mock_process, client):
        """Test that TranscriptionError maps to 500 via /result."""
        mock_process.side_effect = TranscriptionError("Whisper crashed")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        result_response = _wait_for_result(client, request_id)

        assert result_response.status_code == 500
        data = result_response.get_json()
        assert data["success"] is False
        assert "Whisper crashed" in data["error"]

    @patch("server.process_video")
    def test_api_error_returns_500(self, mock_process, client):
        """Test that APIError maps to 500 via /result."""
        mock_process.side_effect = APIError("Rate limited")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        result_response = _wait_for_result(client, request_id)

        assert result_response.status_code == 500
        data = result_response.get_json()
        assert data["success"] is False
        assert "Rate limited" in data["error"]

    @patch("server.process_video")
    def test_value_error_returns_400(self, mock_process, client):
        """Test that ValueError (bad URL) maps to 400 via /result."""
        mock_process.side_effect = ValueError("Could not extract video ID")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://example.com/not-youtube"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        result_response = _wait_for_result(client, request_id)

        assert result_response.status_code == 400
        data = result_response.get_json()
        assert data["success"] is False

    @patch("server.process_video")
    def test_unexpected_error_returns_500(self, mock_process, client):
        """Test that unexpected exceptions map to 500 via /result."""
        mock_process.side_effect = RuntimeError("Something weird")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        result_response = _wait_for_result(client, request_id)

        assert result_response.status_code == 500
        data = result_response.get_json()
        assert data["success"] is False
        assert "Unexpected error" in data["error"]

    @patch("server.process_video")
    def test_os_error_returns_500(self, mock_process, client):
        """Test that OSError maps to 500 with unexpected error message."""
        mock_process.side_effect = OSError("Disk full")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        result_response = _wait_for_result(client, request_id)

        assert result_response.status_code == 500
        data = result_response.get_json()
        assert data["success"] is False
        assert "Unexpected error" in data["error"]

    @patch("server.process_video")
    def test_default_model_is_small(self, mock_process, client):
        """Test that model defaults to 'small' when not specified."""
        mock_process.return_value = ProcessingResult(
            video_id="id",
            video_title="t",
            transcript_raw="r",
            transcript="c",
            summary="s",
            quiz="q",
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        _wait_for_result(client, request_id)

        call_args = mock_process.call_args
        assert call_args[1].get("model") == "small" or call_args[0][1] == "small"

    @patch("server.process_video")
    def test_keep_audio_moves_file(self, mock_process, client, tmp_path):
        """Test that keep_audio=true moves audio to persistent location."""
        import server as server_mod

        # Create a fake audio file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        audio_file = work_dir / "audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_process.return_value = ProcessingResult(
            video_id="abc123",
            video_title="Test Video",
            transcript_raw="raw",
            transcript="clean",
            summary="summary",
            quiz="quiz",
            audio_path=audio_file,
        )

        # Redirect server's output directory to tmp_path
        original_file = server_mod.__file__
        server_mod.__file__ = str(tmp_path / "server.py")

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
                response = client.post(
                    "/process",
                    data=json.dumps(
                        {
                            "url": "https://youtube.com/watch?v=abc123",
                            "keep_audio": True,
                        }
                    ),
                    content_type="application/json",
                )

            request_id = response.get_json()["request_id"]
            result_response = _wait_for_result(client, request_id)
        finally:
            server_mod.__file__ = original_file

        assert result_response.status_code == 200
        data = result_response.get_json()
        assert data["success"] is True
        assert "audio_path" in data
        assert "abc123" in data["audio_path"]

        # Audio file should have been moved
        assert not audio_file.exists()
        assert Path(data["audio_path"]).exists()


class TestProgressEndpoint:
    """Tests for GET /progress/<request_id>."""

    @patch("server.process_video")
    def test_progress_endpoint_returns_stage(self, mock_process, client):
        """Test that /progress returns current stage and percentage."""
        import threading

        started = threading.Event()
        proceed = threading.Event()

        def slow_process(*args, **kwargs):
            on_progress = kwargs.get("on_progress")
            if on_progress:
                on_progress("Downloading audio", 0)
            started.set()
            proceed.wait(timeout=5)
            return ProcessingResult(
                video_id="id",
                video_title="t",
                transcript_raw="r",
                transcript="c",
                summary="s",
                quiz="q",
            )

        mock_process.side_effect = slow_process

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        started.wait(timeout=5)

        progress_response = client.get(f"/progress/{request_id}")
        data = progress_response.get_json()
        assert progress_response.status_code == 200
        assert "stage" in data
        assert "progress" in data

        proceed.set()
        _wait_for_result(client, request_id)

    def test_progress_unknown_id_returns_404(self, client):
        """Test that unknown request_id returns 404."""
        response = client.get("/progress/nonexistent-id")
        assert response.status_code == 404

    def test_result_unknown_id_returns_404(self, client):
        """Test that unknown request_id returns 404 from /result."""
        response = client.get("/result/nonexistent-id")
        assert response.status_code == 404

    @patch("server.process_video")
    def test_result_returns_pending_while_processing(self, mock_process, client):
        """Test that /result returns pending status while job is running."""
        import threading

        started = threading.Event()
        proceed = threading.Event()

        def slow_process(*args, **kwargs):
            started.set()
            proceed.wait(timeout=5)
            return ProcessingResult(
                video_id="id",
                video_title="t",
                transcript_raw="r",
                transcript="c",
                summary="s",
                quiz="q",
            )

        mock_process.side_effect = slow_process

        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            response = client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )

        request_id = response.get_json()["request_id"]
        started.wait(timeout=5)

        result_response = client.get(f"/result/{request_id}")
        data = result_response.get_json()
        assert data["status"] == "pending"

        proceed.set()
        _wait_for_result(client, request_id)


class TestRateLimiting:
    """Tests for rate limiting on endpoints."""

    def test_process_rate_limit(self, rate_limited_client):
        """Test that /process is rate limited to 5 per minute."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            for _ in range(5):
                response = rate_limited_client.post(
                    "/process",
                    data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                    content_type="application/json",
                )
                # These will fail with 400/500 but not 429
                assert response.status_code != 429

            # 6th request should be rate limited
            response = rate_limited_client.post(
                "/process",
                data=json.dumps({"url": "https://youtube.com/watch?v=abc"}),
                content_type="application/json",
            )
            assert response.status_code == 429

    def test_health_not_rate_limited_under_normal_use(self, rate_limited_client):
        """Test that /health allows at least 10 rapid requests."""
        for _ in range(10):
            response = rate_limited_client.get("/health")
            assert response.status_code == 200
