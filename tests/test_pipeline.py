"""Tests for pipeline module."""

from unittest.mock import patch

from any_video.config import VideoMetadata
from any_video.pipeline import find_existing_output, process


class TestFindExistingOutput:
    def test_finds_existing(self, tmp_path):
        (tmp_path / "abc123_my-video").mkdir()
        result = find_existing_output(tmp_path, "abc123")
        assert result is not None
        assert result.name == "abc123_my-video"

    def test_returns_none_when_missing(self, tmp_path):
        assert find_existing_output(tmp_path, "abc123") is None

    def test_returns_most_recent_when_multiple(self, tmp_path):
        """When multiple dirs match (e.g., after title change), return most recent."""
        old_dir = tmp_path / "abc123_old-title"
        old_dir.mkdir()
        # Touch a file so mtime differs
        (old_dir / "marker").write_text("old")

        new_dir = tmp_path / "abc123_new-title"
        new_dir.mkdir()
        (new_dir / "marker").write_text("new")

        result = find_existing_output(tmp_path, "abc123")
        assert result == new_dir


class TestProcess:
    @patch("any_video.pipeline.generate_quiz", return_value="quiz")
    @patch("any_video.pipeline.generate_summary", return_value="summary")
    @patch("any_video.pipeline.beautify_transcript", return_value="beautified")
    @patch("any_video.pipeline.transcribe", return_value="raw transcript")
    @patch("any_video.pipeline.load_model")
    @patch("any_video.pipeline.download_audio")
    @patch("any_video.pipeline.get_video_metadata")
    def test_full_pipeline(
        self,
        mock_metadata,
        mock_download,
        mock_load_model,
        mock_transcribe,
        mock_beautify,
        mock_summary,
        mock_quiz,
        tmp_path,
    ):
        mock_metadata.return_value = VideoMetadata(
            video_id="abc123", title="Test Video", slug_title="test-video"
        )
        mock_download.return_value = tmp_path / "audio.mp3"
        (tmp_path / "audio.mp3").write_text("fake")

        output_path = process(
            url="https://www.youtube.com/watch?v=abc123",
            model_name="small",
            output_dir=tmp_path,
            keep_audio=False,
            force=False,
        )

        assert (output_path / "transcript_raw.md").read_text() == "raw transcript"
        assert (output_path / "transcript.md").read_text() == "beautified"
        assert (output_path / "summary.md").read_text() == "summary"
        assert (output_path / "quiz.md").read_text() == "quiz"

    @patch("any_video.pipeline.get_video_metadata")
    def test_returns_cached_output(self, mock_metadata, tmp_path):
        mock_metadata.return_value = VideoMetadata(
            video_id="abc123", title="Test", slug_title="test"
        )
        existing = tmp_path / "abc123_test"
        existing.mkdir()
        # Create all required output files to simulate complete output
        (existing / "transcript_raw.md").write_text("raw")
        (existing / "transcript.md").write_text("clean")
        (existing / "summary.md").write_text("summary")
        (existing / "quiz.md").write_text("quiz")

        result = process(
            url="https://www.youtube.com/watch?v=abc123",
            model_name="small",
            output_dir=tmp_path,
            keep_audio=False,
            force=False,
        )

        assert result == existing

    @patch("any_video.pipeline.beautify_transcript", return_value="beautified")
    @patch("any_video.pipeline.generate_summary", return_value="summary")
    @patch("any_video.pipeline.generate_quiz", return_value="quiz")
    @patch("any_video.pipeline.get_video_metadata")
    def test_resumes_incomplete_output(
        self, mock_metadata, mock_quiz, mock_summary, mock_beautify, tmp_path
    ):
        """If raw transcript exists but GPT outputs don't, resume from GPT step."""
        mock_metadata.return_value = VideoMetadata(
            video_id="abc123", title="Test", slug_title="test"
        )
        existing = tmp_path / "abc123_test"
        existing.mkdir()
        (existing / "transcript_raw.md").write_text("raw transcript")

        result = process(
            url="https://www.youtube.com/watch?v=abc123",
            model_name="small",
            output_dir=tmp_path,
            keep_audio=False,
            force=False,
        )

        assert (result / "transcript.md").read_text() == "beautified"
        assert (result / "summary.md").read_text() == "summary"
        assert (result / "quiz.md").read_text() == "quiz"
        mock_beautify.assert_called_once_with("raw transcript")

    @patch("any_video.pipeline.generate_quiz", return_value="quiz")
    @patch("any_video.pipeline.generate_summary", return_value="summary")
    @patch("any_video.pipeline.get_video_metadata")
    def test_skips_completed_gpt_steps_on_resume(
        self, mock_metadata, mock_summary, mock_quiz, tmp_path
    ):
        """If beautify already done but summary/quiz missing, skip beautify."""
        mock_metadata.return_value = VideoMetadata(
            video_id="abc123", title="Test", slug_title="test"
        )
        existing = tmp_path / "abc123_test"
        existing.mkdir()
        (existing / "transcript_raw.md").write_text("raw transcript")
        (existing / "transcript.md").write_text("already beautified")

        result = process(
            url="https://www.youtube.com/watch?v=abc123",
            model_name="small",
            output_dir=tmp_path,
            keep_audio=False,
            force=False,
        )

        # Beautified transcript should be preserved, not overwritten
        assert (result / "transcript.md").read_text() == "already beautified"
        assert (result / "summary.md").read_text() == "summary"
        assert (result / "quiz.md").read_text() == "quiz"
        # Summary received the existing beautified text, not a new GPT call
        mock_summary.assert_called_once_with("already beautified")

    @patch("any_video.pipeline.generate_quiz", return_value="quiz")
    @patch("any_video.pipeline.generate_summary", return_value="summary")
    @patch("any_video.pipeline.beautify_transcript", return_value="beautified")
    @patch("any_video.pipeline.transcribe", return_value="raw transcript")
    @patch("any_video.pipeline.load_model")
    @patch("any_video.pipeline.download_audio")
    @patch("any_video.pipeline.get_video_metadata")
    def test_force_reprocesses(
        self,
        mock_metadata,
        mock_download,
        mock_load_model,
        mock_transcribe,
        mock_beautify,
        mock_summary,
        mock_quiz,
        tmp_path,
    ):
        mock_metadata.return_value = VideoMetadata(
            video_id="abc123", title="Test", slug_title="test"
        )
        existing = tmp_path / "abc123_test"
        existing.mkdir()
        (existing / "old_file.txt").write_text("old")

        mock_download.return_value = tmp_path / "audio.mp3"
        (tmp_path / "audio.mp3").write_text("fake")

        result = process(
            url="https://www.youtube.com/watch?v=abc123",
            model_name="small",
            output_dir=tmp_path,
            keep_audio=False,
            force=True,
        )

        assert not (result / "old_file.txt").exists()
        assert (result / "transcript.md").exists()
