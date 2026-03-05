"""Tests for pipeline module."""

from unittest.mock import patch

from any_video.config import ProcessingResult, VideoMetadata
from any_video.pipeline import find_existing_output, process, write_output


class TestFindExistingOutput:
    def test_finds_existing(self, tmp_path):
        (tmp_path / "abc123_my-video").mkdir()
        result = find_existing_output(tmp_path, "abc123")
        assert result is not None
        assert result.name == "abc123_my-video"

    def test_returns_none_when_missing(self, tmp_path):
        assert find_existing_output(tmp_path, "abc123") is None


class TestWriteOutput:
    def test_writes_all_files(self, tmp_path):
        metadata = VideoMetadata(video_id="abc123", title="Test", slug_title="test")
        result = ProcessingResult(
            raw_transcript="raw text",
            beautified_transcript="clean text",
            summary="summary text",
            quiz="quiz text",
        )

        output_path = write_output(tmp_path, metadata, result, audio_path=None, keep_audio=False)

        assert output_path.name == "abc123_test"
        assert (output_path / "transcript_raw.md").read_text() == "raw text"
        assert (output_path / "transcript.md").read_text() == "clean text"
        assert (output_path / "summary.md").read_text() == "summary text"
        assert (output_path / "quiz.md").read_text() == "quiz text"
        assert not (output_path / "audio.mp3").exists()

    def test_copies_audio_when_keep_audio(self, tmp_path):
        metadata = VideoMetadata(video_id="abc123", title="Test", slug_title="test")
        result = ProcessingResult(
            raw_transcript="raw",
            beautified_transcript="clean",
            summary="summary",
            quiz="quiz",
        )
        audio_file = tmp_path / "source_audio.mp3"
        audio_file.write_text("fake audio data")

        output_path = write_output(
            tmp_path, metadata, result, audio_path=audio_file, keep_audio=True
        )

        assert (output_path / "audio.mp3").exists()


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

        result = process(
            url="https://www.youtube.com/watch?v=abc123",
            model_name="small",
            output_dir=tmp_path,
            keep_audio=False,
            force=False,
        )

        assert result == existing

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
