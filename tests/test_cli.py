"""Tests for CLI module."""

from unittest.mock import patch

import pytest

from any_video.cli import main, parse_args


class TestParseArgs:
    def test_minimal_args(self):
        args = parse_args(["https://www.youtube.com/watch?v=abc123"])
        assert args.url == "https://www.youtube.com/watch?v=abc123"
        assert args.model == "small"
        assert args.output_dir == "./output"
        assert args.keep_audio is False
        assert args.force is False
        assert args.verbose is False

    def test_all_flags(self):
        args = parse_args(
            [
                "https://youtu.be/abc123",
                "--model",
                "large-v3",
                "--output-dir",
                "/tmp/out",
                "--keep-audio",
                "--force",
                "-v",
            ]
        )
        assert args.model == "large-v3"
        assert args.output_dir == "/tmp/out"
        assert args.keep_audio is True
        assert args.force is True
        assert args.verbose is True

    def test_invalid_model_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["https://youtu.be/abc123", "--model", "invalid"])

    def test_missing_url_rejected(self):
        with pytest.raises(SystemExit):
            parse_args([])


class TestMain:
    @patch("any_video.cli.process")
    def test_success_prints_path(self, mock_process, tmp_path, capsys):
        mock_process.return_value = tmp_path / "abc123_test"

        main(["https://www.youtube.com/watch?v=abc123"])

        captured = capsys.readouterr()
        assert "abc123_test" in captured.out

    @patch("any_video.cli.process")
    def test_error_exits_with_1(self, mock_process):
        from any_video.config import DownloadError

        mock_process.side_effect = DownloadError("bad url")

        with pytest.raises(SystemExit) as exc_info:
            main(["https://www.youtube.com/watch?v=abc123"])

        assert exc_info.value.code == 1
