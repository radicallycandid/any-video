# CLAUDE.md — any-video

## Project context

CLI tool that downloads YouTube videos, transcribes them locally with Whisper, and generates learning materials (beautified transcripts, summaries, quizzes) via OpenAI GPT. One developer maintains it. The priority is correctness and clarity, not production robustness or extensibility.

## Tech stack

- **Python 3.10+** with type hints and dataclasses
- **openai-whisper** — local transcription
- **openai** — GPT API for beautification, summary, quiz
- **yt-dlp** — YouTube downloading
- **uv** — package manager

## Commands

```bash
# Run the CLI
uv run any-video "<youtube-url>" [--model small] [--output-dir ./output] [--keep-audio] [--force] [-v]

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Format code
uv run ruff format .
```

## Code style

- **Formatter:** Black, 100 char line length
- **Linter:** Ruff — rules: E, F, W, I, UP, B, C4, SIM (E501 ignored)
- **Target:** Python 3.10+
- **Tests:** pytest + pytest-mock

## Anti-patterns to refuse

- Do not add speculative abstractions (classes, interfaces, utility modules) for functionality that does not exist yet.
- Do not add configuration options that are not currently needed.
- Do not increase test coverage by adding tests that test implementation details rather than behavior.
- When in doubt between two approaches, choose the simpler one.

## Project structure

```
any_video/
  __init__.py         # Public API re-exports
  __main__.py         # python -m any_video entry point
  cli.py              # CLI argument parsing, output writing, cache check
  config.py           # Constants, logging setup, shared types/dataclasses
  downloader.py       # yt-dlp integration, URL validation, video ID extraction
  openai_client.py    # OpenAI API: beautify, summarize, quiz, chunking, retry
  pipeline.py         # Processing orchestrator
  transcriber.py      # Whisper model loading and transcription
tests/
  test_*.py           # Unit tests with mocks
pyproject.toml        # Project config, deps, tool settings
PRD.md                # Product requirements document
```

## Output structure

```
{output-dir}/{VIDEO_ID}_{slug-title}/
  transcript_raw.md   # Raw Whisper output
  transcript.md       # AI-beautified transcript
  summary.md          # AI-generated summary
  quiz.md             # 10 multiple-choice questions
  audio.mp3           # Only with --keep-audio
```

## Idempotency

Before processing, check if output for this video ID already exists. If it does and `--force` is not set, skip and print the existing path. If `--force` is set, delete existing output and re-process.

## External requirements

- **ffmpeg** must be installed system-wide
- **OPENAI_API_KEY** env var must be set
- Whisper models download automatically on first run
