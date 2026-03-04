# CLAUDE.md — any-video

## Project Overview

YouTube video transcriber and learning tool. Downloads videos, transcribes locally with OpenAI Whisper, then generates AI-powered beautified transcripts, summaries, and quizzes using GPT models. Also available as a Chrome extension backed by a local Flask server.

## Project Structure

```
any_video.py          # Main CLI tool (~700 lines)
server.py             # Flask server for Chrome extension
tests/test_any_video.py  # Test suite (pytest)
extension/            # Chrome extension (Manifest V3)
  popup.html/js/css   # Extension UI
  manifest.json       # Extension config
pyproject.toml        # Project config, deps, tool settings
```

## Tech Stack

- **Python 3.10+** with type hints throughout
- **openai-whisper** — local transcription (no API cost)
- **openai** — GPT-4.1 (beautification), GPT-5.2 (summary/quiz)
- **yt-dlp** — YouTube downloading
- **Flask + flask-cors** — local server for Chrome extension
- **uv** — recommended package manager

## Commands

```bash
# Run the CLI
uv run any-video "<youtube-url>" [--model small|tiny|large-v3] [--output-dir ./output] [--keep-audio] [-v]

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Start the Flask server (Chrome extension backend)
uv run python server.py
```

## Code Style

- **Formatter:** Black, 100 char line length
- **Linter:** Ruff — rules: E, F, W, I, UP, B, C4, SIM (E501 ignored, handled by Black)
- **Target:** Python 3.10+
- **Test framework:** pytest with pytest-mock

## Key Patterns

- **Custom exceptions:** `TranscriptionError`, `APIError`, `DownloadError` — each for a distinct failure mode
- **Retry decorator:** `@retry_with_backoff` with exponential backoff on OpenAI rate limit / connection / timeout errors
- **Model token params:** uses `max_completion_tokens` for gpt-4.1+, gpt-5.x, o-series; `max_tokens` for older models
- **Transcript truncation:** truncates at sentence boundaries, max 100K chars

## External Requirements

- **ffmpeg** must be installed system-wide (`brew install ffmpeg`)
- **OPENAI_API_KEY** env var must be set
- Whisper models download to `~/whisper/` on first run

## Output

Each processed video creates a folder under `output/{VIDEO_ID}_{slug-title}/` containing:
- `transcript_raw.md` — raw Whisper output
- `transcript.md` — AI-beautified transcript
- `summary.md` — AI-generated summary
- `quiz.md` — 10 multiple-choice questions
- `audio.mp3` — only with `--keep-audio`

## Server API

- `GET /health` — returns status and whether OpenAI key is configured
- `POST /process` — accepts `{url, model?, keep_audio?}`, returns transcription results

Server runs on `localhost:8765`.
