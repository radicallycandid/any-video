# any-video v2 — Product Requirements Document

## Overview

**any-video** is a CLI tool that downloads YouTube videos, transcribes them locally using OpenAI Whisper, and generates AI-powered learning materials (beautified transcripts, summaries, and quizzes) using OpenAI GPT models.

This is a from-scratch rewrite of v1, focused on a clean CLI-only tool with idempotent output caching and strict typing.

## Target User

A developer who clones the repo and runs the tool locally. No hosted service, no browser extension — just a CLI.

## External Dependencies

- **ffmpeg** — must be installed system-wide (e.g., `brew install ffmpeg`)
- **OPENAI_API_KEY** — environment variable for GPT API calls
- Whisper models download automatically to `~/.cache/whisper/` on first run

## CLI Interface

```
any-video "<youtube-url>" [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `small` | Whisper model size (`tiny`, `small`, `medium`, `large-v3`) |
| `--output-dir` | `./output` | Base directory for output |
| `--keep-audio` | `false` | Retain the downloaded audio file in output |
| `--force` | `false` | Re-process even if output already exists |
| `-v` / `--verbose` | `false` | Verbose logging |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (invalid URL, missing deps, API failure, etc.) |

## Processing Pipeline

```
URL → Download Audio → Transcribe (Whisper) → Beautify → Summarize → Quiz → Write Output
```

### Steps

1. **Parse & validate URL** — Extract YouTube video ID from the URL. Support standard (`youtube.com/watch?v=`), short (`youtu.be/`), and embed (`youtube.com/embed/`) formats.

2. **Check cache** — Look for existing output directory for this video ID. If found and `--force` is not set, skip processing and exit with a message. If `--force` is set, delete the existing output and re-process.

3. **Get video metadata** — Fetch the video title using yt-dlp (no download).

4. **Download audio** — Download audio as MP3 using yt-dlp + ffmpeg.

5. **Transcribe** — Run Whisper locally with the selected model. Produces raw transcript text.

6. **Beautify transcript** — Send raw transcript to GPT to clean up formatting, fix punctuation, and add paragraph breaks. Long transcripts are split into chunks at sentence boundaries to fit model context limits.

7. **Generate summary** — Send the beautified transcript to GPT to produce a structured summary.

8. **Generate quiz** — Send the beautified transcript to GPT to produce 10 multiple-choice questions with answers.

9. **Write output** — Write all artifacts to the output directory.

10. **Cleanup** — Delete the downloaded audio file unless `--keep-audio` is set.

## Output Structure

```
{output-dir}/{VIDEO_ID}_{slug-title}/
  transcript_raw.md    — Raw Whisper output
  transcript.md        — AI-beautified transcript
  summary.md           — AI-generated summary
  quiz.md              — 10 multiple-choice questions
  audio.mp3            — Only if --keep-audio
```

The slug title is derived from the video title: lowercased, non-alphanumeric characters replaced with hyphens, truncated to a reasonable length.

## Idempotency / Caching

- Before processing, check if `{output-dir}/{VIDEO_ID}_*/` already exists (match on video ID prefix).
- If it exists and `--force` is not set: print a message with the existing path and exit 0.
- If `--force` is set: delete the existing directory and re-process from scratch.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ with type hints |
| Transcription | openai-whisper (local) |
| LLM | OpenAI API (GPT-4.1+ for beautification, GPT-5.x for summary/quiz) |
| Download | yt-dlp |
| Package manager | uv |
| Linting | Ruff (rules: E, F, W, I, UP, B, C4, SIM; ignore E501) |
| Formatting | Black (100 char line length) |
| Testing | pytest + pytest-mock |

## Code Architecture

Moderate modularity — not a single script, but not over-abstracted either.

### Modules

| Module | Responsibility |
|--------|---------------|
| `any_video/cli.py` | Argument parsing, output writing, entry point, cache checking |
| `any_video/downloader.py` | YouTube URL validation, video ID extraction, metadata fetching, audio download |
| `any_video/transcriber.py` | Whisper model loading and transcription |
| `any_video/openai_client.py` | All OpenAI API interactions: beautify, summarize, quiz. Chunking logic, retry with backoff. |
| `any_video/pipeline.py` | Orchestrates the full processing pipeline, ties modules together |
| `any_video/config.py` | Constants, logging setup, shared types/dataclasses |
| `any_video/__init__.py` | Public API re-exports |
| `any_video/__main__.py` | `python -m any_video` entry point |

### Typing

- Use `dataclasses` for structured data (processing results, video metadata).
- Type hints on all function signatures.
- No raw dicts for domain data — use typed structures.

### Error Handling

- Custom exceptions for distinct failure modes (download, transcription, API errors).
- Retry with exponential backoff for transient OpenAI errors (rate limits, timeouts).
- Clear error messages printed to stderr; no tracebacks in non-verbose mode.

### Testing

- Unit tests with mocks for all external calls (yt-dlp, Whisper, OpenAI API).
- pytest + pytest-mock.
- Coverage target: 80%+.

## Non-Goals (v2)

- No web server or HTTP API
- No Chrome extension
- No GUI
- No support for non-YouTube sources
- No streaming/real-time transcription
- No database or persistent state beyond output files
