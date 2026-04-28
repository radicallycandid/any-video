# CLAUDE.md — any-video

## Project context

CLI tool that downloads YouTube videos, transcribes them locally with Whisper, and generates learning materials (beautified transcripts, summaries, quizzes) via the Anthropic API (Claude). One developer maintains it. The priority is correctness and clarity, not production robustness or extensibility.

## Tech stack

- **Python 3.10+** with type hints and dataclasses
- **faster-whisper** — local transcription (CTranslate2 backend)
- **anthropic** — Claude API for beautification, summary, quiz
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
  anthropic_client.py # Anthropic API: beautify, summarize, quiz
  pipeline.py         # Processing orchestrator
  transcriber.py      # Whisper model loading and transcription
tests/
  test_*.py           # Unit tests with mocks
pyproject.toml        # Project config, deps, tool settings
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

Before processing, check if output for this video ID already exists. If a complete output exists and `--force` is not set, skip and print the existing path. If the output exists but is incomplete (e.g. a previous run failed mid-pipeline), resume from the persisted artifacts — the raw transcript and any Claude outputs already on disk are reused, and only the missing steps run. If `--force` is set, delete existing output and re-process from scratch.

## Processing pipeline

```
URL → Download Audio → Transcribe (Whisper) → Beautify → Summarize → Quiz → Write Output
```

After validating the URL and resolving cache state (see Idempotency), the pipeline runs five logged steps — the loglines use `[1/5]…[5/5]`:

1. **[1/5]** Download audio as MP3 (yt-dlp, into a temp dir)
2. **[2/5]** Transcribe with Whisper; persist `transcript_raw.md` immediately so it survives later failures
3. **[3/5]** Beautify transcript via Claude; write `transcript.md`
4. **[4/5]** Generate summary via Claude; write `summary.md`
5. **[5/5]** Generate quiz via Claude (10 multiple-choice questions); write `quiz.md`

Each Claude call goes through `client.messages.stream()` with `effort: "low"` and `thinking: disabled` — the workload is content generation, not reasoning, so this minimizes tokens. Streaming is used to avoid SDK HTTP timeouts on long-running responses (e.g. beautifying a multi-hour transcript). Sonnet 4.6's 1M context window is large enough that the full transcript fits in a single request; no chunking is performed.

Each artifact is written as soon as it's produced, so a failed run can be resumed without re-downloading or re-running earlier steps. Audio is discarded when the temp dir is cleaned up unless `--keep-audio` is set, in which case it's copied to the output directory before cleanup.

Exit codes: 0 = success, 1 = general error.

## Error handling conventions

- Custom exceptions for distinct failure modes (download, transcription, API errors)
- The Anthropic SDK auto-retries 429/5xx with exponential backoff (configured via `max_retries` on the client)
- Clear error messages to stderr; no tracebacks in non-verbose mode

## Typing conventions

- Dataclasses for structured data (processing results, video metadata)
- Type hints on all function signatures
- No raw dicts for domain data — use typed structures

## Non-goals

- No web server or HTTP API
- No GUI or browser extension
- No support for non-YouTube sources
- No streaming/real-time transcription
- No database or persistent state beyond output files

## External requirements

- **ffmpeg** must be installed system-wide
- **ANTHROPIC_API_KEY** env var must be set
- Whisper models download automatically on first run
