# Engineering Quality Assessment — any-video

**Date:** 2026-03-05
**Commit:** v2 branch (4c52ff1)
**Tests:** 52 passed | **Linter:** All checks passed

---

## Dimension Scores

### 1. Correctness — 5 / 5

**Evidence:** Every external call (yt-dlp, Whisper, OpenAI) is wrapped in a try/except that converts to a domain-specific exception with context. The pipeline persists raw transcript immediately so GPT failures don't lose transcription work ([pipeline.py:82](any_video/pipeline.py#L82)). Retry with exponential backoff handles transient API errors. `download_audio` verifies the MP3 file exists post-download. Unsupported URL formats are explicitly rejected with a clear error message.

**To improve:** Add support for `youtube.com/shorts/` URLs in `extract_video_id` — an increasingly common format that is currently cleanly rejected.

### 2. Readability — 4 / 5

**Evidence:** The pipeline module reads like a checklist: numbered steps (`[1/5]`…`[5/5]`), clear skip-if-exists guards, and descriptive function names throughout. A new developer can trace the full flow from [cli.py:54](any_video/cli.py#L54) → [pipeline.py:33](any_video/pipeline.py#L33) → individual modules in minutes.

**To improve:** The `process()` function at 70+ lines handles cache checking, downloading, transcription, and three GPT steps inline. Extracting the GPT-processing block into a helper would make the two phases (acquire transcript vs. generate materials) visually distinct.

### 3. Architecture fit — 5 / 5

**Evidence:** The module structure (cli → pipeline → downloader/transcriber/openai_client, with config for shared types) matches the problem perfectly. No abstract base classes, no dependency injection, no plugin system — just functions calling functions. The `_get_client()` singleton is the simplest viable pattern for a CLI tool.

**To improve:** Already well-matched. No action needed.

### 4. Test quality — 5 / 5

**Evidence:** Tests cover real behavioral scenarios: full pipeline run, cache hit, resumption from partial output, force reprocessing, and skipping already-completed GPT steps ([test_pipeline.py:95-151](tests/test_pipeline.py#L95-L151)). `_is_output_complete` is exercised indirectly through cache-hit and resume tests that set up the exact file states for both branches. Retry behavior is tested with mocked `time.sleep`. Edge cases like None content (refusal), non-retryable auth errors, and chunking at sentence boundaries are all covered.

**To improve:** Add a test for `--keep-audio` actually copying the audio file to the output directory.

### 5. Leanness — 5 / 5

**Evidence:** Every module, constant, and function is used. No "just in case" abstractions, no unused config options, no commented-out code, no speculative future-proofing. `__init__.py` contains only a docstring. The `OUTPUT_FILES` dict centralizes filenames and is referenced consistently.

**To improve:** Already lean. No action needed.

### 6. Operational fitness — 5 / 5

**Evidence:** Step-by-step progress logging (`[1/5] Downloading audio...`), skip messages on resume, retry warnings with attempt counts, and clean stderr/stdout separation (logs to stderr, output path to stdout for scripting). The custom exception hierarchy surfaces actionable messages at the CLI level ([cli.py:69](any_video/cli.py#L69)). Errors are always observable and actionable.

**To improve:** Elapsed time in the final log message would be nice for comparing Whisper model sizes.

### 7. Consistency — 5 / 5

**Evidence:** Every module follows identical patterns: `logger = logging.getLogger("any_video")` at module top, domain exceptions wrapping third-party errors, consistent docstring style, consistent test structure (class-based grouping with descriptive method names). Import ordering follows ruff's `I` rule throughout.

**To improve:** Already consistent. No action needed.

### 8. Dependency hygiene — 4 / 5

**Evidence:** Three runtime dependencies, all essential and irreplaceable for the task. Version bounds are reasonable (`openai>=1.0,<3`, `yt-dlp>=2024.0,<2027`). Dev dependencies are minimal (pytest, pytest-mock, ruff). No unnecessary transitive dependencies pulled in by choice.

**To improve:** Pin the `openai-whisper` upper bound more precisely — `<20260000` is a date-based scheme that will need updating. A comment explaining the versioning scheme would help future maintenance.

---

## Aggregate Score: 4.8 / 5.0

| # | Dimension | Score |
|---|-----------|-------|
| 1 | Correctness | 5 |
| 2 | Readability | 4 |
| 3 | Architecture fit | 5 |
| 4 | Test quality | 5 |
| 5 | Leanness | 5 |
| 6 | Operational fitness | 5 |
| 7 | Consistency | 5 |
| 8 | Dependency hygiene | 4 |

---

## Issue List

### P1 — Fix this cycle

None.

### P2 — Fix when convenient

```
[P2] [test quality] [tests/test_pipeline.py] — No test for --keep-audio flag actually copying the audio file to the output directory
[P2] [correctness] [any_video/downloader.py:14-18] — YouTube Shorts URLs (youtube.com/shorts/VIDEO_ID) are not matched by any pattern in _YOUTUBE_PATTERNS; the tool rejects them cleanly but it's an increasingly common format
[P2] [dependency hygiene] [pyproject.toml:11] — openai-whisper upper bound <20260000 uses an opaque date-based version scheme with no explanatory comment
[P2] [readability] [any_video/pipeline.py:33-114] — process() is 80 lines handling two distinct phases (acquire transcript, generate materials); extracting the GPT phase would improve scanability
```
