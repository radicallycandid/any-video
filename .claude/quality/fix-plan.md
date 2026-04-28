# Fix Plan

**Date:** 2026-04-27
**Source assessment:** `.claude/quality/last-assessment.md` (2026-04-27)

No P0 or P1 issues exist. This plan addresses all three P2s — they're all small, high-leverage cleanups (one regex line, two pyproject lines), so they're worth doing as a batch rather than skipping.

---

## Task 1 — Add YouTube Shorts URL support

**Issue(s):** P2 (correctness) — README claims shorts support; downloader doesn't implement it.

**What to do:** Add a fourth regex pattern to `_YOUTUBE_PATTERNS` in [any_video/downloader.py](any_video/downloader.py) matching `youtube.com/shorts/<id>`. Modeled on the existing embed pattern. Then add a test in [tests/test_downloader.py](tests/test_downloader.py) verifying `extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"` (and a `www.` variant).

Concrete patch:

```python
# any_video/downloader.py — add to _YOUTUBE_PATTERNS list (line 14-18):
re.compile(r"(?:youtube\.com/shorts/)([\w-]{11})"),
```

```python
# tests/test_downloader.py — add to TestExtractVideoId class:
def test_shorts_url(self):
    assert extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_shorts_url_with_www(self):
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
```

**Files involved:**
- `any_video/downloader.py`
- `tests/test_downloader.py`

**Verification:**
- `uv run pytest tests/test_downloader.py -v` — new tests pass, existing tests still pass.
- `uv run ruff check .` — clean.

---

## Task 2 — Update pyproject.toml description (post-migration)

**Issue(s):** P2 (dependency-hygiene) — pyproject description still says "via GPT" after Anthropic migration.

**What to do:** Change the `description` field in [pyproject.toml:8](pyproject.toml#L8) from `"...generate learning materials via GPT"` to `"...generate learning materials via Claude"`.

**Files involved:**
- `pyproject.toml`

**Verification:**
- `grep -n "via GPT" pyproject.toml` returns nothing.
- `uv run pytest` — still passes (description change has no functional effect; sanity check only).

---

## Task 3 — Tighten openai-whisper upper version bound

**Issue(s):** P2 (dependency-hygiene) — `openai-whisper>=20240930,<20260000` upper bound is effectively unbounded (~13 years).

**What to do:** Change [pyproject.toml:11](pyproject.toml#L11) upper bound from `<20260000` to `<20270000`. This is still a generous ~3-year window from current versions but provides meaningful breaking-change protection. Then regenerate the lock file with `uv lock`.

**Files involved:**
- `pyproject.toml`
- `uv.lock` (regenerated, not hand-edited)

**Verification:**
- `uv lock` succeeds without resolution errors.
- `uv run pytest` — still passes.

---

## Batch grouping

Tasks 2 and 3 both touch `pyproject.toml`. Per the spec ("If two tasks touch the same file or have any dependency, they must be in the same batch"), they go together. Task 1 touches different files and has no dependency on the others — it can run in parallel.

### Batch A — pyproject.toml cleanups (sequential within batch)

1. **Task 2** — Update description text (HEAD `via GPT` → `via Claude`).
2. **Task 3** — Tighten openai-whisper upper bound + `uv lock`.

Verification at end of batch:
- `cat pyproject.toml` — both edits present.
- `uv lock` clean.
- `uv run pytest` — 43 tests pass.

### Batch B — Shorts URL support

1. **Task 1** — Add shorts regex pattern + tests.

Verification at end of batch:
- `uv run pytest tests/test_downloader.py` — new tests pass, existing still pass.
- `uv run ruff check .` — clean.

---

## Summary

| Batch | Tasks | Files | Risk |
|-------|-------|-------|------|
| A | 2, 3 | `pyproject.toml`, `uv.lock` | Low — text change + version bound tightening |
| B | 1 | `any_video/downloader.py`, `tests/test_downloader.py` | Low — additive regex + test |

Both batches are independent and can run in parallel. Combined estimated effort: < 10 minutes.
