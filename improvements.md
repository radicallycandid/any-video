# Improvements

## Technical Debt

No known issues.

---

## Feature Work

### Non-YouTube sources
The tool is called "any-video" but `extract_video_id` only handles YouTube URLs. yt-dlp supports hundreds of sites (Vimeo, Twitter/X, podcasts, direct MP4 URLs). Generalizing to accept any yt-dlp-compatible URL — falling back to a URL hash for folder naming — would make the tool live up to its name.

### Transcription-only mode (no API key required)
A `--transcribe-only` flag to download and transcribe locally without calling the OpenAI API. Useful for users who just need the raw transcript, want to use a different LLM, or don't want to pay for API calls.

### Timestamps in transcript
Whisper returns segment-level timestamps in `result["segments"]`, but `transcribe_audio` only keeps `result["text"]`, discarding them. Preserving timestamps would enable time-aligned transcripts (e.g., `[00:03:42] paragraph text`), SRT/VTT subtitle export, and linking back to specific moments in the video.

### Caching / idempotent reruns
Re-running the same video URL re-downloads audio, re-transcribes, and re-calls the API from scratch. Detecting an existing output folder by video ID and skipping completed steps would save significant time and API cost. A `--force` flag could override.

### Progress reporting
The pipeline can take 10+ minutes for long videos. The CLI shows static messages with no granularity, and the extension shows a spinner with hardcoded "Downloading audio..." that never updates. Adding structured progress events — ideally via Server-Sent Events for the extension — would let both interfaces show real pipeline stage.

### Language selection and auto-detection
Whisper supports a `language` parameter and automatic language detection. Exposing `--language` on the CLI and in the server request would improve accuracy for non-English content.

### Playlist / batch processing
Accept a YouTube playlist URL and process all videos in sequence, outputting a summary index file linking to each video's results.

### Speaker diarization
For interviews, podcasts, and panels, identifying distinct speakers and labeling their turns would improve transcript readability. Whisper doesn't do this natively, but `pyannote.audio` can be integrated as an optional post-processing step.

### Export formats
Currently outputs only Markdown. SRT/VTT subtitles (requires timestamps), PDF, plain text, and structured JSON (especially for the quiz, so the extension could render it interactively) would cover more use cases.

### Configurable quiz parameters
The quiz is fixed at 10 multiple-choice questions at medium difficulty. Exposing question count, difficulty, and format (true/false, open-ended, fill-in-the-blank) via CLI flags and server request would make the tool more flexible.

### Customizable prompts
Let users provide their own system prompts for summarization, beautification, or quiz generation — via config files or CLI flags — so they can tailor outputs to their needs (academic summaries, meeting notes, casual recaps).
