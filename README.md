# any-video

A YouTube video transcriber that generates AI-powered summaries and quizzes.

## Features

- **Local Transcription**: Uses OpenAI's Whisper model running locally (no API costs for transcription)
- **Transcript Beautification**: AI-powered cleanup of raw transcripts - fixes typos, corrects proper nouns, adds paragraph breaks
- **AI Summaries**: Generates concise summaries using Claude Sonnet 4.6
- **Quiz Generation**: Creates 10-question multiple-choice quizzes for learning reinforcement
- **Multiple Whisper Models**: Choose between `tiny`, `small`, `medium`, or `large-v3` based on your accuracy/speed needs
- **Idempotent with Resume**: Re-running on a URL skips work that's already been done. If a previous run failed mid-pipeline, the next run picks up from where it left off (raw transcript, summary, etc.). Use `--force` to start over.
- **Robust Error Handling**: The Anthropic SDK automatically retries transient errors (rate limits, 5xx) with exponential backoff
- **Verbose Mode**: Debug logging with `-v` flag for troubleshooting

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- **ffmpeg** (required for audio processing)
- Anthropic API key (for summary, quiz, and transcript beautification)

### Installing ffmpeg

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
```bash
winget install ffmpeg
# Or download from https://ffmpeg.org/download.html
```

## Quick Start

```bash
# Clone and enter the repo
git clone https://github.com/radicallycandid/any-video.git
cd any-video

# Set your Anthropic API key
export ANTHROPIC_API_KEY='your-key-here'

# Run it (uv handles everything automatically)
uv run any-video "https://www.youtube.com/watch?v=VIDEO_ID"
```

That's it. No manual venv creation, no `pip install` - `uv run` does it all.

## Usage

```bash
# Basic usage
uv run any-video "https://www.youtube.com/watch?v=VIDEO_ID"

# Use a larger model for better accuracy
uv run any-video "https://youtu.be/VIDEO_ID" --model large-v3

# Specify a custom output directory
uv run any-video "https://youtube.com/shorts/VIDEO_ID" --output-dir ./my-results

# Verbose mode for debugging
uv run any-video "https://www.youtube.com/watch?v=VIDEO_ID" -v
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `url` | YouTube video URL (required) | - |
| `--model` | Whisper model: `tiny`, `small`, `medium`, `large-v3` | `small` |
| `--output-dir` | Output directory for generated files | `./output` |
| `--keep-audio` | Keep the downloaded audio file | off |
| `--force` | Re-process even if output already exists | off |
| `-v, --verbose` | Enable debug output | off |

### Supported URL Formats

- Standard: `https://www.youtube.com/watch?v=VIDEO_ID`
- Shortened: `https://youtu.be/VIDEO_ID`
- Embeds: `https://www.youtube.com/embed/VIDEO_ID`
- Shorts: `https://youtube.com/shorts/VIDEO_ID`

## Output

For each video, the tool creates a folder with four markdown files:

```
output/
└── VIDEO_ID_video-title-slug/
    ├── transcript.md       # Beautified, readable transcript
    ├── transcript_raw.md   # Original Whisper output (for reference)
    ├── summary.md          # AI-generated summary
    ├── quiz.md             # 10 multiple-choice questions
    └── audio.mp3           # Only with --keep-audio
```

## Whisper Models

Whisper models are downloaded automatically on first use to `~/whisper/`. The initial download may take a few minutes depending on model size.

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | 75 MB | Fastest | Good for simple content |
| `small` | 483 MB | Balanced | Good for most videos |
| `medium` | 1.5 GB | Slower | Better than `small` |
| `large-v3` | 3 GB | Slowest | Best accuracy |

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

### Alternative: pip installation

If you prefer not to use `uv`:

```bash
pip install -e ".[dev]"
any-video "https://www.youtube.com/watch?v=VIDEO_ID"
```

## License

MIT
