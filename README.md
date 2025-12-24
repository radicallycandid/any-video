# any-video

A YouTube video transcriber that generates AI-powered summaries and quizzes.

## Features

- **Local Transcription**: Uses OpenAI's Whisper model running locally (no API costs for transcription)
- **Transcript Beautification**: AI-powered cleanup of raw transcripts - fixes typos, corrects proper nouns, adds paragraph breaks
- **AI Summaries**: Generates concise summaries using GPT-4.1
- **Quiz Generation**: Creates 10-question multiple-choice quizzes for learning reinforcement
- **Multiple Whisper Models**: Choose between `tiny`, `small`, or `large-v3` based on your accuracy/speed needs
- **Robust Error Handling**: Automatic retries with exponential backoff for API failures
- **Verbose Mode**: Debug logging with `-v` flag for troubleshooting

## Requirements

- Python 3.10+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading YouTube audio
- OpenAI API key (for summary and quiz generation)
- Whisper model files in `~/whisper/`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/any-video.git
   cd any-video
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

   Or with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Download Whisper models to `~/whisper/`:
   ```bash
   mkdir -p ~/whisper
   # Models will be downloaded automatically on first use, or you can pre-download them
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

## Usage

Basic usage:
```bash
python any_video.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

With options:
```bash
# Use a larger model for better accuracy
python any_video.py "https://youtu.be/VIDEO_ID" --model large-v3

# Specify a custom output directory
python any_video.py "https://youtube.com/shorts/VIDEO_ID" --output-dir ./my-results
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `url` | YouTube video URL (required) | - |
| `--model` | Whisper model: `tiny`, `small`, `large-v3` | `small` |
| `--output-dir` | Output directory for generated files | `./output` |
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
    └── quiz.md             # 10 multiple-choice questions
```

## Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | 75 MB | Fastest | Good for simple content |
| `small` | 483 MB | Balanced | Good for most videos |
| `large-v3` | 3 GB | Slowest | Best accuracy |

## Development

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest -v
```

Format code:
```bash
black any_video.py tests/
```

Lint:
```bash
ruff check any_video.py tests/
```

## License

MIT
