"""Command-line interface for any-video."""

import argparse
import os
import sys
from pathlib import Path

from any_video.config import (
    DEFAULT_OUTPUT_DIR,
    GPT_MODEL,
    GPT_MODEL_ADVANCED,
    logger,
    setup_logging,
)
from any_video.downloader import extract_video_id, get_video_title, slugify
from any_video.exceptions import APIError, DownloadError, TranscriptionError
from any_video.pipeline import process_video


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos and generate summaries with quizzes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"
  %(prog)s "https://youtu.be/VIDEO_ID" --model large-v3
  %(prog)s "https://youtube.com/shorts/VIDEO_ID" --output-dir ./results
        """,
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--model",
        choices=["tiny", "small", "large-v3"],
        default="small",
        help="Whisper model to use (default: small)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with debug information",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the downloaded audio file instead of deleting it",
    )
    parser.add_argument(
        "--gpt-model",
        default=None,
        help=f"GPT model for transcript beautification (default: {GPT_MODEL})",
    )
    parser.add_argument(
        "--gpt-model-advanced",
        default=None,
        help=f"GPT model for summary/quiz generation (default: {GPT_MODEL_ADVANCED})",
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging(verbose=args.verbose)

    try:
        # Check for API key early
        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set.")
            logger.error("Export it with: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)

        logger.info(f"Processing: {args.url}")

        # Extract video info for output directory naming
        video_id = extract_video_id(args.url)
        video_title = get_video_title(args.url)
        folder_name = f"{video_id}_{slugify(video_title)}"
        output_path = args.output_dir / folder_name

        # Run the pipeline, passing pre-computed info to avoid duplicate work
        result = process_video(
            args.url,
            args.model,
            work_dir=output_path,
            gpt_model=args.gpt_model,
            gpt_model_advanced=args.gpt_model_advanced,
            video_id=video_id,
            video_title=video_title,
        )

        # Save raw transcript for reference
        raw_transcript_file = output_path / "transcript_raw.md"
        raw_transcript_file.write_text(
            f"# Raw Transcript: {result.video_title}\n\n{result.transcript_raw}\n"
        )
        logger.debug(f"Saved raw transcript: {raw_transcript_file}")

        # Save beautified transcript
        transcript_file = output_path / "transcript.md"
        transcript_file.write_text(f"# Transcript: {result.video_title}\n\n{result.transcript}\n")
        logger.info(f"Saved: {transcript_file}")

        # Save summary
        summary_file = output_path / "summary.md"
        summary_file.write_text(f"# Summary: {result.video_title}\n\n{result.summary}\n")
        logger.info(f"Saved: {summary_file}")

        # Save quiz
        quiz_file = output_path / "quiz.md"
        quiz_file.write_text(f"# Quiz: {result.video_title}\n\n{result.quiz}\n")
        logger.info(f"Saved: {quiz_file}")

        # Clean up audio file (unless user wants to keep it)
        if args.keep_audio:
            logger.info(f"Audio file kept: {result.audio_path}")
        elif result.audio_path and result.audio_path.exists():
            result.audio_path.unlink()
            logger.debug("Cleaned up temporary audio file")

        logger.info(f"\nDone! Files saved to: {output_path}")

    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except DownloadError as e:
        logger.error(f"Download error: {e}")
        sys.exit(1)
    except TranscriptionError as e:
        logger.error(f"Transcription error: {e}")
        sys.exit(1)
    except APIError as e:
        logger.error(f"API error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user.")
        sys.exit(130)
