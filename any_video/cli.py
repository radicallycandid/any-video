"""CLI argument parsing, entry point, and error handling."""

import argparse
import sys
from pathlib import Path

import truststore

from any_video.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WHISPER_MODEL,
    WHISPER_MODELS,
    AnyVideoError,
    setup_logging,
)
from any_video.pipeline import process

# Route TLS validation through the OS trust store so corporate proxy CAs
# installed in the system keychain are honored. No-op on systems without
# such CAs — public roots are present in OS trust stores too.
truststore.inject_into_ssl()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="any-video",
        description="Download YouTube videos, transcribe with Whisper, and generate learning materials via GPT.",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        choices=WHISPER_MODELS,
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base directory for output (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Retain the downloaded audio file in output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if output already exists",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the CLI."""
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        output_path = process(
            url=args.url,
            model_name=args.model,
            output_dir=Path(args.output_dir),
            keep_audio=args.keep_audio,
            force=args.force,
        )
        print(output_path)
    except AnyVideoError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
