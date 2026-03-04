"""Configuration, logging, and utility decorators for any-video."""

import logging
import os
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TypeVar

# Configuration
WHISPER_MODELS_DIR = Path.home() / "whisper"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "output"
GPT_MODEL = os.environ.get("ANY_VIDEO_GPT_MODEL", "gpt-4.1")
GPT_MODEL_ADVANCED = os.environ.get("ANY_VIDEO_GPT_MODEL_ADVANCED", "gpt-5.2")
MAX_TRANSCRIPT_CHARS = 100_000
BEAUTIFY_CHUNK_SIZE = 50_000

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Base delay in seconds (will be multiplied for exponential backoff)

# Set up logging
logger = logging.getLogger("any-video")

T = TypeVar("T")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    if logger.handlers:
        return
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_DELAY,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries (doubles each attempt).
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed.")
            raise last_exception

        return wrapper

    return decorator
