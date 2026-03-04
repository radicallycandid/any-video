"""Custom exceptions for the any-video tool."""


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""

    pass


class APIError(Exception):
    """Raised when OpenAI API calls fail."""

    pass


class DownloadError(Exception):
    """Raised when video download fails."""

    pass
