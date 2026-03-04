"""OpenAI API integration for transcript processing."""

import os
from concurrent.futures import ThreadPoolExecutor

import openai

from any_video.config import (
    BEAUTIFY_CHUNK_SIZE,
    GPT_MODEL,
    GPT_MODEL_ADVANCED,
    MAX_RETRIES,
    MAX_TRANSCRIPT_CHARS,
    RETRY_DELAY,
    logger,
    retry_with_backoff,
)
from any_video.exceptions import APIError


def _get_openai_client() -> openai.OpenAI:
    """Create an OpenAI client, validating the API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise APIError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it with: export OPENAI_API_KEY='your-key-here'"
        )
    return openai.OpenAI()


@retry_with_backoff(
    max_retries=MAX_RETRIES,
    base_delay=RETRY_DELAY,
    exceptions=(openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
)
def _raw_openai_call(client: openai.OpenAI, messages: list, max_tokens: int, model: str) -> str:
    """Make a raw OpenAI API call with automatic retry on transient errors."""
    use_new_param = any(model.startswith(p) for p in ("gpt-4.1", "gpt-5", "o1", "o3"))
    token_param = (
        {"max_completion_tokens": max_tokens} if use_new_param else {"max_tokens": max_tokens}
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **token_param,
    )
    return response.choices[0].message.content


def _call_openai_api(messages: list, max_tokens: int, model: str | None = None) -> str:
    """
    Make an OpenAI API call with retry logic and error translation.

    All OpenAI-specific exceptions are caught and re-raised as APIError,
    providing a single place for error handling across the entire module.
    """
    client = _get_openai_client()
    actual_model = model or GPT_MODEL
    try:
        return _raw_openai_call(client, messages, max_tokens, actual_model)
    except openai.AuthenticationError as e:
        raise APIError("Invalid OpenAI API key. Please check your OPENAI_API_KEY.") from e
    except openai.RateLimitError as e:
        raise APIError(
            "OpenAI API rate limit exceeded after multiple retries. Please try again later."
        ) from e
    except openai.APIConnectionError as e:
        raise APIError(
            "Failed to connect to OpenAI API after multiple retries. Check your internet connection."
        ) from e
    except openai.APITimeoutError as e:
        raise APIError(
            "OpenAI API request timed out after multiple retries. Please try again later."
        ) from e
    except openai.APIError as e:
        raise APIError(f"OpenAI API error: {e}") from e


# --- Transcript processing ---

BEAUTIFY_SYSTEM_PROMPT = """You are an expert transcript editor. Your job is to clean up raw speech-to-text transcripts while preserving the original meaning exactly.

Your tasks:
1. Fix obvious transcription errors and typos
2. Correct proper nouns (people's names, company names, technical terms) based on context
3. Add appropriate paragraph breaks for readability (every 3-5 sentences or at topic changes)
4. Fix punctuation and capitalization
5. Remove filler words like "um", "uh", "you know" (but keep natural speech patterns)
6. Do NOT add, remove, or change the actual content or meaning
7. Do NOT add summaries, headers, or commentary
8. Do NOT use markdown formatting except for paragraph breaks

Return ONLY the cleaned transcript text, nothing else."""


def _split_into_chunks(text: str, chunk_size: int = BEAUTIFY_CHUNK_SIZE) -> list[str]:
    """Split text into chunks at sentence boundaries for processing."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break
        candidate = remaining[:chunk_size]
        last_period = candidate.rfind(". ")
        split_at = (last_period + 2) if last_period > chunk_size * 0.8 else chunk_size
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return chunks


def truncate_transcript(transcript: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> tuple[str, bool]:
    """
    Truncate transcript if it exceeds the maximum length.

    Args:
        transcript: The full transcript text.
        max_chars: Maximum number of characters allowed.

    Returns:
        A tuple of (possibly truncated transcript, was_truncated boolean).
    """
    if len(transcript) <= max_chars:
        return transcript, False

    # Try to truncate at a sentence boundary
    truncated = transcript[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.8:  # Only use if we're not losing too much
        truncated = truncated[: last_period + 1]

    return truncated, True


def beautify_transcript(raw_transcript: str, video_title: str, model: str | None = None) -> str:
    """
    Clean up and format raw Whisper transcript using an LLM.

    Processes long transcripts in chunks to avoid content loss from truncation.

    Args:
        raw_transcript: The raw transcript from Whisper.
        video_title: The video title for context.
        model: Optional model override. Defaults to GPT_MODEL.

    Returns:
        The beautified transcript text.

    Raises:
        APIError: If the API call fails.
    """
    chunks = _split_into_chunks(raw_transcript)
    if len(chunks) > 1:
        logger.info(f"Processing transcript in {len(chunks)} chunks...")

    beautified_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(f"Beautifying chunk {i + 1}/{len(chunks)}...")
        else:
            logger.info("Beautifying transcript...")

        beautified = _call_openai_api(
            [
                {"role": "system", "content": BEAUTIFY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f'Please clean up this transcript from the video "{video_title}":'
                        f"\n\n{chunk}"
                    ),
                },
            ],
            max_tokens=16000,
            model=model or GPT_MODEL,
        )
        beautified_chunks.append(beautified)

    return "\n\n".join(beautified_chunks)


def generate_summary_and_quiz(
    transcript: str, video_title: str, model: str | None = None
) -> tuple[str, str]:
    """
    Generate summary and quiz from a transcript, in parallel.

    Args:
        transcript: The video transcript text.
        video_title: The title of the video.
        model: Optional model override. Defaults to GPT_MODEL_ADVANCED.

    Returns:
        A tuple of (summary, quiz) as strings.

    Raises:
        APIError: If the API calls fail.
    """
    # Truncate transcript if needed
    transcript_for_api, was_truncated = truncate_transcript(transcript)
    if was_truncated:
        logger.warning(
            f"Transcript was truncated from {len(transcript):,} to "
            f"{len(transcript_for_api):,} characters to fit API limits."
        )

    actual_model = model or GPT_MODEL_ADVANCED

    summary_messages = [
        {
            "role": "system",
            "content": """You are an expert content summarizer. Your summaries should be direct and opinionated, capturing the actual claims, arguments, and opinions expressed in the content.

Key guidelines:
- Use direct, assertive language: "X argues that..." or "X states that..." rather than "X reflects on..." or "X discusses..."
- Capture the speaker's actual positions and opinions, not just topics covered
- Include specific claims, arguments, and conclusions made by the speaker
- Present the substance of what was said, not a meta-description of the video
- Be concise but substantive - every sentence should convey meaningful content
- If the speaker expresses strong opinions or makes bold claims, include them directly""",
        },
        {
            "role": "user",
            "content": f"""Summarize the key points, arguments, and opinions from this video transcript.
Video title: "{video_title}"

Transcript:
{transcript_for_api}""",
        },
    ]

    quiz_messages = [
        {
            "role": "system",
            "content": """You are an expert quiz creator. Create high-quality multiple choice questions that genuinely test comprehension.

Critical requirements for answer options:
1. ALL four options (A, B, C, D) must be similar in length, tone, and level of detail
2. Wrong answers must sound equally plausible and professional as the correct answer
3. Avoid making the correct answer longer, more detailed, or more "polished" than wrong answers
4. Distribute correct answers roughly evenly across A, B, C, and D (not clustering on any letter)

Question quality guidelines:
- Target MEDIUM difficulty - not trivially obvious, but answerable if one paid attention
- Questions should require actual comprehension, not just keyword matching
- Wrong answers should be reasonable interpretations that someone might believe if they misunderstood
- Avoid "all of the above" or "none of the above" options
- Each question should test a distinct concept or claim from the content""",
        },
        {
            "role": "user",
            "content": f"""Create a 10-question multiple choice quiz based on this video transcript.
Video title: "{video_title}"

Transcript:
{transcript_for_api}

Format each question exactly like this:

## Question 1
[Question text]

- A) [Option - similar length and tone to others]
- B) [Option - similar length and tone to others]
- C) [Option - similar length and tone to others]
- D) [Option - similar length and tone to others]

**Correct Answer: [Letter]**

---

Remember: Vary which letter is correct across questions, and ensure all options look equally plausible.""",
        },
    ]

    logger.info("Generating summary and quiz...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        summary_future = executor.submit(_call_openai_api, summary_messages, 4000, actual_model)
        quiz_future = executor.submit(_call_openai_api, quiz_messages, 3000, actual_model)
        summary = summary_future.result()
        quiz = quiz_future.result()

    return summary, quiz
