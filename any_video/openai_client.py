"""OpenAI API interactions: beautify, summarize, quiz. Chunking logic and retry with backoff."""

import logging
import time

import openai

from any_video.config import GPT_MODEL, MAX_CHUNK_CHARS, MAX_RETRIES, OpenAIError

logger = logging.getLogger("any_video")

_BEAUTIFY_SYSTEM_PROMPT = """\
You are a transcript editor. Clean up the following raw transcript:
- Fix punctuation, capitalization, and grammar errors caused by speech-to-text
- Add paragraph breaks at natural topic boundaries
- Remove filler words (um, uh, like, you know) when they don't add meaning
- Preserve the original meaning and speaker's intent exactly
- Do NOT add headings, summaries, or commentary
- Output only the cleaned transcript text"""

_SUMMARY_SYSTEM_PROMPT = """\
You are a learning assistant. Create a structured summary of the following transcript.
Use markdown formatting with:
- A brief overview (2-3 sentences)
- Key topics covered (as a bulleted list)
- Main takeaways (as a numbered list)
- Any notable quotes or examples mentioned
Keep the summary concise but comprehensive."""

_QUIZ_SYSTEM_PROMPT = """\
You are a learning assistant. Based on the following transcript, create exactly 10 \
multiple-choice questions to test comprehension.

Format each question as:
## Question N

**Q: [question text]**

- A) [option]
- B) [option]
- C) [option]
- D) [option]

**Answer: [letter]**

---

Make questions that test understanding of key concepts, not trivial details."""


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        # Find last sentence boundary within limit
        split_at = remaining.rfind(". ", 0, max_chars)
        if split_at == -1:
            # No sentence boundary found, split at max_chars
            split_at = max_chars
        else:
            split_at += 2  # Include the ". "
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return chunks


def _call_openai(system_prompt: str, user_content: str) -> str:
    """Make an OpenAI API call with retry and exponential backoff."""
    client = openai.OpenAI()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return response.choices[0].message.content.strip()
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                raise OpenAIError(f"OpenAI API failed after {MAX_RETRIES} retries: {e}") from e
            wait = 2**attempt
            logger.warning(
                "OpenAI API error (attempt %d/%d), retrying in %ds...",
                attempt + 1,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
        except openai.OpenAIError as e:
            raise OpenAIError(f"OpenAI API error: {e}") from e

    raise OpenAIError("Unexpected: exhausted retries without result")  # pragma: no cover


def beautify_transcript(raw_transcript: str) -> str:
    """Beautify a raw transcript using GPT, chunking if needed."""
    chunks = _chunk_text(raw_transcript)
    logger.info("Beautifying transcript (%d chunk(s))...", len(chunks))

    beautified_parts = []
    for i, chunk in enumerate(chunks):
        logger.debug("Beautifying chunk %d/%d...", i + 1, len(chunks))
        result = _call_openai(_BEAUTIFY_SYSTEM_PROMPT, chunk)
        beautified_parts.append(result)

    return "\n\n".join(beautified_parts)


def generate_summary(transcript: str) -> str:
    """Generate a structured summary from a transcript."""
    logger.info("Generating summary...")
    return _call_openai(_SUMMARY_SYSTEM_PROMPT, transcript)


def generate_quiz(transcript: str) -> str:
    """Generate 10 multiple-choice questions from a transcript."""
    logger.info("Generating quiz...")
    return _call_openai(_QUIZ_SYSTEM_PROMPT, transcript)
