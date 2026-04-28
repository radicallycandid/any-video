"""Anthropic API interactions: beautify, summarize, quiz."""

import logging

import anthropic

from any_video.config import CLAUDE_MODEL, AnthropicError

logger = logging.getLogger("any_video")

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Return a shared Anthropic client, creating it on first use."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=5)
    return _client


_BEAUTIFY_SYSTEM_PROMPT = """\
You are a transcript editor. Your job is to turn a raw speech-to-text transcript into clean, \
readable prose while preserving exactly what the speaker said.

Apply these transformations:
- Fix punctuation, capitalization, and obvious grammatical errors caused by speech recognition
- Correct misheard proper nouns when context makes the right form unambiguous (names of people, \
products, technical terms, places)
- Remove disfluencies that don't carry meaning: filler words (um, uh, like, you know), repeated \
words, false starts (e.g. "I— I mean we should")
- Add paragraph breaks at natural topic shifts
- Keep all substantive content — every claim, example, qualifier, and aside the speaker made

Do not:
- Summarize, condense, or paraphrase
- Add headings, section titles, or commentary
- Reorder material or "improve" the speaker's argument
- Change technical or domain-specific phrasing even if it sounds informal

Output only the cleaned transcript text — no preamble, no explanation."""

_SUMMARY_SYSTEM_PROMPT = """\
You are a learning assistant. Produce a structured summary of a video transcript that a learner \
can use to review the material.

Output sections, in this order, using markdown:

## Overview
2–3 sentences capturing what the video is about and the main thesis or argument.

## Key Topics
A bulleted list of 4–8 distinct topics covered. Each bullet: one short noun phrase naming the \
topic, optionally followed by a colon and a one-line description.

## Main Takeaways
A numbered list of 3–6 concrete takeaways — what the learner should walk away knowing or being \
able to do. State them as full sentences, not topic labels.

## Notable Examples or Quotes
Include this section only if the transcript contains specific examples, case studies, or \
quotable lines worth preserving. Otherwise omit the section entirely (do not write "None"). \
Keep to 2–4 items.

Write for someone who has not seen the video. Use the speaker's own framing and terminology. \
Do not pad — if a section can be shorter, make it shorter."""

_QUIZ_SYSTEM_PROMPT = """\
You are creating a multiple-choice quiz to test comprehension of a video transcript.

Produce exactly 10 questions. Distribute them across the transcript so the quiz covers the full \
video, not just the opening. Mix question types:
- Most questions should test understanding of key concepts and arguments
- Include 2–3 questions that test application or inference (e.g. "what would the speaker likely \
say about…", "which of these examples best illustrates…")
- Avoid trivia (specific numbers, names, or dates unless they are central to the material)

For each question, write four options (A–D) where:
- Exactly one is correct
- The three distractors are plausible — they should be wrong but reflect real misconceptions or \
partial understanding, not absurd or off-topic choices
- Options are roughly the same length and grammatical form
- The position of the correct answer varies across the 10 questions (do not always put it as A or B)

Format each question as:

## Question N

**Q: [question text]**

- A) [option]
- B) [option]
- C) [option]
- D) [option]

**Answer: [letter]**

---

Output only the 10 questions in this format. No preamble, no closing note."""


def _call_claude(system_prompt: str, user_content: str, effort: str = "low") -> str:
    """Call Claude with streaming and return the concatenated text response."""
    client = _get_client()
    try:
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=64000,
            thinking={"type": "disabled"},
            output_config={"effort": effort},
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            response = stream.get_final_message()
    except anthropic.APIError as e:
        raise AnthropicError(f"Anthropic API error: {e}") from e

    text_parts = [b.text for b in response.content if b.type == "text"]
    if not text_parts:
        raise AnthropicError("Anthropic returned no text content (possible refusal)")
    return "\n".join(text_parts).strip()


def beautify_transcript(raw_transcript: str) -> str:
    """Beautify a raw transcript using Claude."""
    logger.debug("Beautifying transcript...")
    return _call_claude(_BEAUTIFY_SYSTEM_PROMPT, raw_transcript)


def generate_summary(transcript: str) -> str:
    """Generate a structured summary from a transcript."""
    logger.debug("Generating summary...")
    return _call_claude(_SUMMARY_SYSTEM_PROMPT, transcript)


def generate_quiz(transcript: str) -> str:
    """Generate 10 multiple-choice questions from a transcript."""
    logger.debug("Generating quiz...")
    return _call_claude(_QUIZ_SYSTEM_PROMPT, transcript, effort="medium")
