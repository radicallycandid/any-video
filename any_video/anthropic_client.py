"""Anthropic API interactions: beautify, summarize, gems, quiz."""

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

_GEMS_SYSTEM_PROMPT = """\
You are extracting load-bearing, non-obvious claims from a transcript. Your goal is to surface \
the points the speaker is actually arguing — the most important, sincerely held, and \
counterintuitive things they said — and ground each one in the speaker's own words.

What qualifies as a gem:
- A claim only this speaker could plausibly make, rooted in their experience, role, or convictions
- A specific, concrete position — "we deleted three of our top features" beats "you have to be \
willing to cut"
- A point that pushes against an obvious framing, or names something polite conversation usually \
leaves unsaid
- A real prediction, opinion, or admission — not a hedge

What does NOT qualify:
- Generic advice any expert in the field might give
- Strong-sounding but vague generalities ("you have to really care", "everything is connected")
- Hedged or hypothetical framings ("you could argue", "some people might say")
- Restating a question rather than answering it
- For interview-format content: anything the interviewer said. If speakers aren't labeled, \
infer from context — questions and topic-shifts come from the interviewer; sustained \
explanations and opinions come from the interviewee.

Output format — for each gem:

## [Short label naming the claim]

[1–3 sentences in your own words articulating the actual argument the speaker is making. Be \
specific about what they're claiming, not just the topic. Avoid restating the quote — \
synthesize the position.]

> [Direct supporting quote from the transcript, edited only to remove disfluencies. Pick the \
line(s) where the speaker most plainly states or earns the claim.]

[Optional second supporting quote on its own > line if the argument is built on more than one \
statement.]

---

Prefer specific, concrete, even uncomfortable claims over polished or universally-agreeable \
ones. Always find gems from whoever is speaking — solo monologues qualify just as well as \
interviews. Output 3–7 gems if the transcript supports them; output fewer only if the transcript \
is genuinely brief or barren of substantive claims. Do not pad to a count, and do not include \
items that don't clear the bar."""

_QUIZ_SYSTEM_PROMPT = """\
You are creating a multiple-choice quiz to test comprehension of a video transcript.

The user message contains the full beautified transcript followed by a `GEMS:` section listing \
the most load-bearing, non-obvious claims from the speaker. At least 3 of the 10 questions \
must test understanding of the gems — those are the parts most worth retaining. Use the rest \
of the transcript for breadth.

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


def _call_claude(
    system_prompt: str,
    user_content: str,
    *,
    effort: str = "low",
    enable_thinking: bool = False,
) -> str:
    """Call Claude with streaming and return the concatenated text response."""
    client = _get_client()
    thinking = {"type": "adaptive"} if enable_thinking else {"type": "disabled"}
    try:
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=64000,
            thinking=thinking,
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


def generate_gems(transcript: str) -> str:
    """Extract load-bearing, non-obvious claims from a transcript with supporting quotes."""
    logger.debug("Extracting gems...")
    return _call_claude(_GEMS_SYSTEM_PROMPT, transcript, effort="high", enable_thinking=True)


def generate_quiz(transcript: str, gems: str) -> str:
    """Generate 10 multiple-choice questions, prioritizing the gems."""
    logger.debug("Generating quiz...")
    user_content = f"{transcript}\n\n---\nGEMS:\n\n{gems}"
    return _call_claude(_QUIZ_SYSTEM_PROMPT, user_content, effort="medium")
