"""Anthropic API interactions: beautify, summarize, gems, quiz."""

import logging
import random
import re
from dataclasses import dataclass

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
- The three distractors are plausible — they reflect real misconceptions or partial \
understanding, not absurd or off-topic choices
- All four options are matched in length — aim for the same word count within ±20%. \
The correct answer MUST NOT be the longest of the four. If your draft makes the correct \
answer the longest or most detailed, rewrite the distractors to be equally specific, or \
shorten the correct answer
- Options share grammatical structure (all noun phrases, or all complete sentences, etc.)

Note: option positions are reshuffled programmatically after generation to enforce a uniform \
correct-answer distribution, so do not worry about placing correct answers at varied letters \
yourself — focus on writing genuinely good questions and equally plausible distractors.

Explicitly guard against a well-known LLM failure mode: making the correct answer the \
longest, most specific, or most sophisticated-sounding of the four while distractors are \
shorter or flatter. A reader who has not seen the video must not be able to identify the \
correct answer simply by picking the most detailed option.

Format each question as:

## Question N

**Q: [question text]**

- A) [option]
- B) [option]
- C) [option]
- D) [option]

---

After all 10 questions, output an answer key section so the reader can attempt the \
quiz before checking. Format it exactly as:

# Answers

1. [letter]
2. [letter]
3. [letter]
4. [letter]
5. [letter]
6. [letter]
7. [letter]
8. [letter]
9. [letter]
10. [letter]

Output only the 10 questions followed by the answer key. No preamble, no closing note."""


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
    """Generate 10 multiple-choice questions, prioritizing the gems.

    The model has a strong middle-position bias for correct answers (B/C dominate
    even with explicit instruction). After generation we re-render with option
    positions shuffled to a planned uniform distribution.
    """
    logger.debug("Generating quiz...")
    user_content = f"{transcript}\n\n---\nGEMS:\n\n{gems}"
    raw = _call_claude(_QUIZ_SYSTEM_PROMPT, user_content, effort="medium")
    return _shuffle_quiz(raw)


# --- Quiz shuffling: parse → permute correct-answer positions → re-render -----


@dataclass
class _Question:
    num: int
    body: str  # e.g. "**Q: text**"
    options: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}


def _parse_questions(md: str) -> list[_Question]:
    """Parse questions from quiz markdown. Returns empty list on parse failure."""
    body = md.split("# Answers", 1)[0] if "# Answers" in md else md
    chunks = re.split(r"^##\s+Question\s+(\d+)\s*\n", body, flags=re.MULTILINE)

    questions: list[_Question] = []
    for i in range(1, len(chunks), 2):
        num_str = chunks[i]
        content = chunks[i + 1]
        opt_a = re.search(r"^-\s+A\)\s*(.+)$", content, re.MULTILINE)
        if not opt_a:
            continue
        body_text = content[: opt_a.start()].strip()
        options: dict[str, str] = {}
        for letter in "ABCD":
            m = re.search(rf"^-\s+{letter}\)\s*(.+)$", content, re.MULTILINE)
            if m:
                options[letter] = m.group(1).strip()
        if len(options) == 4 and body_text:
            questions.append(_Question(num=int(num_str), body=body_text, options=options))
    return questions


def _parse_answer_key(md: str) -> list[str]:
    """Parse the answer key. Returns empty list if no answer section found."""
    if "# Answers" not in md:
        return []
    answers_section = md.split("# Answers", 1)[1]
    return re.findall(r"^\s*\d+\.\s*([A-D])\s*$", answers_section, re.MULTILINE)


def _plan_positions(n: int) -> list[str]:
    """Return n target letters with each letter appearing as evenly as possible."""
    letters = ["A", "B", "C", "D"]
    base, extra = divmod(n, 4)
    plan = letters * base
    plan.extend(random.sample(letters, extra))
    random.shuffle(plan)
    return plan


def _permute_options(
    options: dict[str, str], correct: str, target: str
) -> dict[str, str]:
    """Move the correct option's content to `target` letter; shuffle distractors."""
    correct_content = options[correct]
    distractors = [v for k, v in options.items() if k != correct]
    random.shuffle(distractors)
    new: dict[str, str] = {}
    di = iter(distractors)
    for letter in "ABCD":
        new[letter] = correct_content if letter == target else next(di)
    return new


def _render_quiz(questions: list[_Question], answer_letters: list[str]) -> str:
    parts: list[str] = []
    for q in questions:
        parts.append(f"## Question {q.num}\n\n")
        parts.append(f"{q.body}\n\n")
        for letter in "ABCD":
            parts.append(f"- {letter}) {q.options[letter]}\n")
        parts.append("\n---\n\n")
    parts.append("# Answers\n\n")
    for i, letter in enumerate(answer_letters, 1):
        parts.append(f"{i}. {letter}\n")
    return "".join(parts).rstrip() + "\n"


def _shuffle_quiz(raw_quiz: str) -> str:
    """Permute correct-answer positions to a uniform distribution across A/B/C/D."""
    questions = _parse_questions(raw_quiz)
    correct_letters = _parse_answer_key(raw_quiz)
    if not questions or len(questions) != len(correct_letters):
        logger.warning("Could not parse quiz output for shuffling; returning unchanged")
        return raw_quiz

    plan = _plan_positions(len(questions))
    new_questions: list[_Question] = []
    new_answers: list[str] = []
    for q, orig, target in zip(questions, correct_letters, plan, strict=True):
        new_opts = _permute_options(q.options, orig, target)
        new_questions.append(_Question(num=q.num, body=q.body, options=new_opts))
        new_answers.append(target)
    return _render_quiz(new_questions, new_answers)
