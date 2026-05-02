"""Tests for anthropic_client module."""

from collections import Counter
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from any_video.anthropic_client import (
    _parse_answer_key,
    _parse_questions,
    _permute_options,
    _plan_positions,
    _shuffle_quiz,
    beautify_transcript,
    generate_gems,
    generate_quiz,
    generate_summary,
)
from any_video.config import AnthropicError


def _mock_message(content: str) -> MagicMock:
    """Create a mock Anthropic Message response with a single text block."""
    block = MagicMock()
    block.type = "text"
    block.text = content
    response = MagicMock()
    response.content = [block]
    return response


def _stream_returning(message: MagicMock) -> MagicMock:
    """Create a mock context manager that yields a stream returning `message`."""
    stream = MagicMock()
    stream.get_final_message.return_value = message
    cm = MagicMock()
    cm.__enter__.return_value = stream
    cm.__exit__.return_value = False
    return cm


class TestBeautifyTranscript:
    @patch("any_video.anthropic_client._get_client")
    def test_beautifies_text(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.return_value = _stream_returning(_mock_message("Clean text."))

        assert beautify_transcript("messy text") == "Clean text."
        mock_client.messages.stream.assert_called_once()


class TestGenerateSummary:
    @patch("any_video.anthropic_client._get_client")
    def test_generates_summary(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.return_value = _stream_returning(
            _mock_message("## Summary\nKey points.")
        )

        assert generate_summary("Some transcript text.") == "## Summary\nKey points."


class TestGenerateGems:
    @patch("any_video.anthropic_client._get_client")
    def test_generates_gems(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.return_value = _stream_returning(
            _mock_message("## A claim\n\nArgument here.\n\n> Quote.")
        )

        result = generate_gems("Some transcript text.")
        assert result == "## A claim\n\nArgument here.\n\n> Quote."


class TestGenerateQuiz:
    @patch("any_video.anthropic_client._get_client")
    def test_generates_quiz(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.return_value = _stream_returning(
            _mock_message("## Question 1\nQ: What?")
        )

        assert generate_quiz("Some transcript text.", "Some gems.") == "## Question 1\nQ: What?"

    @patch("any_video.anthropic_client._get_client")
    def test_passes_transcript_and_gems_to_user_content(self, mock_get_client):
        mock_client = mock_get_client.return_value
        # Quiz output gets passed through _shuffle_quiz; if parsing fails (e.g. on
        # the placeholder "..." used elsewhere), the raw text is returned unchanged.
        mock_client.messages.stream.return_value = _stream_returning(_mock_message("..."))

        generate_quiz("transcript body", "gems body")

        call_kwargs = mock_client.messages.stream.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "transcript body" in user_content
        assert "GEMS:" in user_content
        assert "gems body" in user_content


_SAMPLE_QUIZ = """\
## Question 1

**Q: First?**

- A) one
- B) two
- C) three
- D) four

---

## Question 2

**Q: Second?**

- A) alpha
- B) beta
- C) gamma
- D) delta

---

# Answers

1. B
2. D
"""


class TestPlanPositions:
    def test_each_letter_appears_at_least_twice_for_n10(self):
        for _ in range(50):
            plan = _plan_positions(10)
            assert len(plan) == 10
            counts = Counter(plan)
            for letter in "ABCD":
                assert counts[letter] >= 2
                assert counts[letter] <= 3

    def test_handles_small_n(self):
        assert len(_plan_positions(0)) == 0
        assert len(_plan_positions(1)) == 1
        assert len(_plan_positions(4)) == 4
        # n=4: each letter exactly once
        assert sorted(_plan_positions(4)) == ["A", "B", "C", "D"]


class TestPermuteOptions:
    def test_correct_content_lands_at_target(self):
        opts = {"A": "right", "B": "wrong1", "C": "wrong2", "D": "wrong3"}
        result = _permute_options(opts, "A", "C")
        assert result["C"] == "right"
        assert set(result.values()) == {"right", "wrong1", "wrong2", "wrong3"}

    def test_works_for_every_target(self):
        opts = {"A": "a", "B": "b", "C": "c", "D": "d"}
        for target in "ABCD":
            result = _permute_options(opts, "B", target)
            assert result[target] == "b"
            other_letters = [letter for letter in "ABCD" if letter != target]
            assert sorted(result[letter] for letter in other_letters) == ["a", "c", "d"]


class TestShuffleQuiz:
    def test_preserves_question_content(self):
        shuffled = _shuffle_quiz(_SAMPLE_QUIZ)
        questions = _parse_questions(shuffled)
        assert len(questions) == 2
        # All original option content is still present, just possibly at new letters
        assert set(questions[0].options.values()) == {"one", "two", "three", "four"}
        assert set(questions[1].options.values()) == {"alpha", "beta", "gamma", "delta"}

    def test_preserves_correct_answer_content(self):
        shuffled = _shuffle_quiz(_SAMPLE_QUIZ)
        questions = _parse_questions(shuffled)
        answers = _parse_answer_key(shuffled)
        # Q1 correct content was "two" (B in original); whatever letter it now sits at,
        # that letter must be the new answer.
        assert questions[0].options[answers[0]] == "two"
        assert questions[1].options[answers[1]] == "delta"

    def test_evens_distribution_over_many_runs(self):
        """A 10-question quiz where the model put everything at B should come out
        roughly uniform after shuffling."""
        biased = _build_biased_quiz(num_questions=10, biased_letter="B")
        seen: Counter[str] = Counter()
        for _ in range(50):
            answers = _parse_answer_key(_shuffle_quiz(biased))
            for letter in answers:
                seen[letter] += 1
        # 50 runs * 10 questions = 500 answers. Uniform expectation = 125 per letter.
        # Every letter should appear comfortably (the plan guarantees ≥2 per run, so ≥100).
        for letter in "ABCD":
            assert seen[letter] >= 100, f"letter {letter} appeared only {seen[letter]} times"

    def test_returns_unchanged_on_malformed_input(self):
        garbage = "this is not a quiz at all"
        assert _shuffle_quiz(garbage) == garbage

    def test_returns_unchanged_when_question_count_mismatches_answer_key(self):
        broken = _SAMPLE_QUIZ.replace("2. D\n", "")  # 2 questions but only 1 answer
        assert _shuffle_quiz(broken) == broken


def _build_biased_quiz(num_questions: int, biased_letter: str) -> str:
    """Build a quiz where every correct answer is at `biased_letter` — to simulate
    the model's bias and verify shuffling spreads correctness uniformly."""
    parts: list[str] = []
    for n in range(1, num_questions + 1):
        parts.append(f"## Question {n}\n\n")
        parts.append(f"**Q: Q{n}?**\n\n")
        for letter in "ABCD":
            marker = "RIGHT" if letter == biased_letter else f"wrong-{letter}"
            parts.append(f"- {letter}) {marker}-Q{n}\n")
        parts.append("\n---\n\n")
    parts.append("# Answers\n\n")
    for n in range(1, num_questions + 1):
        parts.append(f"{n}. {biased_letter}\n")
    return "".join(parts)


class TestErrorHandling:
    @patch("any_video.anthropic_client._get_client")
    def test_wraps_api_error(self, mock_get_client):
        mock_client = mock_get_client.return_value
        mock_client.messages.stream.side_effect = anthropic.APIError(
            "boom", request=MagicMock(), body=None
        )

        with pytest.raises(AnthropicError, match="Anthropic API error"):
            generate_summary("Some text.")

    @patch("any_video.anthropic_client._get_client")
    def test_raises_on_no_text_blocks(self, mock_get_client):
        mock_client = mock_get_client.return_value
        empty = MagicMock()
        empty.content = []
        mock_client.messages.stream.return_value = _stream_returning(empty)

        with pytest.raises(AnthropicError, match="no text content"):
            generate_summary("Some text.")
