from __future__ import annotations

from harness.components.evaluator_checker import (
    REASON_NO_ANSWER,
    REASON_SUCCESS,
    REASON_WRONG,
    CheckerEvaluator,
)
from harness.interfaces import Evaluator
from harness.registry import resolve
from harness.tasks.generator import generate
from harness.types import Observation
from tests.contracts.evaluator import EvaluatorContract
from tests.tasks.specs import spec


def _build() -> Evaluator:
    return CheckerEvaluator()


def _read(step: int, content: str) -> Observation:
    return Observation(step=step, content=content, tool_name="read_file")


def _answer(step: int, text: str) -> Observation:
    return Observation(step=step, content=text, tool_name="answer")


class TestCheckerEvaluatorContract(EvaluatorContract):
    evaluator_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("evaluator", "checker") is CheckerEvaluator


def test_correct_answer_succeeds() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)

    # Act
    verdict = CheckerEvaluator().judge(subject, [_answer(1, task.answer)])

    # Assert
    assert verdict.success is True
    assert verdict.score == 1.0
    assert verdict.reason == REASON_SUCCESS


def test_answer_embedded_in_a_sentence_still_counts() -> None:
    """We are measuring recall, not output formatting."""
    # Arrange
    subject = spec()
    task = generate(subject)

    # Act
    verdict = CheckerEvaluator().judge(subject, [_answer(1, f"the code is {task.answer}.")])

    # Assert
    assert verdict.success is True


def test_a_near_miss_is_not_credited_as_a_match() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)

    # Act
    verdict = CheckerEvaluator().judge(subject, [_answer(1, f"{task.answer}8")])

    # Assert
    assert verdict.success is False
    assert verdict.reason == REASON_WRONG


def test_an_errored_answer_is_not_an_answer() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)
    errored = Observation(step=1, content=task.answer, tool_name="answer", error="missing argument")

    # Act
    verdict = CheckerEvaluator().judge(subject, [errored])

    # Assert
    assert verdict.success is False
    assert verdict.reason == REASON_NO_ANSWER


def test_partial_credit_tracks_the_required_facts_surfaced() -> None:
    """A near miss and a run that never found anything must not score the same."""
    # Arrange
    subject = spec()
    task = generate(subject)

    # Act
    verdict = CheckerEvaluator().judge(subject, [_read(1, task.required[0].content)])

    # Assert
    assert verdict.score == 1 / len(task.required)


def test_reading_a_distractor_earns_nothing() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)

    # Act
    verdict = CheckerEvaluator().judge(subject, [_read(1, task.distractors[0].content)])

    # Assert
    assert verdict.score == 0.0
    assert verdict.per_step == [False]


def test_per_step_credits_first_discovery_only() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)
    fact = task.required[0].content
    trace = [_read(1, fact), _read(2, fact), _answer(3, task.answer)]

    # Act
    verdict = CheckerEvaluator().judge(subject, trace)

    # Assert
    assert verdict.per_step == [True, False, True]
