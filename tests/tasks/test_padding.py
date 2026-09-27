from __future__ import annotations

import re

from harness.tasks.generator import ATTRIBUTE, generate
from harness.tasks.padding import FILLER, pad
from harness.tasks.suite import SuiteSpec, build_suite
from harness.tokens import approx_token_count
from tests.tasks.specs import spec

FACT = "The access code for site_000_01 is 4271."


def test_zero_target_leaves_the_content_unchanged() -> None:
    assert pad(FACT, 0, "k") == FACT


def test_a_target_already_met_leaves_the_content_unchanged() -> None:
    assert pad(FACT, approx_token_count(FACT), "k") == FACT


def test_padding_reaches_the_target_and_keeps_the_fact_whole() -> None:
    # Act
    document = pad(FACT, 300, "k")

    # Assert
    assert approx_token_count(document) >= 300
    assert FACT in document


def test_padding_is_deterministic_in_its_key() -> None:
    assert pad(FACT, 300, "k") == pad(FACT, 300, "k")
    assert pad(FACT, 300, "k") != pad(FACT, 300, "other")


def test_filler_is_inert() -> None:
    """No digits, no identifier-shaped names, never the attribute the goal asks about."""
    for sentence in FILLER:
        assert not re.search(r"\d|_", sentence)
        assert ATTRIBUTE not in sentence
        assert sentence.isascii()


def test_padded_task_keeps_its_codes_entities_and_answer() -> None:
    """Padding changes length only: the task itself is the unpadded task."""
    # Arrange
    plain = generate(spec())
    padded = generate(spec(document_tokens=200))

    # Assert
    assert padded.answer == plain.answer
    assert padded.required == plain.required
    assert padded.distractors == plain.distractors
    for (name, text), (plain_name, plain_text) in zip(
        padded.documents, plain.documents, strict=True
    ):
        assert name == plain_name
        assert plain_text in text
        assert approx_token_count(text) >= 200


def test_the_suite_threads_document_tokens_into_every_task() -> None:
    suite = SuiteSpec(
        count=2,
        seed=1,
        chain_length=2,
        recall_distance=3,
        distractor_density=0.0,
        horizon=8,
        document_tokens=50,
    )
    assert {task.document_tokens for task in build_suite(suite)} == {50}
