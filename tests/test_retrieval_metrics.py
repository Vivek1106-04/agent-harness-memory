from __future__ import annotations

import pytest

from harness.retrieval_metrics import RetrievalTally

FACT_A = "code for a_1 is 1111."
FACT_B = "code for b_1 is 2222."
NOISE = "code for z_9 is 9999."


def test_steps_before_any_required_fact_is_seen_are_not_scored() -> None:
    # Arrange
    tally = RetrievalTally([FACT_A])

    # Act
    tally.observe(history=[NOISE], retrieved=[NOISE])

    # Assert
    assert tally.precision is None
    assert tally.recall is None


def test_perfect_retrieval_scores_one_on_both() -> None:
    # Arrange
    tally = RetrievalTally([FACT_A, FACT_B])

    # Act
    tally.observe(history=[FACT_A, NOISE], retrieved=[FACT_A])

    # Assert
    assert tally.precision == pytest.approx(1.0)
    assert tally.recall == pytest.approx(1.0)


def test_partial_retrieval_is_averaged_across_steps() -> None:
    # Arrange
    tally = RetrievalTally([FACT_A, FACT_B])

    # Act
    tally.observe(history=[FACT_A, FACT_B], retrieved=[FACT_A, NOISE])
    tally.observe(history=[FACT_A, FACT_B], retrieved=[FACT_A, FACT_B])

    # Assert
    assert tally.precision == pytest.approx((0.5 + 1.0) / 2)
    assert tally.recall == pytest.approx((0.5 + 1.0) / 2)


def test_an_empty_retrieval_scores_recall_but_not_precision() -> None:
    """NoOp retrieves nothing: zero recall is real, precision is undefined."""
    # Arrange
    tally = RetrievalTally([FACT_A])

    # Act
    tally.observe(history=[FACT_A], retrieved=[])

    # Assert
    assert tally.precision is None
    assert tally.recall == pytest.approx(0.0)


def test_relevance_is_by_containment_so_wrapped_facts_count() -> None:
    # Arrange
    tally = RetrievalTally([FACT_A])

    # Act
    tally.observe(history=[f"doc says: {FACT_A}"], retrieved=[f"[memory] {FACT_A}"])

    # Assert
    assert tally.recall == pytest.approx(1.0)


def test_final_recall_is_the_last_scored_step_not_the_average() -> None:
    """A graph finds a fact only once the document linking it has been read. The average
    charges it for every step before that; the final step is what the answer was given on."""
    # Arrange
    tally = RetrievalTally([FACT_A])

    # Act
    tally.observe(history=[FACT_A], retrieved=[NOISE])
    tally.observe(history=[FACT_A], retrieved=[FACT_A])

    # Assert
    assert tally.recall == pytest.approx(0.5)
    assert tally.final_recall == pytest.approx(1.0)


def test_final_recall_ignores_steps_before_any_required_fact_is_seen() -> None:
    # Arrange
    tally = RetrievalTally([FACT_A])

    # Act
    tally.observe(history=[FACT_A], retrieved=[FACT_A])
    before = tally.final_recall
    empty = RetrievalTally([FACT_A])
    empty.observe(history=[NOISE], retrieved=[NOISE])

    # Assert
    assert before == pytest.approx(1.0)
    assert empty.final_recall is None
