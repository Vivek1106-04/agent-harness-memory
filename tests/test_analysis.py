from __future__ import annotations

import math

import numpy as np
import pytest

from harness.analysis import AnalysisError, _interval, analyse, render
from harness.results import RunResult


def _row(
    condition: str,
    task: str,
    success: bool,
    seed: int = 1,
    budget: int = 8000,
    precision: float | None = None,
    recall: float | None = None,
    final: float | None = None,
) -> RunResult:
    return RunResult(
        run_id=f"p:{condition}:{budget}:{task}:s{seed}",
        task_id=task,
        condition=condition,
        token_budget=budget,
        seed=seed,
        success=success,
        score=1.0 if success else 0.0,
        tokens_read=100,
        tokens_write=10,
        steps=5,
        wall_ms=1.0,
        config_hash="h",
        git_commit="c",
        payload_version="1.0",
        model_id="m",
        model_revision="r",
        embedding_model="e",
        embedding_revision="er",
        retrieval_precision=precision,
        retrieval_recall=recall,
        retrieval_recall_final=final,
    )


def _pilot(graph: list[bool], vector: list[bool]) -> list[RunResult]:
    """Floor fails and ceiling passes every task; graph and vector as given."""
    rows: list[RunResult] = []
    for index, (g, v) in enumerate(zip(graph, vector, strict=True)):
        task = f"t{index}"
        rows += [
            _row("noop", task, False),
            _row("oracle", task, True),
            _row("graph", task, g),
            _row("vector", task, v),
        ]
    return rows


def test_success_is_the_mean_over_tasks() -> None:
    # Arrange
    rows = _pilot(graph=[True, True, False, False], vector=[False] * 4)

    # Act
    report = analyse(rows, resamples=200)

    # Assert
    graph = next(s for s in report.conditions if s.condition == "graph")
    assert graph.success.estimate == pytest.approx(0.5)
    assert graph.success.low <= 0.5 <= graph.success.high


def test_headroom_is_zero_at_the_floor_and_one_at_the_ceiling() -> None:
    # Arrange
    rows = _pilot(graph=[True, False], vector=[False, False])

    # Act
    report = analyse(rows, resamples=200)

    # Assert
    summaries = {s.condition: s for s in report.conditions}
    assert summaries["noop"].headroom is not None
    assert summaries["noop"].headroom.estimate == pytest.approx(0.0)
    assert summaries["oracle"].headroom is not None
    assert summaries["oracle"].headroom.estimate == pytest.approx(1.0)
    assert summaries["graph"].headroom is not None
    assert summaries["graph"].headroom.estimate == pytest.approx(0.5)


def test_seeds_are_averaged_within_a_task_before_tasks_are_averaged() -> None:
    """A task run twice is one observation, not two."""
    # Arrange
    rows = [
        _row("vector", "t0", True, seed=1),
        _row("vector", "t0", False, seed=2),
        _row("vector", "t1", False, seed=1),
        _row("vector", "t1", False, seed=2),
    ]

    # Act
    report = analyse(rows, resamples=50)

    # Assert
    assert report.conditions[0].success.estimate == pytest.approx(0.25)
    assert report.conditions[0].runs == 4


def test_a_task_the_floor_solves_is_struck_from_every_condition() -> None:
    # Arrange
    rows = [*_pilot(graph=[True], vector=[False]), _row("noop", "easy", True)]
    rows += [_row(c, "easy", True) for c in ("oracle", "graph", "vector")]

    # Act
    report = analyse(rows, resamples=50)

    # Assert
    assert report.struck == ("easy",)
    assert report.tasks == ("t0",)


def test_a_task_missing_from_one_condition_is_counted_not_analysed() -> None:
    # Arrange
    rows = [*_pilot(graph=[True], vector=[True]), _row("graph", "orphan", True)]

    # Act
    report = analyse(rows, resamples=50)

    # Assert
    assert report.incomplete == ("orphan",)
    assert "orphan" not in report.tasks


def test_primary_comparison_is_graph_minus_vector_paired_on_tasks() -> None:
    # Arrange
    rows = _pilot(graph=[True, True, True, False], vector=[False, True, False, False])

    # Act
    report = analyse(rows, resamples=200)

    # Assert
    assert report.primary is not None
    first, second, diff = report.primary
    assert (first, second) == ("graph", "vector")
    assert diff.estimate == pytest.approx(0.5)


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    rows = _pilot(graph=[True, False, True], vector=[False, False, True])
    assert analyse(rows, seed=7) == analyse(rows, seed=7)


def test_no_primary_and_no_headroom_without_their_conditions() -> None:
    # Arrange
    rows = [_row("episodic", "t0", True), _row("episodic", "t1", False)]

    # Act
    report = analyse(rows, resamples=50)

    # Assert
    assert report.primary is None
    assert report.conditions[0].headroom is None


def test_a_zero_width_bracket_has_no_headroom() -> None:
    """Oracle no better than NoOp: headroom is undefined, not infinite."""
    # Arrange
    rows = [_row("noop", "t0", False), _row("oracle", "t0", False), _row("graph", "t0", False)]

    # Act / Assert
    assert all(s.headroom is None for s in analyse(rows, resamples=50).conditions)


def test_retrieval_scores_average_only_the_runs_that_define_them() -> None:
    # Arrange
    rows = [
        _row("graph", "t0", True, precision=1.0, recall=0.5),
        _row("graph", "t1", True, precision=None, recall=None),
    ]

    # Act
    summary = analyse(rows, resamples=50).conditions[0]

    # Assert
    assert summary.retrieval_precision == pytest.approx(1.0)
    assert summary.retrieval_recall == pytest.approx(0.5)


def test_final_recall_averages_only_the_runs_that_define_it() -> None:
    # Arrange
    rows = [
        _row("graph", "t0", True, recall=0.1, final=1.0),
        _row("graph", "t1", True, recall=0.2, final=0.5),
        _row("graph", "t2", True, final=None),
    ]

    # Act
    summary = analyse(rows, resamples=50).conditions[0]

    # Assert
    assert summary.retrieval_recall_final == pytest.approx(0.75)


def test_rows_from_two_budgets_are_refused() -> None:
    rows = [_row("graph", "t0", True, budget=4000), _row("graph", "t0", True, budget=8000)]
    with pytest.raises(AnalysisError, match="one token budget"):
        analyse(rows)


def test_nothing_left_after_the_screen_is_an_error() -> None:
    rows = [_row("noop", "t0", True), _row("oracle", "t0", True)]
    with pytest.raises(AnalysisError, match="no task is left"):
        analyse(rows)


def test_render_marks_the_pilot_preliminary_and_lists_every_condition() -> None:
    # Arrange
    rows = _pilot(graph=[True, False], vector=[False, False])

    # Act
    text = render(analyse(rows, resamples=50))

    # Assert
    assert "Preliminary - no significance claimed" in text
    for condition in ("noop", "oracle", "graph", "vector"):
        assert f"| {condition} |" in text
    assert "graph - vector" in text


def test_render_shows_a_dash_for_undefined_values() -> None:
    rows = [_row("episodic", "t0", True), _row("episodic", "t1", False)]
    assert "| - | 100 | 10 | - | - | - |" in render(analyse(rows, resamples=50))


def test_interval_on_all_undefined_draws_is_nan() -> None:
    """Every resample hit a zero-width bracket: the interval says so rather than guessing."""
    # Act
    interval = _interval(0.5, np.array([np.nan, np.inf]))

    # Assert
    assert interval.estimate == 0.5
    assert math.isnan(interval.low) and math.isnan(interval.high)


def test_the_same_cell_recorded_twice_is_refused() -> None:
    """Same condition, task and seed under two prefixes is not two replicates."""
    rows = [_row("graph", "t0", True), _row("graph", "t0", False)]
    with pytest.raises(AnalysisError, match="duplicate rows"):
        analyse(rows)
