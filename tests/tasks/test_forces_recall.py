from __future__ import annotations

from harness.tasks.forces_recall import check, filter_specs, static_gate
from harness.types import TaskSpec, Verdict
from tests.tasks.specs import spec


def _verdict(success: bool) -> Verdict:
    return Verdict(success=success, score=1.0 if success else 0.0, per_step=[], reason="test")


def test_static_gate_rejects_a_fact_still_inside_the_noop_window() -> None:
    assert static_gate(spec(recall_distance=3), noop_window=5) is not None


def test_static_gate_passes_a_fact_outside_the_noop_window() -> None:
    assert static_gate(spec(recall_distance=6), noop_window=5) is None


def test_static_rejection_never_spends_a_model_call() -> None:
    """The cheap gate exists so the expensive one only runs on plausible specs."""
    # Arrange
    calls: list[str] = []

    def run(subject: TaskSpec) -> Verdict:
        calls.append(subject.task_id)
        return _verdict(False)

    # Act
    report = check(spec(recall_distance=2), run, noop_window=5)

    # Assert
    assert report.accepted is False
    assert calls == []


def test_task_solved_under_noop_is_rejected() -> None:
    # Act
    report = check(spec(recall_distance=6), lambda _: _verdict(True), noop_window=5)

    # Assert
    assert report.accepted is False
    assert "does not force recall" in report.reason


def test_task_failing_under_noop_is_accepted() -> None:
    # Act
    report = check(spec(recall_distance=6), lambda _: _verdict(False), noop_window=5)

    # Assert
    assert report.accepted is True
    assert report.task_id == "task-1"


def test_filter_partitions_the_candidate_set_and_reports_on_every_spec() -> None:
    # Arrange
    candidates = [
        spec(task_id="keep", recall_distance=6),
        spec(task_id="too-near", recall_distance=1),
        spec(task_id="solved", recall_distance=7),
    ]

    def run(subject: TaskSpec) -> Verdict:
        return _verdict(subject.task_id == "solved")

    # Act
    accepted, reports = filter_specs(candidates, run, noop_window=5)

    # Assert
    assert [subject.task_id for subject in accepted] == ["keep"]
    assert [report.task_id for report in reports] == ["keep", "too-near", "solved"]


def test_filter_accepts_a_generator_of_specs() -> None:
    """The candidate set is streamed in real use; consuming it twice would silently drop it."""
    # Arrange
    candidates = (spec(task_id=f"t{i}", recall_distance=6) for i in range(3))

    # Act
    accepted, reports = filter_specs(candidates, lambda _: _verdict(False), noop_window=5)

    # Assert
    assert len(accepted) == 3
    assert len(reports) == 3
