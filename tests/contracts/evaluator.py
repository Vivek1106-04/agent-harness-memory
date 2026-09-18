"""EvaluatorContract - the admission criterion for the evaluator layer.

Deterministic and offline, always. No LLM-as-judge: beyond the confound it reintroduces, a
judge reading model-generated traces is directly injectable. SECURITY.md section 8.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from harness.interfaces import Evaluator
from harness.types import Observation, TaskSpec, Verdict


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="contract-task",
        family="multihop_qa",
        fact_dependency_graph={"answer": ["fact_a", "fact_b"]},
        recall_distance=12,
        distractor_density=0.5,
        horizon=20,
        entity_pool=["alpha", "beta", "gamma", "delta"],
        seed=1,
    )


def _trace() -> list[Observation]:
    return [
        Observation(step=0, content="found fact_a", tool_name="read_file"),
        Observation(step=1, content="found fact_b", tool_name="read_file"),
        Observation(step=2, content="the answer is 4271", tool_name="answer"),
    ]


class EvaluatorContract:
    evaluator_factory: Callable[[], Evaluator]

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return type(self).evaluator_factory()

    def test_verdict_is_deterministic_for_identical_traces(self, evaluator: Evaluator) -> None:
        # Arrange
        task, trace = _task(), _trace()

        # Act
        first = evaluator.judge(task, trace)
        second = evaluator.judge(task, trace)

        # Assert
        assert first == second

    def test_score_is_within_unit_interval(self, evaluator: Evaluator) -> None:
        """Headroom recovered divides by (Oracle - NoOp); a score off-scale corrupts it."""
        # Arrange
        task, trace = _task(), _trace()

        # Act
        verdict = evaluator.judge(task, trace)

        # Assert
        assert isinstance(verdict, Verdict)
        assert 0.0 <= verdict.score <= 1.0

    def test_per_step_covers_the_trace(self, evaluator: Evaluator) -> None:
        # Arrange
        task, trace = _task(), _trace()

        # Act
        verdict = evaluator.judge(task, trace)

        # Assert
        assert len(verdict.per_step) == len(trace)

    def test_empty_trace_is_judged_not_crashed(self, evaluator: Evaluator) -> None:
        # Arrange
        task: TaskSpec = _task()

        # Act
        verdict = evaluator.judge(task, [])

        # Assert
        assert verdict.success is False
        assert verdict.per_step == []
