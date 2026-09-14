from __future__ import annotations

from harness.components.evaluator_always_fail import AlwaysFailEvaluator
from harness.interfaces import Evaluator
from harness.types import Observation, TaskSpec
from tests.contracts.evaluator import EvaluatorContract


def _build() -> Evaluator:
    return AlwaysFailEvaluator()


class TestAlwaysFailEvaluatorContract(EvaluatorContract):
    evaluator_factory = staticmethod(_build)


def test_every_step_is_marked_failed() -> None:
    # Arrange
    evaluator = AlwaysFailEvaluator()
    task = TaskSpec(
        task_id="t1",
        family="multihop_qa",
        fact_dependency_graph={},
        recall_distance=5,
        distractor_density=0.0,
        horizon=10,
        entity_pool=[],
        seed=1,
    )
    trace = [Observation(step=i, content="anything", tool_name="echo") for i in range(4)]

    # Act
    verdict = evaluator.judge(task, trace)

    # Assert
    assert verdict.success is False
    assert verdict.score == 0.0
    assert verdict.per_step == [False, False, False, False]
