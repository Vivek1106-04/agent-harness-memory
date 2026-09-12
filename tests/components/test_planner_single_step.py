from __future__ import annotations

from harness.components.planner_single_step import ANSWER_TOOL, SingleStepPlanner
from harness.interfaces import LLMClient, Planner
from harness.types import ContextWindow
from tests.contracts.planner import PlannerContract
from tests.fakes import FakeLLMClient


def _build(max_steps: int, llm: LLMClient) -> Planner:
    return SingleStepPlanner(llm, max_steps=max_steps)


class TestSingleStepPlannerContract(PlannerContract):
    planner_factory = staticmethod(_build)


def test_answer_carries_the_model_output_verbatim() -> None:
    # Arrange
    llm = FakeLLMClient(["the combination is 4271"])
    planner = SingleStepPlanner(llm)
    window = ContextWindow(segments=[("goal", "find it")], token_count=3, dropped=[])

    # Act
    action = planner.next_action(window)

    # Assert
    assert action is not None
    assert action.tool_name == ANSWER_TOOL
    assert action.arguments["text"] == "the combination is 4271"


def test_second_call_terminates_without_another_model_call() -> None:
    # Arrange
    llm = FakeLLMClient(["first"])
    planner = SingleStepPlanner(llm)
    window = ContextWindow(segments=[("goal", "find it")], token_count=3, dropped=[])

    # Act
    planner.next_action(window)
    second = planner.next_action(window)

    # Assert
    assert second is None
    assert len(llm.calls) == 1
