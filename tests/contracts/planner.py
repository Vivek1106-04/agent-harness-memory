"""PlannerContract - the admission criterion for the planner layer.

Subclass it, set `planner_factory`, inherit the tests. The factory receives the step budget
and an `LLMClient`; a planner that does not consult a model simply ignores the client.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from harness.interfaces import LLMClient, Planner
from harness.types import Action, ContextWindow
from tests.fakes import FakeLLMClient

MAX_STEPS = 3

MALFORMED_OUTPUTS = [
    "",
    "not json at all {{{",
    '{"tool_name": ',
    '{"unexpected": "shape"}',
    "Thought: I will now" * 200,
]


def _window(text: str = "goal: find the safe combination") -> ContextWindow:
    return ContextWindow(
        segments=[("system", "you are an agent"), ("goal", text)], token_count=10, dropped=[]
    )


class PlannerContract:
    planner_factory: Callable[[int, LLMClient], Planner]

    def _make(self, responses: list[str], max_steps: int = MAX_STEPS) -> Planner:
        return type(self).planner_factory(max_steps, FakeLLMClient(responses))

    def test_emits_valid_action_or_terminates(self) -> None:
        # Arrange
        planner = self._make(['{"tool_name": "read_file", "arguments": {"path": "notes.txt"}}'])

        # Act
        action = planner.next_action(_window())

        # Assert
        assert action is None or isinstance(action, Action)
        if action is not None:
            assert action.tool_name
            assert isinstance(action.arguments, dict)

    def test_respects_max_steps(self) -> None:
        """The planner stops on its own budget. The runtime enforces its own cap as well."""
        # Arrange
        planner = self._make(['{"tool_name": "echo", "arguments": {"text": "hi"}}'])

        # Act
        for _ in range(MAX_STEPS):
            planner.next_action(_window())
        after_budget = [planner.next_action(_window()) for _ in range(2)]

        # Assert
        assert after_budget == [None, None]

    @pytest.mark.parametrize("output", MALFORMED_OUTPUTS)
    def test_malformed_model_output_does_not_crash(self, output: str) -> None:
        """Malformed output is an expected condition, not an exception. SECURITY.md section 3."""
        # Arrange
        planner = self._make([output])

        # Act
        action = planner.next_action(_window())

        # Assert
        assert action is None or isinstance(action, Action)
