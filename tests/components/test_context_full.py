from __future__ import annotations

from harness.components.context_full import UNBOUNDED, FullContextManager
from harness.interfaces import ContextManager
from harness.segments import ROLE_GOAL, ROLE_MEMORY, ROLE_OBSERVATION, ROLE_SYSTEM
from harness.tokens import approx_token_count
from harness.types import MemoryItem, Observation
from tests.contracts.context import ContextManagerContract


def _build(token_budget: int) -> ContextManager:
    """The requested budget is deliberately discarded: this manager declares no budget."""
    return FullContextManager()


class TestFullContextManagerContract(ContextManagerContract):
    manager_factory = staticmethod(_build)


def test_declares_an_unbounded_budget() -> None:
    assert FullContextManager().token_budget == UNBOUNDED


def test_keeps_every_input_and_drops_nothing() -> None:
    # Arrange
    manager = FullContextManager()
    history = [Observation(step=i, content=f"obs {i}", tool_name="echo") for i in range(5)]
    retrieved = [
        MemoryItem(content=f"mem {i}", score=1.0, provenance=f"run:{i}", kind="episodic")
        for i in range(3)
    ]

    # Act
    window = manager.assemble("the goal", history, retrieved)

    # Assert
    roles = [role for role, _ in window.segments]
    assert roles[:2] == [ROLE_SYSTEM, ROLE_GOAL]
    assert roles.count(ROLE_MEMORY) == 3
    assert roles.count(ROLE_OBSERVATION) == 5
    assert window.dropped == []


def test_token_count_matches_the_counter_it_was_given() -> None:
    # Arrange
    manager = FullContextManager(count_tokens=approx_token_count)
    history = [Observation(step=0, content="a" * 400, tool_name="echo")]

    # Act
    window = manager.assemble("goal text", history, [])

    # Assert
    assert window.token_count == sum(approx_token_count(text) for _, text in window.segments)


def test_error_observations_are_rendered_as_errors() -> None:
    # Arrange
    manager = FullContextManager()
    history = [Observation(step=2, content="", tool_name="read_file", error="no such file")]

    # Act
    window = manager.assemble("goal", history, [])

    # Assert
    assert any("error from read_file: no such file" in text for _, text in window.segments)
