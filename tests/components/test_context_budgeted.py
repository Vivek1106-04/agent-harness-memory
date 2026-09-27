from __future__ import annotations

import pytest

from harness.components.context_budgeted import (
    REASON_BUDGET,
    BudgetedContextManager,
    BudgetTooSmallError,
)
from harness.interfaces import ContextManager
from harness.prompts import SYSTEM_INSTRUCTIONS, render_observation
from harness.registry import resolve
from harness.segments import ROLE_MEMORY, ROLE_OBSERVATION, memory_id, observation_id
from harness.tokens import approx_token_count
from harness.types import MemoryItem, Observation
from tests.contracts.context import ContextManagerContract

GOAL = "find the access code"


def _build(token_budget: int) -> ContextManager:
    return BudgetedContextManager(token_budget)


def _observations(n: int, content: str = "observation padding") -> list[Observation]:
    return [Observation(step=i, content=f"{content} {i}", tool_name="read_file") for i in range(n)]


def _items(n: int) -> list[MemoryItem]:
    return [
        MemoryItem(content=f"recalled fact {i}", score=1.0, provenance=f"run:{i}", kind="semantic")
        for i in range(n)
    ]


def _floor(goal: str = GOAL) -> int:
    return approx_token_count(SYSTEM_INSTRUCTIONS) + approx_token_count(goal)


class TestBudgetedContextManagerContract(ContextManagerContract):
    manager_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("context", "budgeted") is BudgetedContextManager


def test_budget_binds_when_the_history_is_far_larger_than_it() -> None:
    """Experiment 2 is the claim that an 8k budget on a 128k model really is 8k."""
    # Arrange
    manager = BudgetedContextManager(_floor() + 10)

    # Act
    window = manager.assemble(GOAL, _observations(200), _items(20))

    # Assert
    assert window.token_count <= manager.token_budget
    assert len(window.dropped) > 0


def test_memory_is_admitted_before_history() -> None:
    """Memory is the treatment; a budget that evicted it first would measure the budget."""
    # Arrange
    manager = BudgetedContextManager(_floor() + approx_token_count("recalled fact 0"))

    # Act
    window = manager.assemble(GOAL, _observations(5), _items(1))

    # Assert
    roles = [role for role, _ in window.segments]
    assert ROLE_MEMORY in roles
    assert ROLE_OBSERVATION not in roles


def test_recent_observations_survive_and_old_ones_are_evicted() -> None:
    # Arrange
    observations = _observations(10)
    room = approx_token_count("observation padding 9") * 2
    manager = BudgetedContextManager(_floor() + room)

    # Act
    window = manager.assemble(GOAL, observations, [])

    # Assert
    kept = [text for role, text in window.segments if role == ROLE_OBSERVATION]
    assert "observation padding 9" in kept[-1]
    assert not any("observation padding 0" in text for text in kept)


def test_surviving_observations_stay_in_chronological_order() -> None:
    # Arrange
    manager = BudgetedContextManager(_floor() + approx_token_count("observation padding 9") * 3)

    # Act
    window = manager.assemble(GOAL, _observations(10), [])

    # Assert
    steps = [
        int(text.split("[step ")[1].split("]")[0])
        for role, text in window.segments
        if role == ROLE_OBSERVATION
    ]
    assert steps == sorted(steps)


def test_dropped_entries_name_the_segment_and_the_reason() -> None:
    # Arrange
    manager = BudgetedContextManager(_floor() + 1)

    # Act
    window = manager.assemble(GOAL, _observations(3), _items(2))

    # Assert
    assert all(entry.endswith(f"|{REASON_BUDGET}") for entry in window.dropped)
    assert any(entry.startswith("mem:") for entry in window.dropped)
    assert any(entry.startswith("obs:") for entry in window.dropped)


def test_an_oversized_item_is_skipped_and_smaller_ones_still_fit() -> None:
    """Truncating would read as a whole fact to the model and corrupt the trace."""
    # Arrange
    manager = BudgetedContextManager(_floor() + 20)
    huge = MemoryItem(content="x" * 4000, score=1.0, provenance="run:huge", kind="semantic")
    small = MemoryItem(content="short fact", score=0.5, provenance="run:small", kind="semantic")

    # Act
    window = manager.assemble(GOAL, [], [huge, small])

    # Assert
    assert any("short fact" in text for _, text in window.segments)
    assert window.dropped == ["mem:run:huge|budget"]


def test_a_budget_that_cannot_hold_the_instructions_fails_loudly() -> None:
    # Arrange
    manager = BudgetedContextManager(5)

    # Act / Assert
    with pytest.raises(BudgetTooSmallError, match="exceeds the declared budget"):
        manager.assemble(GOAL, [], [])


def test_a_non_positive_budget_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="token_budget must be positive"):
        BudgetedContextManager(0)


def test_token_count_matches_the_counter_it_was_given() -> None:
    # Arrange
    manager = BudgetedContextManager(10_000, count_tokens=approx_token_count)

    # Act
    window = manager.assemble(GOAL, _observations(4), _items(2))

    # Assert
    assert window.token_count == sum(approx_token_count(text) for _, text in window.segments)


def test_every_evicted_segment_is_named_in_dropped() -> None:
    """The week-4 hardening bar: no segment leaves the window without an entry.

    Without a complete record, an Experiment 2 result cannot tell "memory failed to
    recover the fact" from "the fact was never evicted in the first place" - and a
    `dropped` list that is merely mostly complete has exactly that failure mode.
    """
    # Arrange
    observations = _observations(30)
    items = _items(20)
    manager = BudgetedContextManager(_floor() + 40)

    # Act
    window = manager.assemble(GOAL, observations, items)

    # Assert
    offered = {memory_id(item.provenance) for item in items} | {
        observation_id(obs.step) for obs in observations
    }
    kept = {text for _, text in window.segments}
    survived = {memory_id(item.provenance) for item in items if item.content in kept} | {
        observation_id(obs.step) for obs in observations if render_observation(obs) in kept
    }
    named = {entry.split("|", 1)[0] for entry in window.dropped}
    assert named == offered - survived


def test_a_dropped_entry_names_an_identifier_and_a_reason_only() -> None:
    """Identifiers and reasons, never evicted payloads. SECURITY.md section 4."""
    # Arrange
    secret = "the access code for delta is 4271"
    manager = BudgetedContextManager(_floor() + 1)

    # Act
    window = manager.assemble(
        GOAL,
        [],
        [MemoryItem(content=secret, score=1.0, provenance="run:9", kind="semantic")],
    )

    # Assert
    assert window.dropped == [f"mem:run:9|{REASON_BUDGET}"]
    assert secret not in " ".join(window.dropped)
