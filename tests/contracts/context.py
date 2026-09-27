"""ContextManagerContract - the admission criterion for the context layer.

`test_never_exceeds_declared_budget` is the load-bearing one. The whole of Experiment 2
rests on the harness budget binding independently of model capability; an unbounded prompt
is also an unbounded bill (SECURITY.md section 4).

The bound asserted is the manager's own declared `token_budget`. A manager that declares an
unbounded budget (`FullContextManager`) passes honestly rather than vacuously - what is
forbidden is exceeding what it declared.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from harness.interfaces import ContextManager
from harness.prompts import SYSTEM_INSTRUCTIONS
from harness.segments import PROTECTED_ROLES
from harness.tokens import approx_token_count
from harness.types import MemoryItem, Observation

GOAL = "GOALMARKER recover the safe combination and write it to answer.txt"

#: Headroom above the unevictable floor: enough to keep a few segments, far short of the
#: history built below, so eviction is forced.
BUDGET_HEADROOM = 64

#: Derived from the frozen system text rather than hardcoded. A literal would turn this
#: whole suite into a test of `BudgetTooSmallError` the next time that text grows, which is
#: a silent way to stop testing the budget at all.
SMALL_BUDGET = approx_token_count(SYSTEM_INSTRUCTIONS) + approx_token_count(GOAL) + BUDGET_HEADROOM


def _history(n: int = 40) -> list[Observation]:
    return [
        Observation(step=i, content=f"OBSMARKER{i} " + ("padding text " * 20), tool_name="echo")
        for i in range(n)
    ]


def _retrieved(n: int = 10) -> list[MemoryItem]:
    return [
        MemoryItem(
            content=f"MEMMARKER{i} " + ("recalled text " * 20),
            score=1.0 - i / 100,
            provenance=f"run:{i}",
            kind="episodic",
        )
        for i in range(n)
    ]


def _markers(history: list[Observation], retrieved: list[MemoryItem]) -> list[str]:
    return [f"OBSMARKER{o.step}" for o in history] + [
        f"MEMMARKER{i}" for i in range(len(retrieved))
    ]


class ContextManagerContract:
    manager_factory: Callable[[int], ContextManager]

    @pytest.fixture
    def manager(self) -> ContextManager:
        return type(self).manager_factory(SMALL_BUDGET)

    def test_never_exceeds_declared_budget(self, manager: ContextManager) -> None:
        # Arrange
        history, retrieved = _history(), _retrieved()

        # Act
        window = manager.assemble(GOAL, history, retrieved)

        # Assert
        assert window.token_count <= manager.token_budget

    def test_assembly_is_deterministic_for_identical_input(self, manager: ContextManager) -> None:
        """Two runs differing only in which segments survived are two different conditions."""
        # Arrange
        history, retrieved = _history(), _retrieved()

        # Act
        first = manager.assemble(GOAL, history, retrieved)
        second = manager.assemble(GOAL, history, retrieved)

        # Assert
        assert first == second

    def test_dropped_list_accounts_for_every_evicted_segment(self, manager: ContextManager) -> None:
        # Arrange
        history, retrieved = _history(), _retrieved()
        markers = _markers(history, retrieved)

        # Act
        window = manager.assemble(GOAL, history, retrieved)

        # Assert
        kept = sum(1 for m in markers if any(m in text for _, text in window.segments))
        assert kept + len(window.dropped) == len(markers)

    def test_dropped_list_records_identifiers_not_payloads(self, manager: ContextManager) -> None:
        """SECURITY.md section 4 - `dropped` must not write evicted content to disk."""
        # Arrange
        history, retrieved = _history(), _retrieved()
        markers = _markers(history, retrieved)

        # Act
        window = manager.assemble(GOAL, history, retrieved)

        # Assert
        assert all("padding text" not in entry for entry in window.dropped)
        assert all("recalled text" not in entry for entry in window.dropped)
        assert all(any(m in entry for m in markers) or ":" in entry for entry in window.dropped)

    def test_system_instructions_and_goal_are_never_evicted(self, manager: ContextManager) -> None:
        # Arrange
        history, retrieved = _history(), _retrieved()

        # Act
        window = manager.assemble(GOAL, history, retrieved)

        # Assert
        roles = {role for role, _ in window.segments}
        assert roles >= PROTECTED_ROLES
        assert any("GOALMARKER" in text for _, text in window.segments)
