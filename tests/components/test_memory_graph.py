from __future__ import annotations

import pytest

from harness.components.memory_graph import GraphMemory
from harness.interfaces import MemoryProvider
from harness.registry import resolve
from harness.types import MemoryItem, MemoryQuery
from tests.contracts.memory import MemoryProviderContract

TERMINAL = "The access code for site_000_01 is the access code for site_000_00."
ROOT = "The access code for site_000_00 is 4271."
DISTRACTOR = "The access code for site_000_05 is 9999."
GOAL = "Find the access code for site_000_01."


def _build() -> MemoryProvider:
    return GraphMemory()


def _item(content: str, step: int = 0) -> MemoryItem:
    return MemoryItem(content=content, score=1.0, provenance=f"run:{step}", kind="episodic")


def _chain_memory(hops: int = 1) -> GraphMemory:
    memory = GraphMemory(hops=hops)
    memory.write(_item(ROOT, step=1))
    memory.write(_item(DISTRACTOR, step=2))
    memory.write(_item(TERMINAL, step=3))
    return memory


class TestGraphMemoryContract(MemoryProviderContract):
    provider_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("memory", "graph") is GraphMemory


def test_declares_graph_capability() -> None:
    assert GraphMemory().capabilities == frozenset({"graph"})


def test_walks_from_the_goal_entity_to_the_fact_it_depends_on() -> None:
    # Arrange
    memory = _chain_memory()

    # Act
    results = memory.retrieve(MemoryQuery(text=GOAL, k=5))

    # Assert
    assert [item.content for item in results] == [TERMINAL, ROOT]


def test_explicit_entities_take_precedence_over_the_query_text() -> None:
    # Arrange
    memory = _chain_memory()

    # Act
    results = memory.retrieve(MemoryQuery(text=GOAL, entities=["site_000_05"], k=5))

    # Assert
    assert [item.content for item in results] == [DISTRACTOR]


def test_the_configured_depth_is_a_floor_the_query_cannot_lower() -> None:
    # Arrange
    memory = _chain_memory(hops=0)

    # Act
    shallow = memory.retrieve(MemoryQuery(text=GOAL, k=5, hops=0))
    deep = memory.retrieve(MemoryQuery(text=GOAL, k=5, hops=1))

    # Assert
    assert [item.content for item in shallow] == [TERMINAL]
    assert [item.content for item in deep] == [TERMINAL, ROOT]


def test_score_falls_with_hop_distance() -> None:
    # Arrange
    memory = _chain_memory()

    # Act
    terminal, root = memory.retrieve(MemoryQuery(text=GOAL, k=5))

    # Assert
    assert terminal.score == pytest.approx(1.0)
    assert root.score == pytest.approx(0.5)


def test_an_unanchored_query_falls_back_to_recency() -> None:
    # Arrange
    memory = _chain_memory()

    # Act
    results = memory.retrieve(MemoryQuery(text="nothing named here", k=2))

    # Assert
    assert [item.content for item in results] == [TERMINAL, DISTRACTOR]


def test_equal_distances_are_ranked_newest_first() -> None:
    # Arrange
    memory = GraphMemory()
    memory.write(_item("old note on site_a_1", step=1))
    memory.write(_item("new note on site_a_1", step=2))

    # Act
    results = memory.retrieve(MemoryQuery(entities=["site_a_1"], k=2))

    # Assert
    assert [item.provenance for item in results] == ["run:2", "run:1"]


def test_provenance_and_kind_survive_retrieval() -> None:
    # Arrange
    memory = GraphMemory()
    memory.write(MemoryItem(content=ROOT, score=0.2, provenance="run:7", kind="failure"))

    # Act
    item = memory.retrieve(MemoryQuery(entities=["site_000_00"], k=1))[0]

    # Assert
    assert (item.provenance, item.kind) == ("run:7", "failure")


def test_non_positive_k_returns_nothing() -> None:
    memory = _chain_memory()
    assert memory.retrieve(MemoryQuery(text=GOAL, k=0)) == []


def test_negative_hops_are_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="hops"):
        GraphMemory(hops=-1)


def test_reset_clears_the_graph_as_well_as_the_items() -> None:
    """A stale graph node outliving its item would point at an index that is reused."""
    # Arrange
    memory = _chain_memory()

    # Act
    memory.reset()
    memory.write(_item(DISTRACTOR))
    results = memory.retrieve(MemoryQuery(text=GOAL, k=5))

    # Assert
    assert [item.content for item in results] == [DISTRACTOR]
