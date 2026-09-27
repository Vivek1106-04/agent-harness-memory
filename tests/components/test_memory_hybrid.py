from __future__ import annotations

import pytest

from harness.components.memory_hybrid import RRF_K, HybridMemory
from harness.interfaces import MemoryProvider
from harness.registry import resolve
from harness.types import MemoryItem, MemoryQuery
from tests.contracts.memory import MemoryProviderContract
from tests.fakes import HashingEncoder

TERMINAL = "The access code for site_000_01 is the access code for site_000_00."
ROOT = "The access code for site_000_00 is 4271."
UNRELATED = "the weather in oslo is cold"
GOAL = "Find the access code for site_000_01."


def _build() -> MemoryProvider:
    return HybridMemory(encoder=HashingEncoder())


def _item(content: str, step: int = 0) -> MemoryItem:
    return MemoryItem(content=content, score=1.0, provenance=f"run:{step}", kind="episodic")


def _memory(*contents: str) -> HybridMemory:
    memory = HybridMemory(encoder=HashingEncoder())
    for step, content in enumerate(contents):
        memory.write(_item(content, step=step))
    return memory


class TestHybridMemoryContract(MemoryProviderContract):
    provider_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("memory", "hybrid") is HybridMemory


def test_declares_graph_and_semantic_capabilities() -> None:
    assert _memory().capabilities == frozenset({"graph", "semantic"})


def test_items_both_halves_rank_come_first() -> None:
    # Arrange
    memory = _memory(UNRELATED, ROOT, TERMINAL)

    # Act
    results = memory.retrieve(MemoryQuery(text=GOAL, k=3))

    # Assert
    assert [item.content for item in results][:2] == [TERMINAL, ROOT]
    assert results[-1].content == UNRELATED


def test_an_item_only_the_semantic_half_finds_is_still_returned() -> None:
    """Hybrid never returns less than its vector half would have."""
    # Arrange
    memory = _memory(UNRELATED)

    # Act
    results = memory.retrieve(MemoryQuery(text="weather in oslo", k=1))

    # Assert
    assert [item.content for item in results] == [UNRELATED]


def test_score_is_the_sum_of_reciprocal_ranks() -> None:
    # Arrange
    memory = _memory(TERMINAL)

    # Act
    item = memory.retrieve(MemoryQuery(text=GOAL, k=1))[0]

    # Assert
    assert item.score == pytest.approx(2.0 / (RRF_K + 1))


def test_entities_without_text_use_the_graph_half_alone() -> None:
    # Arrange
    memory = _memory(UNRELATED, ROOT)

    # Act
    results = memory.retrieve(MemoryQuery(entities=["site_000_00"], k=5))

    # Assert
    assert [item.content for item in results] == [ROOT]


def test_a_query_with_neither_text_nor_entities_falls_back_to_recency() -> None:
    # Arrange
    memory = _memory(UNRELATED, ROOT, TERMINAL)

    # Act
    results = memory.retrieve(MemoryQuery(k=2))

    # Assert
    assert [item.content for item in results] == [TERMINAL, ROOT]


def test_provenance_and_kind_survive_retrieval() -> None:
    # Arrange
    memory = HybridMemory(encoder=HashingEncoder())
    memory.write(MemoryItem(content=ROOT, score=0.2, provenance="run:7", kind="failure"))

    # Act
    item = memory.retrieve(MemoryQuery(text=ROOT, k=1))[0]

    # Assert
    assert (item.provenance, item.kind) == ("run:7", "failure")


def test_non_positive_k_returns_nothing() -> None:
    assert _memory(ROOT).retrieve(MemoryQuery(text=GOAL, k=0)) == []


def test_negative_hops_are_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="hops"):
        HybridMemory(encoder=HashingEncoder(), hops=-1)


def test_reset_clears_vectors_graph_and_items() -> None:
    # Arrange
    memory = _memory(ROOT, TERMINAL)

    # Act
    memory.reset()
    memory.write(_item(UNRELATED))
    results = memory.retrieve(MemoryQuery(text=GOAL, k=5))

    # Assert
    assert [item.content for item in results] == [UNRELATED]
