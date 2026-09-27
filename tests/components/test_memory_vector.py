from __future__ import annotations

import pytest

from harness.components.memory_vector import VectorMemory
from harness.interfaces import MemoryProvider
from harness.registry import resolve
from harness.types import MemoryItem, MemoryQuery
from tests.contracts.memory import MemoryProviderContract
from tests.fakes import HashingEncoder


def _build() -> MemoryProvider:
    return VectorMemory(encoder=HashingEncoder())


def _item(content: str, step: int = 0) -> MemoryItem:
    return MemoryItem(content=content, score=1.0, provenance=f"run:{step}", kind="episodic")


class TestVectorMemoryContract(MemoryProviderContract):
    provider_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("memory", "vector") is VectorMemory


def test_declares_semantic_capability() -> None:
    assert VectorMemory(encoder=HashingEncoder()).capabilities == frozenset({"semantic"})


def test_ranks_the_similar_item_above_the_unrelated_one() -> None:
    # Arrange
    memory = VectorMemory(encoder=HashingEncoder())
    memory.write(_item("the access code for delta is 4271", step=0))
    memory.write(_item("the weather in oslo is cold", step=1))

    # Act
    results = memory.retrieve(MemoryQuery(text="access code for delta", k=2))

    # Assert
    assert results[0].content == "the access code for delta is 4271"


def test_score_carries_the_cosine_similarity() -> None:
    """The item's stored score is replaced by its similarity to this query, not kept."""
    # Arrange
    memory = VectorMemory(encoder=HashingEncoder())
    memory.write(_item("alpha beta gamma"))

    # Act
    exact = memory.retrieve(MemoryQuery(text="alpha beta gamma", k=1))[0]

    # Assert
    assert exact.score == pytest.approx(1.0)


def test_provenance_and_kind_survive_retrieval() -> None:
    # Arrange
    memory = VectorMemory(encoder=HashingEncoder())
    memory.write(MemoryItem(content="a fact", score=0.5, provenance="run:7", kind="failure"))

    # Act
    item = memory.retrieve(MemoryQuery(text="a fact", k=1))[0]

    # Assert
    assert item.provenance == "run:7"
    assert item.kind == "failure"


def test_a_query_without_text_falls_back_to_recency() -> None:
    """Nothing to be similar to is not the same as nothing to return."""
    # Arrange
    memory = VectorMemory(encoder=HashingEncoder())
    for index in range(3):
        memory.write(_item(f"fact {index}", step=index))

    # Act
    results = memory.retrieve(MemoryQuery(k=2))

    # Assert
    assert [item.content for item in results] == ["fact 2", "fact 1"]


def test_ties_are_broken_by_write_order() -> None:
    """Two equally similar items swapping between runs would look like a real effect."""
    # Arrange
    memory = VectorMemory(encoder=HashingEncoder())
    memory.write(_item("identical text", step=0))
    memory.write(_item("identical text", step=1))

    # Act
    results = memory.retrieve(MemoryQuery(text="identical text", k=2))

    # Assert
    assert [item.provenance for item in results] == ["run:0", "run:1"]


def test_non_positive_k_returns_nothing_rather_than_everything() -> None:
    # Arrange
    memory = VectorMemory(encoder=HashingEncoder())
    memory.write(_item("a fact"))

    # Act / Assert
    assert memory.retrieve(MemoryQuery(text="a fact", k=0)) == []
    assert memory.retrieve(MemoryQuery(text="a fact", k=-3)) == []


def test_reset_clears_the_vectors_as_well_as_the_items() -> None:
    """A stale vector outliving its item would rank a fact the run never saw."""
    # Arrange
    memory = VectorMemory(encoder=HashingEncoder())
    memory.write(_item("a fact"))

    # Act
    memory.reset()
    memory.write(_item("a different fact", step=1))
    results = memory.retrieve(MemoryQuery(text="a fact", k=5))

    # Assert
    assert [item.content for item in results] == ["a different fact"]
