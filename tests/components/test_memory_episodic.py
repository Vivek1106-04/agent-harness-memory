from __future__ import annotations

from harness.components.memory_episodic import EpisodicMemory
from harness.interfaces import MemoryProvider
from harness.registry import resolve
from harness.types import MemoryItem, MemoryQuery
from tests.contracts.memory import MemoryProviderContract


def _build() -> MemoryProvider:
    return EpisodicMemory()


def _item(content: str, step: int = 0) -> MemoryItem:
    return MemoryItem(content=content, score=1.0, provenance=f"run:{step}", kind="episodic")


class TestEpisodicMemoryContract(MemoryProviderContract):
    provider_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("memory", "episodic") is EpisodicMemory


def test_declares_temporal_capability() -> None:
    assert EpisodicMemory().capabilities == frozenset({"temporal"})


def test_returns_the_most_recent_first() -> None:
    # Arrange
    memory = EpisodicMemory()
    for step in range(4):
        memory.write(_item(f"fact {step}", step=step))

    # Act
    results = memory.retrieve(MemoryQuery(k=2))

    # Assert
    assert [item.content for item in results] == ["fact 3", "fact 2"]


def test_time_range_restricts_retrieval_to_the_window() -> None:
    """The reason MemoryQuery is wide: a time window cannot be smuggled through a string."""
    # Arrange
    memory = EpisodicMemory()
    for step in range(6):
        memory.write(_item(f"fact {step}", step=step))

    # Act
    results = memory.retrieve(MemoryQuery(time_range=(1, 3), k=10))

    # Assert
    assert [item.content for item in results] == ["fact 3", "fact 2", "fact 1"]


def test_time_range_bounds_are_inclusive() -> None:
    # Arrange
    memory = EpisodicMemory()
    memory.write(_item("only fact", step=5))

    # Act / Assert
    assert memory.retrieve(MemoryQuery(time_range=(5, 5), k=1)) != []
    assert memory.retrieve(MemoryQuery(time_range=(6, 9), k=1)) == []


def test_step_is_read_from_the_provenance_the_runtime_writes() -> None:
    """The runtime builds provenance as '<run_id>:<step>'; there is no timestamp field."""
    # Arrange
    memory = EpisodicMemory()
    memory.write(MemoryItem(content="late", score=1.0, provenance="run-a:9", kind="episodic"))
    memory.write(MemoryItem(content="early", score=1.0, provenance="run-a:2", kind="episodic"))

    # Act
    results = memory.retrieve(MemoryQuery(time_range=(0, 5), k=10))

    # Assert
    assert [item.content for item in results] == ["early"]


def test_provenance_without_a_step_falls_back_to_write_order() -> None:
    # Arrange
    memory = EpisodicMemory()
    memory.write(MemoryItem(content="first", score=1.0, provenance="nostep", kind="episodic"))
    memory.write(MemoryItem(content="second", score=1.0, provenance="also:bad", kind="episodic"))

    # Act
    results = memory.retrieve(MemoryQuery(time_range=(1, 1), k=10))

    # Assert
    assert [item.content for item in results] == ["second"]


def test_query_text_is_ignored_rather_than_refused() -> None:
    """This provider ranks by time and declares only `temporal`. Raising is inadmissible."""
    # Arrange
    memory = EpisodicMemory()
    memory.write(_item("nothing like the query", step=0))

    # Act
    results = memory.retrieve(MemoryQuery(text="completely unrelated", k=1))

    # Assert
    assert [item.content for item in results] == ["nothing like the query"]


def test_non_positive_k_returns_nothing_rather_than_everything() -> None:
    # Arrange
    memory = EpisodicMemory()
    memory.write(_item("a fact"))

    # Act / Assert
    assert memory.retrieve(MemoryQuery(k=0)) == []
    assert memory.retrieve(MemoryQuery(k=-3)) == []
