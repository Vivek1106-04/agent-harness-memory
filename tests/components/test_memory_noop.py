from __future__ import annotations

from harness.components.memory_noop import NoOpMemory
from harness.interfaces import MemoryProvider
from harness.registry import resolve
from harness.types import MemoryItem, MemoryQuery
from tests.contracts.memory import MemoryProviderContract


def _build() -> MemoryProvider:
    return NoOpMemory()


class TestNoOpMemoryContract(MemoryProviderContract):
    provider_factory = staticmethod(_build)
    stores_writes = False


def test_registered_under_its_config_name() -> None:
    assert resolve("memory", "noop") is NoOpMemory


def test_retrieval_is_empty_however_much_was_written() -> None:
    # Arrange
    memory = NoOpMemory()
    for i in range(50):
        memory.write(
            MemoryItem(content=f"fact {i}", score=1.0, provenance=f"r:{i}", kind="episodic")
        )

    # Act
    results = memory.retrieve(MemoryQuery(text="fact", k=10))

    # Assert
    assert results == []


def test_reset_is_accepted() -> None:
    memory = NoOpMemory()
    memory.reset()
    assert memory.retrieve(MemoryQuery(k=1)) == []


def test_declares_no_capabilities() -> None:
    """The floor of the bracket honours nothing and says so."""
    assert NoOpMemory().capabilities == frozenset()
