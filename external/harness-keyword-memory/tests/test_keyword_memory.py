"""The harness's own admission test, inherited by a component that lives outside it."""

from __future__ import annotations

from harness_keyword_memory import KeywordMemory

from harness.types import MemoryItem, MemoryQuery
from tests.contracts.memory import MemoryProviderContract


class TestKeywordMemoryContract(MemoryProviderContract):
    provider_factory = KeywordMemory


def _item(content: str, step: int) -> MemoryItem:
    return MemoryItem(content=content, score=1.0, provenance=f"t:{step}", kind="episodic")


def test_ranks_by_shared_words_then_newest() -> None:
    # Arrange
    memory = KeywordMemory()
    memory.write(_item("the access code for site_a is 12", 1))
    memory.write(_item("weather report", 2))
    memory.write(_item("the access code for site_b is 34", 3))

    # Act
    results = memory.retrieve(MemoryQuery(text="access code for site_a", k=2))

    # Assert
    assert [r.content for r in results] == [
        "the access code for site_a is 12",
        "the access code for site_b is 34",
    ]
