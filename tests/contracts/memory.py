"""MemoryProviderContract — the admission criterion for the memory layer.

Subclass it, set `provider_factory`, inherit the tests. An implementation without a
contract subclass is not admissible and will not be merged.

This suite is also the public specification a third-party component must satisfy:
executable, rather than prose.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from harness.interfaces import MemoryProvider
from harness.types import MemoryItem, MemoryQuery


def _item(content: str, step: int = 0, kind: str = "episodic") -> MemoryItem:
    return MemoryItem(content=content, score=1.0, provenance=f"test:{step}", kind=kind)


class MemoryProviderContract:
    provider_factory: Callable[[], MemoryProvider]

    @pytest.fixture
    def provider(self) -> MemoryProvider:
        return type(self).provider_factory()

    def test_write_then_retrieve_returns_item(self, provider: MemoryProvider) -> None:
        # Arrange
        provider.write(_item("the safe combination is 4271"))

        # Act
        results = provider.retrieve(MemoryQuery(text="safe combination", k=5))

        # Assert
        assert any("4271" in r.content for r in results)

    def test_retrieve_on_empty_returns_empty_list(self, provider: MemoryProvider) -> None:
        assert provider.retrieve(MemoryQuery(text="anything", k=5)) == []

    def test_reset_clears_all_state(self, provider: MemoryProvider) -> None:
        provider.write(_item("something memorable"))
        provider.reset()
        assert provider.retrieve(MemoryQuery(text="something memorable", k=5)) == []

    def test_unsupported_query_fields_do_not_crash(self, provider: MemoryProvider) -> None:
        """A vector memory receiving hops=3 ignores it. One that raises is inadmissible."""
        provider.write(_item("alpha relates to beta"))
        query = MemoryQuery(
            text="alpha",
            k=3,
            time_range=(0, 100),
            entities=["alpha", "beta"],
            hops=3,
            filters={"kind": "episodic"},
        )
        provider.retrieve(query)  # must not raise

    def test_retrieve_never_exceeds_k(self, provider: MemoryProvider) -> None:
        for i in range(20):
            provider.write(_item(f"fact number {i}", step=i))
        assert len(provider.retrieve(MemoryQuery(text="fact", k=5))) <= 5

    def test_results_are_deterministic_for_identical_input(
        self, provider: MemoryProvider
    ) -> None:
        """Nondeterminism here is indistinguishable from a real effect in the results."""
        for i in range(10):
            provider.write(_item(f"fact number {i}", step=i))
        query = MemoryQuery(text="fact", k=5)
        first = [r.content for r in provider.retrieve(query)]
        second = [r.content for r in provider.retrieve(query)]
        assert first == second

    def test_declares_capabilities(self, provider: MemoryProvider) -> None:
        assert isinstance(provider.capabilities, frozenset)
        assert provider.capabilities <= {"temporal", "graph", "semantic"}
