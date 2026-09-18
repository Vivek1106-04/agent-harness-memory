"""MemoryProviderContract - the admission criterion for the memory layer.

Subclass it, set `provider_factory`, inherit the tests. An implementation without a
contract subclass is not admissible and will not be merged.

This suite is also the public specification a third-party component must satisfy:
executable, rather than prose.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from harness.interfaces import MemoryProvider
from harness.types import MemoryItem, MemoryQuery, TaskSpec


def _item(content: str, step: int = 0, kind: str = "episodic") -> MemoryItem:
    return MemoryItem(content=content, score=1.0, provenance=f"test:{step}", kind=kind)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="contract-task",
        family="multihop_qa",
        fact_dependency_graph={"answer": ["fact_a", "fact_b"]},
        recall_distance=12,
        distractor_density=0.5,
        horizon=20,
        entity_pool=["alpha", "beta", "gamma", "delta"],
        seed=1,
    )


class MemoryProviderContract:
    provider_factory: Callable[[], MemoryProvider]

    #: `NoOpMemory` stores nothing by design. Rather than let it claim to be a memory that
    #: merely performs badly, it declares the fact and the storage tests invert.
    stores_writes: bool = True

    @pytest.fixture
    def provider(self) -> MemoryProvider:
        return type(self).provider_factory()

    def test_write_then_retrieve_returns_item(self, provider: MemoryProvider) -> None:
        # Arrange
        provider.write(_item("the safe combination is 4271"))

        # Act
        results = provider.retrieve(MemoryQuery(text="safe combination", k=5))

        # Assert
        if type(self).stores_writes:
            assert any("4271" in r.content for r in results)
        else:
            assert results == []

    def test_retrieve_on_empty_returns_empty_list(self, provider: MemoryProvider) -> None:
        assert provider.retrieve(MemoryQuery(text="anything", k=5)) == []

    def test_reset_clears_all_state(self, provider: MemoryProvider) -> None:
        provider.write(_item("something memorable"))
        provider.reset()
        assert provider.retrieve(MemoryQuery(text="something memorable", k=5)) == []

    def test_prepare_binds_a_task_without_raising(self, provider: MemoryProvider) -> None:
        """Every provider is offered the task; all but the baselines ignore it."""
        provider.prepare(_task())
        provider.write(_item("the safe combination is 4271"))
        provider.retrieve(MemoryQuery(text="safe combination", k=5))  # must not raise

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

    def test_results_are_deterministic_for_identical_input(self, provider: MemoryProvider) -> None:
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
