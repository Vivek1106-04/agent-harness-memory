from __future__ import annotations

from harness.components.memory_oracle import OracleMemory
from harness.interfaces import MemoryProvider
from harness.registry import resolve
from harness.tasks.generator import generate
from harness.types import MemoryItem, MemoryQuery
from tests.contracts.memory import MemoryProviderContract
from tests.tasks.specs import spec


def _build() -> MemoryProvider:
    return OracleMemory()


def _observed(content: str, step: int = 0) -> MemoryItem:
    return MemoryItem(content=content, score=1.0, provenance=f"run:{step}", kind="episodic")


class TestOracleMemoryContract(MemoryProviderContract):
    provider_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("memory", "oracle") is OracleMemory


def test_prepared_oracle_returns_required_facts_that_were_seen() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)
    memory = OracleMemory()
    memory.prepare(subject)
    for fact in task.required:
        memory.write(_observed(fact.content, fact.index))

    # Act
    results = memory.retrieve(MemoryQuery(text=task.goal, k=5))

    # Assert
    assert [item.content for item in results] == [fact.content for fact in task.required]


def test_prepared_oracle_withholds_facts_the_agent_has_not_reached() -> None:
    """A ceiling that hands over facts before they were encountered is not a ceiling, it is
    a different task."""
    # Arrange
    subject = spec()
    task = generate(subject)
    memory = OracleMemory()
    memory.prepare(subject)
    memory.write(_observed(task.required[0].content))

    # Act
    results = memory.retrieve(MemoryQuery(k=5))

    # Assert
    assert [item.content for item in results] == [task.required[0].content]


def test_prepared_oracle_never_returns_a_distractor() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)
    memory = OracleMemory()
    memory.prepare(subject)
    for _, content in task.documents:
        memory.write(_observed(content))

    # Act
    results = memory.retrieve(MemoryQuery(k=20))

    # Assert
    contents = {item.content for item in results}
    assert contents == {fact.content for fact in task.required}


def test_prepared_oracle_honours_k() -> None:
    """The ceiling is measured under the same retrieval budget as the conditions it bounds."""
    # Arrange
    subject = spec(graph={"code": ["middle"], "middle": ["origin"]}, recall_distance=4, horizon=14)
    task = generate(subject)
    memory = OracleMemory()
    memory.prepare(subject)
    for fact in task.required:
        memory.write(_observed(fact.content))

    # Act
    results = memory.retrieve(MemoryQuery(k=2))

    # Assert
    assert len(results) == 2


def test_retrieved_items_carry_oracle_provenance() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)
    memory = OracleMemory()
    memory.prepare(subject)
    memory.write(_observed(task.required[0].content))

    # Act
    item = memory.retrieve(MemoryQuery(k=5))[0]

    # Assert
    assert item.provenance == f"oracle:{task.required[0].fact_id}"
    assert item.kind == "semantic"


def test_unprepared_oracle_returns_total_recall_newest_first() -> None:
    # Arrange
    memory = OracleMemory()
    for i in range(3):
        memory.write(_observed(f"fact {i}", i))

    # Act
    results = memory.retrieve(MemoryQuery(k=2))

    # Assert
    assert [item.content for item in results] == ["fact 2", "fact 1"]


def test_reset_clears_both_the_task_binding_and_the_writes() -> None:
    # Arrange
    subject = spec()
    task = generate(subject)
    memory = OracleMemory()
    memory.prepare(subject)
    memory.write(_observed(task.required[0].content))

    # Act
    memory.reset()

    # Assert
    assert memory.retrieve(MemoryQuery(k=5)) == []


def test_non_positive_k_returns_nothing_rather_than_everything() -> None:
    # Arrange
    memory = OracleMemory()
    memory.write(_observed("a fact"))

    # Act / Assert
    assert memory.retrieve(MemoryQuery(k=0)) == []
    assert memory.retrieve(MemoryQuery(k=-3)) == []
