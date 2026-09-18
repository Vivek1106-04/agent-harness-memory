"""OracleMemory - the ceiling of the bracket. Perfect retrieval, by construction.

It returns exactly the required facts the agent has already encountered: no miss, no
distractor, no ranking error. No implementable memory can beat it, which is what makes it a
ceiling rather than a competitor.

Why the ceiling is not optional. Without it, a low score is unreadable: it could mean the
memory failed to recover the fact, or that the model cannot use the fact even when handed
it perfectly. Those are different problems with different fixes, and only the ceiling tells
them apart - which is the whole point of the week 3 gate.

Two modes, and the difference matters:

* **prepared** - `prepare(spec)` re-derives the task and the gold facts, and retrieval is
  filtered to the gold facts already written. This is the ceiling condition.
* **unprepared** - no task bound, so total recall: the most recently written items, newest
  first. Still an upper bound on any real memory, and it keeps the provider honest against
  the contract suite, which knows nothing about tasks.
"""

from __future__ import annotations

from harness.interfaces import MemoryProvider
from harness.registry import register
from harness.tasks.generator import Fact, generate
from harness.types import MemoryItem, MemoryQuery, TaskSpec

KIND = "semantic"


@register("memory", name="oracle")
class OracleMemory(MemoryProvider):
    capabilities: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self._required: tuple[Fact, ...] = ()
        self._written: list[MemoryItem] = []

    def prepare(self, task: TaskSpec) -> None:
        """Re-derive the task's required facts. Generation is pure, so this is safe."""
        self._required = generate(task).required

    def write(self, item: MemoryItem) -> None:
        self._written.append(item)

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        """Query text, hops, entities and time windows are all ignored - an oracle has no
        retrieval strategy to express. `k` is honoured: the ceiling is measured under the
        same retrieval budget as every condition it is compared against."""
        if self._required:
            items = self._gold_items()
        else:
            items = [
                MemoryItem(
                    content=item.content,
                    score=1.0,
                    provenance=item.provenance,
                    kind=item.kind,
                )
                for item in reversed(self._written)
            ]
        return items[: max(query.k, 0)]

    def reset(self) -> None:
        self._required = ()
        self._written = []

    def _gold_items(self) -> list[MemoryItem]:
        """Required facts the agent has already seen, in dependency order.

        Dependency order rather than observation order because it is the order the chain
        reads in, and because it is stable: a retrieval order that varied run to run would
        be indistinguishable from a real effect in the results.
        """
        seen = "\n".join(item.content for item in self._written)
        return [
            MemoryItem(
                content=fact.content,
                score=1.0,
                provenance=f"oracle:{fact.fact_id}",
                kind=KIND,
            )
            for fact in self._required
            if fact.content in seen
        ]
