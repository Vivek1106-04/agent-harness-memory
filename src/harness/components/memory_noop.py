"""NoOpMemory - the floor of the bracket. Writes vanish, retrieval is always empty.

Half of the measurement bracket (`OracleMemory` is the other half), and the reference every
reported number is normalised against:

    headroom recovered = (condition - NoOp) / (Oracle - NoOp)

It is also the instrument behind the forces-recall assertion: a generated task that
succeeds under NoOp was solvable without memory at all, so it measures nothing and is
rejected from the set. See SPEC.md section 9 and `harness.tasks.forces_recall`.

It stores nothing on purpose, so it declares `stores_writes = False` to its contract suite
rather than pretending to be a memory that merely performs badly.
"""

from __future__ import annotations

from harness.interfaces import MemoryProvider
from harness.registry import register
from harness.types import MemoryItem, MemoryQuery


@register("memory", name="noop")
class NoOpMemory(MemoryProvider):
    capabilities: frozenset[str] = frozenset()

    def write(self, item: MemoryItem) -> None:
        """Accepted and discarded. Never raises: the runtime must not special-case it."""

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        return []

    def reset(self) -> None:
        """Nothing is held, so nothing is cleared."""
