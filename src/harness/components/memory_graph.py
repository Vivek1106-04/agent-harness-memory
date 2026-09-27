"""GraphMemory - retrieval by walking entity links out from the entities the query names.

The representation this study's primary comparison is about (graph vs vector, SPEC.md
section 11). A multi-hop task is a chain of references between entities, and a graph
answers "what is connected to the thing I was asked about" directly, where a cosine index
can only answer "what reads like the question".

Anchors. `MemoryQuery.entities` is used when the caller supplies it. The runtime loop does
not - it queries with the goal text alone - so otherwise the anchors are the entities the
query text mentions. That keeps the provider honest against the loop as it is frozen,
rather than depending on a query field no caller fills in.

Depth. The loop sends the default `hops=1`. The configured `hops` is therefore a floor, and
the query can only ask for more: `depth = max(query.hops, configured hops)`.

Ranking is by hop distance, then newest first. A query that anchors on nothing the graph
knows falls back to recency, for the same reason `VectorMemory` does: returning nothing
would make every retrieval before the first relevant read look like a retrieval failure.
"""

from __future__ import annotations

from harness.entity_graph import EntityGraph, extract_entities
from harness.interfaces import MemoryProvider
from harness.registry import register
from harness.types import MemoryItem, MemoryQuery

DEFAULT_HOPS = 2


@register("memory", name="graph")
class GraphMemory(MemoryProvider):
    capabilities: frozenset[str] = frozenset({"graph"})

    def __init__(self, hops: int = DEFAULT_HOPS) -> None:
        if hops < 0:
            raise ValueError(f"hops must be non-negative, got {hops}")
        self._hops = hops
        self._items: list[MemoryItem] = []
        self._graph = EntityGraph()

    def write(self, item: MemoryItem) -> None:
        self._graph.add(len(self._items), extract_entities(item.content))
        self._items.append(item)

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        """`time_range` and `filters` are ignored; `text` is used only to find anchors."""
        limit = max(query.k, 0)
        if not self._items or limit == 0:
            return []
        anchors = query.entities or extract_entities(query.text or "")
        distances = self._graph.item_hops(anchors, max(query.hops, self._hops))
        if not distances:
            return list(reversed(self._items))[:limit]
        ranked = sorted(distances, key=lambda index: (distances[index], -index))
        return [self._scored(index, distances[index]) for index in ranked[:limit]]

    def reset(self) -> None:
        self._items = []
        self._graph.clear()

    def _scored(self, index: int, hops: int) -> MemoryItem:
        item = self._items[index]
        return MemoryItem(
            content=item.content,
            score=1.0 / (1.0 + hops),
            provenance=item.provenance,
            kind=item.kind,
        )
