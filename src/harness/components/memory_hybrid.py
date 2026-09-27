"""HybridMemory - cosine similarity and entity links, fused by reciprocal rank.

Owns both indices itself rather than wrapping `VectorMemory` and `GraphMemory`: providers
wrapping providers is out of scope (SPEC.md section 12). The embedding model is the one
from the frozen `embedding` block, handed in by the builder exactly as `VectorMemory` gets
it, so the semantic half of this condition is identical to the vector condition.

Why reciprocal rank fusion. The two halves score on incomparable scales - a cosine in
[-1, 1] and a hop count - and any weighted sum would need a weight, which is a tuning knob,
which is effort spent on one condition and not the others (the effort confound). RRF uses
ranks only and has one constant fixed from the literature, not tuned here.

An item the graph cannot reach contributes its semantic rank alone, so hybrid never
returns less than the vector half would have found.
"""

from __future__ import annotations

import numpy as np

from harness.embedding import Encoder, Vectors
from harness.entity_graph import EntityGraph, extract_entities
from harness.interfaces import MemoryProvider
from harness.registry import register
from harness.types import MemoryItem, MemoryQuery

DEFAULT_HOPS = 2
#: Cormack, Clarke and Buettcher (2009). Fixed, not tuned - see the module docstring.
RRF_K = 60


@register("memory", name="hybrid")
class HybridMemory(MemoryProvider):
    capabilities: frozenset[str] = frozenset({"graph", "semantic"})

    def __init__(self, encoder: Encoder, hops: int = DEFAULT_HOPS) -> None:
        if hops < 0:
            raise ValueError(f"hops must be non-negative, got {hops}")
        self._encoder = encoder
        self._hops = hops
        self._items: list[MemoryItem] = []
        self._vectors: list[Vectors] = []
        self._graph = EntityGraph()

    def write(self, item: MemoryItem) -> None:
        self._graph.add(len(self._items), extract_entities(item.content))
        self._items.append(item)
        self._vectors.append(self._encoder.encode([item.content])[0])

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        limit = max(query.k, 0)
        if not self._items or limit == 0:
            return []
        if not query.text and not query.entities:
            return list(reversed(self._items))[:limit]
        fused = self._fused(query)
        order = [int(index) for index in np.argsort(-fused, kind="stable") if fused[index] > 0]
        return [self._scored(index, float(fused[index])) for index in order[:limit]]

    def reset(self) -> None:
        self._items = []
        self._vectors = []
        self._graph.clear()

    def _fused(self, query: MemoryQuery) -> Vectors:
        fused = np.zeros(len(self._items), dtype=np.float32)
        if query.text:
            cosine = np.stack(self._vectors) @ self._encoder.encode_queries([query.text])[0]
            for rank, index in enumerate(np.argsort(-cosine, kind="stable")):
                fused[index] += 1.0 / (RRF_K + rank + 1)
        anchors = query.entities or extract_entities(query.text or "")
        distances = self._graph.item_hops(anchors, max(query.hops, self._hops))
        graph_order = sorted(distances, key=lambda index: (distances[index], -index))
        for rank, index in enumerate(graph_order):
            fused[index] += 1.0 / (RRF_K + rank + 1)
        return fused

    def _scored(self, index: int, score: float) -> MemoryItem:
        item = self._items[index]
        return MemoryItem(
            content=item.content, score=score, provenance=item.provenance, kind=item.kind
        )
