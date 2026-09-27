"""VectorMemory - exact cosine retrieval over everything written.

The first memory implementation that is actually under test rather than bracketing the
test. It is the obvious baseline every memory paper compares against, so it has to be a
fair version of itself: exact search, not an approximate index.

Exact search, and why that is not laziness. An ANN index trades recall for speed, and that
trade would land inside the number we are measuring - a condition that scored low because
its index missed a neighbour is indistinguishable from a memory design that does not work.
A run's corpus is hundreds of items, so a dense matrix multiply is fast enough that the
approximation buys nothing and costs a confound.

Query fields other than `text` and `k` are ignored, and the provider says so through
`capabilities`. Ignoring is the contract; raising is inadmissible.
"""

from __future__ import annotations

import numpy as np

from harness.embedding import Encoder, Vectors
from harness.interfaces import MemoryProvider
from harness.registry import register
from harness.types import MemoryItem, MemoryQuery


@register("memory", name="vector")
class VectorMemory(MemoryProvider):
    capabilities: frozenset[str] = frozenset({"semantic"})

    def __init__(self, encoder: Encoder) -> None:
        """`encoder` is supplied by the builder from the frozen `embedding` config block.

        Taking the encoder rather than a model name keeps the pinned revision in one place
        - the config that is hashed into every trace header - instead of letting a memory
        block name a second, unhashed model.
        """
        self._encoder = encoder
        self._items: list[MemoryItem] = []
        self._vectors: list[Vectors] = []

    def write(self, item: MemoryItem) -> None:
        self._items.append(item)
        self._vectors.append(self._encoder.encode([item.content])[0])

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        limit = max(query.k, 0)
        if not self._items or limit == 0:
            return []
        if not query.text:
            return self._recent(limit)

        scores = np.stack(self._vectors) @ self._encoder.encode_queries([query.text])[0]
        return [self._scored(index, scores[index]) for index in self._ranked(scores, limit)]

    def reset(self) -> None:
        self._items = []
        self._vectors = []

    def _recent(self, limit: int) -> list[MemoryItem]:
        """A query with no text has nothing to be similar to, so fall back to recency.

        Returning nothing would be defensible too, and would also make every retrieval the
        loop performs before the goal is known look like a retrieval failure.
        """
        return list(reversed(self._items))[:limit]

    @staticmethod
    def _ranked(scores: Vectors, limit: int) -> list[int]:
        """Indices of the `limit` best scores, highest first, ties broken by write order.

        A stable sort on the negated scores is what fixes the tie-break. Without it two
        equally similar items could swap places between runs, and nondeterminism in
        retrieval order is indistinguishable from a real effect in the results.
        """
        order = np.argsort(-scores, kind="stable")
        return [int(index) for index in order[:limit]]

    def _scored(self, index: int, score: float) -> MemoryItem:
        item = self._items[index]
        return MemoryItem(
            content=item.content,
            score=float(score),
            provenance=item.provenance,
            kind=item.kind,
        )
