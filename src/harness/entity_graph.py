"""Entity extraction and the item-entity graph shared by the graph-shaped memories.

Not a `MemoryProvider`. Providers wrapping providers is out of scope (SPEC.md section 12),
so `GraphMemory` and `HybridMemory` each own an index built from this module rather than
one of them delegating to the other.

The graph is bipartite: item nodes and entity nodes, an edge wherever an item mentions an
entity. Two entities are one hop apart when some item mentions both, which is exactly the
shape of a multi-hop fact - "the code for A is the code for B" links A to B. Keeping items
in the graph, rather than collapsing it to entity-entity edges, is what lets a traversal
come back with the items that justify each hop.

Extraction is lexical and deterministic, on purpose. An LLM extractor would put model
tokens on the write path and nondeterminism into the index, and both would land inside the
number being measured. The price is recorded in DECISIONS.md: the extractor recognises
identifier-shaped names (`site_003_01`, `doc_07`), which is the surface form entities take
in the generated families, and it will need replacing for free-text benchmarks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import networkx as nx

#: Identifier-shaped names: a letter, then at least one underscore-joined segment.
ENTITY_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")

#: A node is (kind, key): ("item", index) or ("entity", name).
Node = tuple[str, int | str]

_ITEM = "item"
_ENTITY = "entity"
#: Item -> entity -> item is one entity hop, and it is two edges in the bipartite graph.
EDGES_PER_HOP = 2


def extract_entities(text: str) -> list[str]:
    """Entity names in `text`, in first-mention order, without duplicates."""
    return list(dict.fromkeys(ENTITY_PATTERN.findall(text)))


class EntityGraph:
    def __init__(self) -> None:
        self._graph: nx.Graph[Node] = nx.Graph()

    def add(self, item_index: int, entities: Iterable[str]) -> None:
        item_node = (_ITEM, item_index)
        self._graph.add_node(item_node)
        for entity in entities:
            self._graph.add_edge(item_node, (_ENTITY, entity))

    def known(self, entities: Iterable[str]) -> list[str]:
        """The subset of `entities` some written item mentions, order preserved."""
        return [entity for entity in entities if self._graph.has_node((_ENTITY, entity))]

    def item_hops(self, anchors: Iterable[str], hops: int) -> dict[int, int]:
        """Item index -> entity hops from the nearest anchor, for items within `hops`.

        An item mentioning an anchor is at 0. Unknown anchors are ignored, and no known
        anchor means no items: the caller decides what an unanchored query returns.
        """
        sources = [(_ENTITY, entity) for entity in self.known(anchors)]
        if not sources:
            return {}
        cutoff = EDGES_PER_HOP * max(hops, 0) + 1
        lengths = nx.multi_source_dijkstra_path_length(self._graph, sources, cutoff=cutoff)
        return {
            int(node[1]): int(length - 1) // EDGES_PER_HOP
            for node, length in lengths.items()
            if node[0] == _ITEM
        }

    def clear(self) -> None:
        self._graph = nx.Graph()
