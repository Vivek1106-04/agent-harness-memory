"""KeywordMemory - retrieval by shared words, from outside the harness.

A deliberately plain provider: an item scores the number of distinct query words it
contains, ties go to the newest. No embeddings, no graph. It exists to show that a memory
written in a separate package, installed alongside the harness, becomes selectable by
name in config and passes the same conformance suite as the built-ins - with no line of
harness source edited.
"""

from __future__ import annotations

import re

from harness.interfaces import MemoryProvider
from harness.types import MemoryItem, MemoryQuery

_WORD = re.compile(r"\w+")


def _words(text: str) -> frozenset[str]:
    return frozenset(word.lower() for word in _WORD.findall(text))


class KeywordMemory(MemoryProvider):
    #: Lexical overlap is none of the harness's declared capabilities, so it declares none
    #: and ignores `time_range`, `entities` and `hops`.
    capabilities: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    def write(self, item: MemoryItem) -> None:
        self._items.append(item)

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        limit = max(query.k, 0)
        if limit == 0 or not self._items:
            return []
        wanted = _words(query.text or "")
        ranked = sorted(
            range(len(self._items)),
            key=lambda index: (-len(wanted & _words(self._items[index].content)), -index),
        )
        return [self._items[index] for index in ranked[:limit]]

    def reset(self) -> None:
        self._items = []
