"""EpisodicMemory - what happened, and when. Retrieval by time window, newest first.

The counterpart to `VectorMemory`, and in this study the more interesting half: a
multi-hop task is a sequence, and "the fact I read around step 4" is a query a cosine
index answers badly and a timeline answers exactly. Declaring `temporal` and honouring
`time_range` is the whole point of `MemoryQuery` being wide - narrowing it to
`retrieve(text, k)` would force this provider to smuggle a time window through a string
and we would be benchmarking a crippled version of it.

Where the time comes from. `MemoryItem` carries no timestamp field and `types.py` is
frozen, so the step is read from `provenance`, which the runtime builds as
`"<run_id>:<step>"`. An item whose provenance does not end in a step falls back to its
write order, which is still a monotonic clock and still deterministic. Wall-clock time is
deliberately not used anywhere: the evaluator must return the same verdict for the same
trace on a machine in a different timezone.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.interfaces import MemoryProvider
from harness.registry import register
from harness.types import MemoryItem, MemoryQuery


@dataclass(frozen=True)
class _Episode:
    item: MemoryItem
    at: int


@register("memory", name="episodic")
class EpisodicMemory(MemoryProvider):
    capabilities: frozenset[str] = frozenset({"temporal"})

    def __init__(self) -> None:
        self._episodes: list[_Episode] = []

    def write(self, item: MemoryItem) -> None:
        self._episodes.append(_Episode(item=item, at=_step_of(item, len(self._episodes))))

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        """Newest first, restricted to `time_range` when one is given.

        `text`, `entities` and `hops` are ignored: this provider ranks by time and says so
        in `capabilities`. A provider that raised on them would be inadmissible.
        """
        limit = max(query.k, 0)
        if limit == 0:
            return []
        window = self._in_window(query.time_range)
        return [episode.item for episode in reversed(window)][:limit]

    def reset(self) -> None:
        self._episodes = []

    def _in_window(self, time_range: tuple[int, int] | None) -> list[_Episode]:
        if time_range is None:
            return self._episodes
        start, end = time_range
        return [episode for episode in self._episodes if start <= episode.at <= end]


def _step_of(item: MemoryItem, fallback: int) -> int:
    """Step from a `"<run_id>:<step>"` provenance string, else the item's write order."""
    _, separator, tail = item.provenance.rpartition(":")
    if not separator:
        return fallback
    try:
        return int(tail)
    except ValueError:
        return fallback
