"""Retrieval precision and recall against the generator's gold facts.

Separates memory quality from agent quality (SPEC.md section 10): a run can fail with
perfect retrieval (the model could not use the fact) or succeed with poor retrieval (the
fact was still in the window), and success rate alone cannot tell those apart.

Scored per step, then averaged over the steps where it is defined. A step counts only once
the agent has read at least one required fact: before that there is nothing a memory could
have retrieved, and scoring those steps would reward whichever provider returns least.

Relevance is by content: an item is relevant when it contains a required fact's text. That
is the same test `OracleMemory` applies, so the ceiling scores 1.0 on both by construction.
Nothing here is visible to the memory under test - it reads the loop's record, not the
provider.
"""

from __future__ import annotations

from collections.abc import Sequence


class RetrievalTally:
    def __init__(self, required: Sequence[str]) -> None:
        self._required = tuple(required)
        self._precisions: list[float] = []
        self._recalls: list[float] = []

    def observe(self, history: Sequence[str], retrieved: Sequence[str]) -> None:
        """Score one step: `history` is what the agent has seen, `retrieved` what memory gave."""
        seen = [fact for fact in self._required if any(fact in text for text in history)]
        if not seen:
            return
        found = [fact for fact in seen if any(fact in text for text in retrieved)]
        self._recalls.append(len(found) / len(seen))
        if retrieved:
            relevant = [text for text in retrieved if any(f in text for f in self._required)]
            self._precisions.append(len(relevant) / len(retrieved))

    @property
    def precision(self) -> float | None:
        return _mean(self._precisions)

    @property
    def recall(self) -> float | None:
        return _mean(self._recalls)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
