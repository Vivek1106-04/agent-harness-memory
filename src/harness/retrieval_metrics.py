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
        self._dropped_recall: float | None = None

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

    def observe_dropped(
        self, history: Sequence[str], in_window: Sequence[str], retrieved: Sequence[str]
    ) -> None:
        """Score one step on the facts the window has lost: read, and no longer in
        `in_window` (the window's text other than what memory supplied). Only those need a
        memory; a fact still in view is not held against a memory that did not return it."""
        seen = [fact for fact in self._required if any(fact in text for text in history)]
        dropped = [fact for fact in seen if not any(fact in text for text in in_window)]
        if not dropped:
            return
        found = [fact for fact in dropped if any(fact in text for text in retrieved)]
        self._dropped_recall = len(found) / len(dropped)

    @property
    def dropped_recall(self) -> float | None:
        """Recall of dropped facts on the last step that had any: at the answer step, did
        memory return what the window had lost."""
        return self._dropped_recall

    @property
    def precision(self) -> float | None:
        return _mean(self._precisions)

    @property
    def recall(self) -> float | None:
        return _mean(self._recalls)

    @property
    def final_recall(self) -> float | None:
        """Recall on the last scored step: the retrieval the answer was given on, when the
        agent answered. The average charges a memory that finds a fact only once the linking
        document is read for every step before; this does not."""
        return self._recalls[-1] if self._recalls else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
