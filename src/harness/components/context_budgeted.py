"""BudgetedContextManager - the budget that makes Experiment 2 mean anything.

The harness budget is imposed independently of model capability: a 128k-context model run
at an 8k budget must genuinely see 8k, or the "uncapped" cell and the "8k" cell are the same
cell and the experiment collapses. So the guarantee here is hard - the assembled window
never exceeds `token_budget` - and every segment the budget refuses is named in `dropped`.

Priority order, from most protected to first evicted:

1. system instructions and the goal - never evictable (SECURITY.md section 3: they are also
   the structural defence against retrieved content displacing the instructions)
2. retrieved memory items, in the order the provider ranked them - this is the treatment
   under test, and a budget that drops it first would measure the budget, not the memory
3. observations, most recent first - recency is the cheapest baseline for relevance, and it
   is what the agent loses when the horizon grows

An oversized item is skipped rather than truncated: half a fact reads as a whole fact to
the model and silently corrupts the trace.
"""

from __future__ import annotations

from collections.abc import Callable

from harness.interfaces import ContextManager
from harness.prompts import SYSTEM_INSTRUCTIONS, render_observation
from harness.registry import register
from harness.segments import (
    ROLE_GOAL,
    ROLE_MEMORY,
    ROLE_OBSERVATION,
    ROLE_SYSTEM,
    dropped_entry,
    memory_id,
    observation_id,
)
from harness.tokens import approx_token_count
from harness.types import ContextWindow, MemoryItem, Observation

REASON_BUDGET = "budget"


class BudgetTooSmallError(Exception):
    """The protected segments alone exceed the budget.

    Raised rather than trimmed. Truncating the system instructions or the goal would run a
    different experiment than the one written down, and doing it quietly is worse.
    """


@register("context", name="budgeted")
class BudgetedContextManager(ContextManager):
    def __init__(
        self,
        token_budget: int,
        count_tokens: Callable[[str], int] = approx_token_count,
    ) -> None:
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")
        self.token_budget = token_budget
        self.count_tokens = count_tokens

    def assemble(
        self,
        goal: str,
        history: list[Observation],
        retrieved: list[MemoryItem],
    ) -> ContextWindow:
        protected: list[tuple[str, str]] = [
            (ROLE_SYSTEM, SYSTEM_INSTRUCTIONS),
            (ROLE_GOAL, goal),
        ]
        used = sum(self.count_tokens(text) for _, text in protected)
        if used > self.token_budget:
            raise BudgetTooSmallError(
                f"system instructions and goal cost {used} tokens, which exceeds the declared "
                f"budget of {self.token_budget}"
            )

        dropped: list[str] = []
        memory_segments, used = self._fit(
            [(memory_id(item.provenance), item.content) for item in retrieved],
            used,
            dropped,
        )
        recent_first, used = self._fit(
            [(observation_id(obs.step), render_observation(obs)) for obs in reversed(history)],
            used,
            dropped,
        )

        segments = (
            protected
            + [(ROLE_MEMORY, text) for text in memory_segments]
            + [(ROLE_OBSERVATION, text) for text in reversed(recent_first)]
        )
        return ContextWindow(segments=segments, token_count=used, dropped=dropped)

    def _fit(
        self,
        candidates: list[tuple[str, str]],
        used: int,
        dropped: list[str],
    ) -> tuple[list[str], int]:
        """Take candidates in order while they fit; name the rest in `dropped`."""
        kept: list[str] = []
        for identifier, text in candidates:
            cost = self.count_tokens(text)
            if used + cost > self.token_budget:
                dropped.append(dropped_entry(identifier, REASON_BUDGET))
                continue
            kept.append(text)
            used += cost
        return kept, used
