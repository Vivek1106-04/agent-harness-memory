"""FullContextManager - assembles everything, evicts nothing.

The trivial second implementation of the context layer, and the "uncapped" cell of the
Experiment 2 budget sweep. It declares an unbounded budget rather than accepting one it
would not honour: the contract asks that a manager never exceed the budget it declares, and
a manager that silently ignores a declared 8k budget would corrupt the sweep it appears in.

WARNING: Unbounded means unbounded - the per-run token cap in `limits` is the backstop here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from harness.interfaces import ContextManager
from harness.prompts import SYSTEM_INSTRUCTIONS, render_observation
from harness.registry import register
from harness.segments import ROLE_GOAL, ROLE_MEMORY, ROLE_OBSERVATION, ROLE_SYSTEM
from harness.tokens import approx_token_count
from harness.types import ContextWindow, MemoryItem, Observation

UNBOUNDED = sys.maxsize


@register("context", name="full")
class FullContextManager(ContextManager):
    token_budget: int = UNBOUNDED

    def __init__(self, count_tokens: Callable[[str], int] = approx_token_count) -> None:
        self.count_tokens = count_tokens

    def assemble(
        self,
        goal: str,
        history: list[Observation],
        retrieved: list[MemoryItem],
    ) -> ContextWindow:
        segments: list[tuple[str, str]] = [
            (ROLE_SYSTEM, SYSTEM_INSTRUCTIONS),
            (ROLE_GOAL, goal),
        ]
        segments += [(ROLE_MEMORY, item.content) for item in retrieved]
        segments += [(ROLE_OBSERVATION, render_observation(obs)) for obs in history]
        token_count = sum(self.count_tokens(text) for _, text in segments)
        return ContextWindow(segments=segments, token_count=token_count, dropped=[])
