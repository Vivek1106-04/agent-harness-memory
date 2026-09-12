"""SingleStepPlanner - one model call, one answer, no tool loop.

The trivial second implementation of the planner layer, and a free ablation baseline: it
measures what the task set yields with no iterative planning at all. Whatever the model
returns is treated as the answer text, so malformed output is impossible by construction
rather than by rescue.
"""

from __future__ import annotations

from harness.interfaces import LLMClient, Planner
from harness.registry import register
from harness.types import Action, ContextWindow

ANSWER_TOOL = "answer"
DEFAULT_MAX_OUTPUT_TOKENS = 300


@register("planner", name="single_step")
class SingleStepPlanner(Planner):
    def __init__(
        self,
        llm: LLMClient,
        max_steps: int = 1,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self._llm = llm
        self._max_steps = max_steps
        self._max_output_tokens = max_output_tokens
        self._steps_taken = 0

    def next_action(self, window: ContextWindow) -> Action | None:
        if self._steps_taken >= min(1, self._max_steps):
            return None
        self._steps_taken += 1
        text = self._llm.complete(window, self._max_output_tokens)
        return Action(
            tool_name=ANSWER_TOOL,
            arguments={"text": text},
            rationale="single-step baseline: answer directly from the assembled window",
        )
