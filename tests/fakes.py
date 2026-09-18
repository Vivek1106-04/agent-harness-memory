"""Test doubles. Unit tests never call a real LLM and never touch the network."""

from __future__ import annotations

import re

from harness.interfaces import LLMClient, Planner
from harness.tasks.generator import ANSWER_TOOL
from harness.tokens import approx_token_count
from harness.types import Action, ContextWindow


class FakeLLMClient(LLMClient):
    """Replays a scripted list of responses, repeating the last one once exhausted.

    Repeating rather than raising keeps planner contract tests honest: a planner is allowed
    to call the model more often than a test anticipated, and that is not a failure.
    """

    def __init__(self, responses: list[str]) -> None:
        if not responses:
            raise ValueError("FakeLLMClient needs at least one scripted response")
        self._responses = list(responses)
        self.calls: list[ContextWindow] = []

    def complete(self, window: ContextWindow, max_output_tokens: int) -> str:
        self.calls.append(window)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]

    def count_tokens(self, text: str) -> int:
        return approx_token_count(text)


class DocumentReadingPlanner(Planner):
    """Reads a document list in order, then answers with the code it can still see.

    A perfect reader with no reasoning and no memory of its own: it answers correctly
    exactly when the fact is in the window it was handed. That isolates the memory layer as
    the only thing that varies between the floor and the ceiling of the bracket, which is
    what makes an offline bracket test meaningful before a model endpoint exists.
    """

    UNKNOWN = "unknown"

    def __init__(self, doc_names: list[str], entity: str) -> None:
        self._doc_names = list(doc_names)
        self._pattern = re.compile(rf"access code for {re.escape(entity)} is (\d+)")
        self._index = 0
        self._answered = False

    def next_action(self, window: ContextWindow) -> Action | None:
        if self._index < len(self._doc_names):
            name = self._doc_names[self._index]
            self._index += 1
            return Action(tool_name="read_file", arguments={"path": name})
        if self._answered:
            return None
        self._answered = True
        match = self._pattern.search("\n".join(text for _, text in window.segments))
        return Action(
            tool_name=ANSWER_TOOL,
            arguments={"text": match.group(1) if match else self.UNKNOWN},
        )
