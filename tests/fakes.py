"""Test doubles. Unit tests never call a real LLM and never touch the network."""

from __future__ import annotations

from harness.interfaces import LLMClient
from harness.tokens import approx_token_count
from harness.types import ContextWindow


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
