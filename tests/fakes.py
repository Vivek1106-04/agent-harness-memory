"""Test doubles. Unit tests never call a real LLM and never touch the network."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from harness.embedding import Vectors, l2_normalise
from harness.interfaces import LLMClient, Planner
from harness.tasks.generator import ANSWER_TOOL
from harness.tokens import approx_token_count
from harness.types import Action, ContextWindow

HASHING_ENCODER_DIM = 64


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


class HashingEncoder:
    """Deterministic bag-of-words hashing into a fixed number of buckets.

    Stands in for the sentence-transformers encoder so that the vector memory can be
    tested without a model download. `hashlib` rather than `hash()` on purpose: Python
    salts string hashing per process, and a fake whose vectors changed between runs would
    make the determinism test in the memory contract pass or fail at random.
    """

    def __init__(self, dim: int = HASHING_ENCODER_DIM) -> None:
        self._dim = dim

    def encode(self, texts: Sequence[str]) -> Vectors:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                matrix[row, self._bucket(token)] += 1.0
        return l2_normalise(matrix)

    def encode_queries(self, texts: Sequence[str]) -> Vectors:
        return self.encode(texts)

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode()).digest()
        return int.from_bytes(digest[:4], "big") % self._dim


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
