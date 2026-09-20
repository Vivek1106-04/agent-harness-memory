"""ReActPlanner - think, act, observe, repeat. The planner the bracket actually needs.

`SingleStepPlanner` answers in one call and cannot read a document, so it cannot produce a
measurement bracket: the floor and the ceiling would both be "whatever the model already
knew". This is the looping planner, and it is what makes `recall_distance` mean anything -
the agent reads documents across many steps, and only memory carries the early ones to the
end.

Malformed output is an expected condition, not an exception (SPEC.md section 3). Three
behaviours follow from that, and the third is the one worth arguing about:

1. The action is the first JSON object in the reply, so 'Thought:' prose, fenced code
   blocks and a trailing apology all parse.
2. A reply with no usable action becomes an `invalid_action` action. The tool layer answers
   with an error observation, the error enters the history, and the model sees its own
   mistake on the next step. That is the ReAct loop doing its job, and it recovers runs a
   hard stop would throw away.
3. After `max_parse_failures` consecutive unusable replies the planner terminates. Without
   the cap a model stuck in a formatting loop spends all 35 steps failing the same way, and
   we would pay for it 1,440 times.
"""

from __future__ import annotations

import json
import re

from harness.interfaces import LLMClient, Planner
from harness.registry import register
from harness.types import Action, ContextWindow

DEFAULT_MAX_OUTPUT_TOKENS = 300
DEFAULT_MAX_PARSE_FAILURES = 3

#: The tool name an unparseable reply is turned into. No runtime implements it, so the tool
#: layer answers with its standard unknown-tool error and the model reads the error next step.
INVALID_ACTION_TOOL = "invalid_action"

RATIONALE_PREFIX = "Thought:"
#: Longest raw reply echoed back in an `invalid_action` action. A model that loops emitting
#: prose would otherwise push its own mistake into the next window at full length.
MAX_ECHOED_CHARS = 200

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_THOUGHT = re.compile(r"Thought:\s*(.+?)(?:\n|$)", re.IGNORECASE)


@register("planner", name="react")
class ReActPlanner(Planner):
    def __init__(
        self,
        llm: LLMClient,
        max_steps: int,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_parse_failures: int = DEFAULT_MAX_PARSE_FAILURES,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self._llm = llm
        self._max_steps = max_steps
        self._max_output_tokens = max_output_tokens
        self._max_parse_failures = max_parse_failures
        self._steps_taken = 0
        self._consecutive_failures = 0

    def next_action(self, window: ContextWindow) -> Action | None:
        if self._steps_taken >= self._max_steps:
            return None
        self._steps_taken += 1

        text = self._llm.complete(window, self._max_output_tokens)
        action = parse_action(text)
        if action is not None:
            self._consecutive_failures = 0
            return action

        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_parse_failures:
            return None
        return Action(
            tool_name=INVALID_ACTION_TOOL,
            arguments={"raw": text[:MAX_ECHOED_CHARS]},
            rationale="reply held no usable Action object",
        )


def parse_action(text: str) -> Action | None:
    """First JSON object in `text` that names a tool, or None.

    Tolerant on purpose about everything that does not change the action: surrounding
    prose, code fences, a missing 'arguments' key. Strict about the two things that do -
    `tool_name` must be a non-empty string and `arguments` must be a mapping, because a
    list or a string smuggled into `arguments` reaches the tool layer as unvalidated model
    output.
    """
    match = _JSON_OBJECT.search(text)
    if match is None:
        return None
    payload = _loads_longest_prefix(match.group())
    if payload is None:
        return None

    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    return Action(
        tool_name=tool_name.strip(),
        arguments=arguments,
        rationale=_rationale(text),
    )


def _loads_longest_prefix(candidate: str) -> dict[str, object] | None:
    """Parse the greedy match, then retreat to the last '}' until something parses.

    A reply holding an action followed by more prose with braces in it produces a greedy
    match that is not valid JSON. Retreating recovers the action instead of discarding a
    perfectly good step.
    """
    text = candidate
    while True:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            closing = text.rfind("}", 0, len(text) - 1)
            if closing == -1:
                return None
            text = text[: closing + 1]
            continue
        return parsed if isinstance(parsed, dict) else None


def _rationale(text: str) -> str | None:
    match = _THOUGHT.search(text)
    return match.group(1).strip() if match else None
