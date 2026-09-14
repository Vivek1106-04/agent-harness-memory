"""The runtime loop. One task, start to finish, through all five layers.

The loop owns sequencing and nothing else. It names no implementation, makes no retrieval
or eviction decision, and scores nothing itself - every one of those belongs to a layer that
can be swapped. What it does own is the order, which is frozen because changing it changes
the experiment:

    reset -> prepare -> (retrieve -> assemble -> plan -> execute -> write)* -> judge

Retrieval happens before assembly so that the context manager sees the memory items and can
account for them against the budget. Writing happens after execution so an observation is
never retrievable on the same step it was produced - a memory that could do that would be
reading the present, not remembering the past.

Token cost is accumulated on the read path and the write path separately: a memory that
wins on read while costing fifty times as much on write has not won (SPEC.md section 8).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from harness.interfaces import ContextManager, Evaluator, MemoryProvider, Planner, ToolRuntime
from harness.tasks.generator import ANSWER_TOOL, GeneratedTask
from harness.tokens import approx_token_count
from harness.trace import TraceBus
from harness.types import MemoryItem, MemoryQuery, Observation, Verdict

DEFAULT_RETRIEVAL_K = 5

KIND_EPISODIC = "episodic"
KIND_FAILURE = "failure"

LAYER_MEMORY = "memory"
LAYER_CONTEXT = "context"
LAYER_PLANNER = "planner"
LAYER_TOOLS = "tools"
LAYER_EVALUATOR = "evaluator"

STOP_PLANNER_TERMINATED = "planner_terminated"
STOP_ANSWERED = "answered"
STOP_MAX_STEPS = "max_steps"


@dataclass(frozen=True)
class LoopOutcome:
    """Everything a results row needs, and nothing the trace does not already carry."""

    verdict: Verdict
    trace: tuple[Observation, ...]
    steps: int
    tokens_read: int
    tokens_write: int
    wall_ms: float
    stop_reason: str


class AgentLoop:
    def __init__(
        self,
        planner: Planner,
        context: ContextManager,
        memory: MemoryProvider,
        tools: ToolRuntime,
        evaluator: Evaluator,
        bus: TraceBus,
        max_steps: int,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        count_tokens: Callable[[str], int] = approx_token_count,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self._planner = planner
        self._context = context
        self._memory = memory
        self._tools = tools
        self._evaluator = evaluator
        self._bus = bus
        self._max_steps = max_steps
        self._retrieval_k = retrieval_k
        self._count_tokens = count_tokens

    def run(self, task: GeneratedTask) -> LoopOutcome:
        started = time.perf_counter()
        self._memory.reset()
        self._memory.prepare(task.spec)
        self._tools.prepare(task.spec)

        history: list[Observation] = []
        tokens_read = 0
        tokens_write = 0
        stop_reason = STOP_MAX_STEPS
        step = 0

        for step in range(1, self._max_steps + 1):
            retrieved = self._retrieve(task.goal, step)
            window = self._context.assemble(task.goal, history, retrieved)
            tokens_read += window.token_count
            self._bus.emit(
                LAYER_CONTEXT,
                "assembly",
                step,
                {"segments": len(window.segments), "dropped": window.dropped},
                {"read": window.token_count},
            )

            action = self._planner.next_action(window)
            if action is None:
                stop_reason = STOP_PLANNER_TERMINATED
                break
            self._bus.emit(
                LAYER_PLANNER,
                "action",
                step,
                {"tool_name": action.tool_name, "arguments": action.arguments},
            )

            observation = self._tools.execute(action, step)
            history.append(observation)
            self._bus.emit(
                LAYER_TOOLS,
                "observation",
                step,
                {"tool_name": observation.tool_name, "error": observation.error},
            )
            tokens_write += self._write(observation, step)

            if action.tool_name == ANSWER_TOOL and observation.error is None:
                stop_reason = STOP_ANSWERED
                break

        verdict = self._evaluator.judge(task.spec, history)
        self._bus.emit(
            LAYER_EVALUATOR,
            "verdict",
            step,
            {"success": verdict.success, "score": verdict.score, "reason": verdict.reason},
        )
        return LoopOutcome(
            verdict=verdict,
            trace=tuple(history),
            steps=len(history),
            tokens_read=tokens_read,
            tokens_write=tokens_write,
            wall_ms=(time.perf_counter() - started) * 1000.0,
            stop_reason=stop_reason,
        )

    def _retrieve(self, goal: str, step: int) -> list[MemoryItem]:
        items = self._memory.retrieve(MemoryQuery(text=goal, k=self._retrieval_k))
        self._bus.emit(
            LAYER_MEMORY,
            "retrieval",
            step,
            {"k": self._retrieval_k, "returned": [item.provenance for item in items]},
        )
        return items

    def _write(self, observation: Observation, step: int) -> int:
        """Returns the write-path token cost of storing this observation."""
        item = MemoryItem(
            content=observation.content,
            score=1.0,
            provenance=f"{self._bus.run_id}:{step}",
            kind=KIND_FAILURE if observation.error is not None else KIND_EPISODIC,
        )
        cost = self._count_tokens(item.content)
        self._memory.write(item)
        self._bus.emit(
            LAYER_MEMORY, "write", step, {"provenance": item.provenance}, {"write": cost}
        )
        return cost
