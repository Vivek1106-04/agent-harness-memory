"""The five layer interfaces.

Every layer meets the same bar: frozen payload contract, this interface, at least two
implementations, a conformance suite, a config-registry entry, trace emission, a named
owner. No layer is a stub written to make another layer measurable.

This module imports nothing from any implementation. If it ever does, substitutability is
broken.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.types import (
    Action,
    ContextWindow,
    MemoryItem,
    MemoryQuery,
    Observation,
    TaskSpec,
    Verdict,
)


class MemoryProvider(ABC):
    """The layer under test.

    `capabilities` declares what an implementation honours rather than leaving it implied:
    a subset of {"temporal", "graph", "semantic"}. An implementation ignores query fields
    it does not support; one that raises on them is inadmissible.
    """

    capabilities: frozenset[str] = frozenset()

    def prepare(self, task: TaskSpec) -> None:  # noqa: B027 - a no-op default is the point
        """Bind the provider to the task about to run. Default: ignore it.

        The runtime calls this once per run, after `reset`. It exists for the measurement
        baselines: `OracleMemory` is the perfect-retrieval ceiling, and a ceiling has to
        know what perfect means. Task generation is a pure function of the spec, so no side
        channel to the generator is needed - the provider re-derives the task itself.

        A memory implementation under test that reads gold labels here is cheating by
        construction. Baselines only. See DECISIONS.md.
        """

    @abstractmethod
    def write(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]: ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all state: indices, caches, vectors, graph nodes, handles.

        Load-bearing. State leaking between runs is both a correctness bug and a
        cross-run contamination bug, and it produces plausible-looking wrong numbers.
        """
        ...


class Planner(ABC):
    @abstractmethod
    def next_action(self, window: ContextWindow) -> Action | None:
        """Return the next action, or None to terminate.

        Malformed model output is an expected condition, not an exception.
        """
        ...


class ContextManager(ABC):
    """Assembles the prompt under a hard budget imposed independently of model capability.

    The entire context-budget argument and the whole of Experiment 2 rest on this budget
    actually binding. System instructions and the goal are never evictable.
    """

    token_budget: int

    @abstractmethod
    def assemble(
        self,
        goal: str,
        history: list[Observation],
        retrieved: list[MemoryItem],
    ) -> ContextWindow: ...


class ToolRuntime(ABC):
    def prepare(self, task: TaskSpec) -> None:  # noqa: B027 - a no-op default is the point
        """Bind the runtime to the task about to run. Default: ignore it.

        The task environment - which documents exist and what they say - reaches the tool
        layer through here, and through nothing else. The runtime re-derives it from the
        spec, so no runtime object has to be threaded through config.
        """

    @abstractmethod
    def execute(self, action: Action, step: int) -> Observation:
        """Execute an action. Errors become Observations; nothing raises to the loop.

        Treat every action as hostile input: allowlist the tool, validate arguments, and
        confine every filesystem operation to the per-run sandbox. See SECURITY.md section 5.
        """
        ...


class Evaluator(ABC):
    @abstractmethod
    def judge(self, task: TaskSpec, trace: list[Observation]) -> Verdict:
        """Deterministic and offline. Same trace always yields the same verdict.

        No LLM-as-judge: it reintroduces the confound this study exists to remove, and a
        judge reading model-generated traces is directly injectable.
        """
        ...


class LLMClient(ABC):
    """Abstracts local vLLM and OpenAI-compatible hosted endpoints.

    Thin by design - about sixty lines beats a framework dependency, and it keeps the
    "everything is an interface" property honest.
    """

    @abstractmethod
    def complete(self, window: ContextWindow, max_output_tokens: int) -> str: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...
