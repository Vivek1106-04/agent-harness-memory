from __future__ import annotations

from pathlib import Path

import pytest

from harness.components.context_budgeted import BudgetedContextManager
from harness.components.evaluator_checker import CheckerEvaluator
from harness.components.memory_graph import GraphMemory
from harness.components.memory_noop import NoOpMemory
from harness.components.memory_oracle import OracleMemory
from harness.components.tools_task_docs import TaskDocumentRuntime
from harness.interfaces import ContextManager, Evaluator, MemoryProvider, Planner, ToolRuntime
from harness.loop import (
    STOP_ANSWERED,
    STOP_MAX_STEPS,
    STOP_PLANNER_TERMINATED,
    STOP_TOKEN_LIMIT,
    AgentLoop,
    LoopOutcome,
)
from harness.prompts import SYSTEM_INSTRUCTIONS
from harness.tasks.generator import GeneratedTask, generate
from harness.tokens import approx_token_count
from harness.trace import MemoryTraceSink, TraceBus
from harness.types import (
    Action,
    ContextWindow,
    MemoryItem,
    MemoryQuery,
    Observation,
    TaskSpec,
    Verdict,
)
from tests.fakes import DocumentReadingPlanner
from tests.tasks.specs import spec

#: Room for the protected segments plus roughly two more segments. Small enough that a fact
#: read early in the run is gone from the window by the time the answer is due.
TIGHT_BUDGET_SLACK = 60


class RecordingMemory(MemoryProvider):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def prepare(self, task: TaskSpec) -> None:
        self._calls.append("memory.prepare")

    def write(self, item: MemoryItem) -> None:
        self._calls.append("memory.write")

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        self._calls.append("memory.retrieve")
        return []

    def reset(self) -> None:
        self._calls.append("memory.reset")


class RecordingContext(ContextManager):
    token_budget = 1000

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def assemble(
        self, goal: str, history: list[Observation], retrieved: list[MemoryItem]
    ) -> ContextWindow:
        self._calls.append(f"context.assemble({len(history)} observations)")
        return ContextWindow(segments=[("goal", goal)], token_count=7, dropped=[])


class ScriptedPlanner(Planner):
    def __init__(self, calls: list[str], actions: list[Action | None]) -> None:
        self._calls = calls
        self._actions = list(actions)

    def next_action(self, window: ContextWindow) -> Action | None:
        self._calls.append("planner.next_action")
        return self._actions.pop(0) if self._actions else None


class RecordingTools(ToolRuntime):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def prepare(self, task: TaskSpec) -> None:
        self._calls.append("tools.prepare")

    def execute(self, action: Action, step: int) -> Observation:
        self._calls.append("tools.execute")
        return Observation(step=step, content="observed", tool_name=action.tool_name)


class RecordingEvaluator(Evaluator):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def judge(self, task: TaskSpec, trace: list[Observation]) -> Verdict:
        self._calls.append("evaluator.judge")
        return Verdict(success=False, score=0.0, per_step=[], reason="recorded")


def _loop(
    calls: list[str],
    actions: list[Action | None],
    max_steps: int = 5,
    sink: MemoryTraceSink | None = None,
    max_tokens_per_run: int | None = None,
) -> AgentLoop:
    return AgentLoop(
        planner=ScriptedPlanner(calls, actions),
        context=RecordingContext(calls),
        memory=RecordingMemory(calls),
        tools=RecordingTools(calls),
        evaluator=RecordingEvaluator(calls),
        bus=TraceBus("run-1", sink or MemoryTraceSink()),
        max_steps=max_steps,
        max_tokens_per_run=max_tokens_per_run,
    )


def _task() -> GeneratedTask:
    return generate(spec())


def _echo(step: int) -> Action:
    return Action(tool_name="echo", arguments={"text": f"step {step}"})


def test_layers_are_called_in_the_frozen_order() -> None:
    """Retrieve before assemble, write after execute. Changing this changes the experiment."""
    # Arrange
    calls: list[str] = []
    loop = _loop(calls, [_echo(1), None])

    # Act
    loop.run(_task())

    # Assert
    assert calls == [
        "memory.reset",
        "memory.prepare",
        "tools.prepare",
        "memory.retrieve",
        "context.assemble(0 observations)",
        "planner.next_action",
        "tools.execute",
        "memory.write",
        "memory.retrieve",
        "context.assemble(1 observations)",
        "planner.next_action",
        "evaluator.judge",
    ]


def test_a_planner_returning_none_stops_the_run() -> None:
    # Arrange
    calls: list[str] = []

    # Act
    outcome = _loop(calls, [None]).run(_task())

    # Assert
    assert outcome.stop_reason == STOP_PLANNER_TERMINATED
    assert outcome.steps == 0


def test_a_successful_answer_stops_the_run() -> None:
    # Arrange
    calls: list[str] = []
    answer = Action(tool_name="answer", arguments={"text": "4271"})

    # Act
    outcome = _loop(calls, [answer, _echo(2)]).run(_task())

    # Assert
    assert outcome.stop_reason == STOP_ANSWERED
    assert outcome.steps == 1


def test_the_step_cap_is_the_runtime_backstop() -> None:
    """The planner stops on its own budget; the runtime never trusts it to."""
    # Arrange
    calls: list[str] = []
    loop = _loop(calls, [_echo(i) for i in range(10)], max_steps=3)

    # Act
    outcome = loop.run(_task())

    # Assert
    assert outcome.stop_reason == STOP_MAX_STEPS
    assert outcome.steps == 3


def test_read_and_write_token_costs_are_reported_separately() -> None:
    """A memory that wins on read while costing fifty times as much on write has not won."""
    # Arrange
    calls: list[str] = []

    # Act
    outcome = _loop(calls, [_echo(1), _echo(2), None]).run(_task())

    # Assert
    # Three assemblies at seven tokens: the step the planner terminates on is assembled too,
    # and it is a real prompt the study pays for.
    assert outcome.tokens_read == 21
    assert outcome.tokens_write == 2 * approx_token_count("observed")


def test_every_layer_reaches_the_trace_bus() -> None:
    # Arrange
    calls: list[str] = []
    sink = MemoryTraceSink()

    # Act
    _loop(calls, [_echo(1), None], sink=sink).run(_task())

    # Assert
    emitted = {(event.layer, event.kind) for event in sink.events}
    assert emitted == {
        ("memory", "retrieval"),
        ("memory", "write"),
        ("context", "assembly"),
        ("planner", "action"),
        ("tools", "observation"),
        ("evaluator", "verdict"),
    }


def test_written_items_carry_the_run_id_and_step() -> None:
    # Arrange
    calls: list[str] = []
    sink = MemoryTraceSink()

    # Act
    _loop(calls, [_echo(1), None], sink=sink).run(_task())

    # Assert
    writes = [event for event in sink.events if event.kind == "write"]
    assert writes[0].payload["provenance"] == "run-1:1"


def test_max_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_steps must be positive"):
        _loop([], [], max_steps=0)


def _bracket_run(memory: MemoryProvider, sandbox: Path) -> tuple[Verdict, int]:
    outcome = _bracket_outcome(memory, sandbox)
    return outcome.verdict, outcome.steps


def _bracket_outcome(
    memory: MemoryProvider, sandbox: Path, subject: TaskSpec | None = None
) -> LoopOutcome:
    """One task, one planner, one budget. Only the memory layer changes."""
    subject = spec() if subject is None else subject
    task = generate(subject)
    budget = (
        approx_token_count(SYSTEM_INSTRUCTIONS) + approx_token_count(task.goal) + TIGHT_BUDGET_SLACK
    )
    loop = AgentLoop(
        planner=DocumentReadingPlanner(
            [name for name, _ in task.documents], task.required[-1].entity
        ),
        context=BudgetedContextManager(budget),
        memory=memory,
        tools=TaskDocumentRuntime(sandbox),
        evaluator=CheckerEvaluator(),
        bus=TraceBus(f"bracket-{memory.__class__.__name__}", MemoryTraceSink()),
        max_steps=task.answer_step + 1,
    )
    return loop.run(task)


def test_noop_is_the_floor_of_the_bracket(tmp_path: Path) -> None:
    """The required fact is evicted long before the answer is due, so the run must fail.

    This is the forces-recall assertion in miniature: a task that passed here would be
    solvable without memory and would contribute pure noise to the study.
    """
    # Act
    verdict, steps = _bracket_run(NoOpMemory(), tmp_path)

    # Assert
    assert verdict.success is False
    assert steps == generate(spec()).answer_step + 1


def test_oracle_is_the_ceiling_of_the_bracket(tmp_path: Path) -> None:
    """Same task, same budget, same planner. Perfect retrieval turns the failure into a pass."""
    # Act
    verdict, _ = _bracket_run(OracleMemory(), tmp_path)

    # Assert
    assert verdict.success is True
    assert verdict.score == 1.0


def test_graph_memory_recovers_the_evicted_fact(tmp_path: Path) -> None:
    """The goal names the terminal entity; one hop reaches the fact it depends on.

    Entities are identifier-shaped, as `tasks.suite` names them: that is the surface form
    the graph's lexical extractor recognises (DECISIONS.md).
    """
    # Arrange
    subject = spec(pool=[f"site_000_{slot:02d}" for slot in range(4)])

    # Act
    outcome = _bracket_outcome(GraphMemory(hops=1), tmp_path, subject)

    # Assert
    assert outcome.verdict.success is True
    assert outcome.retrieval_recall_final == 1.0


def test_retrieval_scores_bracket_the_same_way_success_does(tmp_path: Path) -> None:
    """The ceiling is perfect by construction; the floor never retrieves anything."""
    # Act
    ceiling = _bracket_outcome(OracleMemory(), tmp_path / "oracle")
    floor = _bracket_outcome(NoOpMemory(), tmp_path / "noop")

    # Assert
    assert (ceiling.retrieval_precision, ceiling.retrieval_recall) == (1.0, 1.0)
    assert floor.retrieval_precision is None
    assert floor.retrieval_recall == 0.0
    assert (ceiling.retrieval_recall_final, floor.retrieval_recall_final) == (1.0, 0.0)


def test_padded_documents_make_an_8k_budget_force_recall(tmp_path: Path) -> None:
    """Regression for the week-3 finding: one-sentence documents all fit in 8k, so NoOp
    solved every task. Padded to the shipped `document_tokens`, the floor fails again."""
    # Arrange
    subject = spec(
        graph={"n_1": ["n_0"]},
        recall_distance=12,
        horizon=20,
        pool=[f"site_000_{slot:02d}" for slot in range(10)],
        document_tokens=900,
    )
    task = generate(subject)

    def run(memory: MemoryProvider, sandbox: Path) -> bool:
        loop = AgentLoop(
            planner=DocumentReadingPlanner(
                [name for name, _ in task.documents], task.required[-1].entity
            ),
            context=BudgetedContextManager(8000),
            memory=memory,
            tools=TaskDocumentRuntime(sandbox),
            evaluator=CheckerEvaluator(),
            bus=TraceBus("padded", MemoryTraceSink()),
            max_steps=35,
        )
        return loop.run(task).verdict.success

    # Act / Assert
    assert run(NoOpMemory(), tmp_path / "noop") is False
    assert run(OracleMemory(), tmp_path / "oracle") is True


def test_the_token_limit_stops_the_run_before_the_next_call() -> None:
    """limits.max_tokens_per_run is a kill-switch, not a comment in a YAML file."""
    # Arrange
    loop = _loop([], [_echo(i) for i in range(10)], max_steps=10, max_tokens_per_run=1)

    # Act
    outcome = loop.run(_task())

    # Assert
    assert outcome.stop_reason == STOP_TOKEN_LIMIT
    assert outcome.steps == 1


def test_an_observation_is_written_to_memory_one_sentence_at_a_time(tmp_path: Path) -> None:
    """Chunking is the loop's, not a provider's: every condition stores the same items."""
    # Arrange
    written: list[MemoryItem] = []

    class Capturing(NoOpMemory):
        def write(self, item: MemoryItem) -> None:
            written.append(item)

    task = generate(spec(document_tokens=60))
    loop = AgentLoop(
        planner=DocumentReadingPlanner([task.documents[0][0]], task.required[-1].entity),
        context=BudgetedContextManager(8000),
        memory=Capturing(),
        tools=TaskDocumentRuntime(tmp_path),
        evaluator=CheckerEvaluator(),
        bus=TraceBus("chunks", MemoryTraceSink()),
        max_steps=1,
    )

    # Act
    loop.run(task)

    # Assert
    assert len(written) > 1
    assert {item.provenance for item in written} == {"chunks:1"}
    assert " ".join(item.content for item in written) == task.documents[0][1]
