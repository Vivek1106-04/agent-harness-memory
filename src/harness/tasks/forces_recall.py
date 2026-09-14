"""The forces-recall assertion: a test, not an assumption (SPEC.md section 9).

A task solvable from the recent window contributes pure noise. It looks exactly like a task
where memory genuinely did not matter, so it flattens every condition toward the same number
and quietly costs statistical power. Every generated task therefore has to earn admission.

Two gates, cheap one first:

1. **Static.** `recall_distance` must exceed the number of observations the NoOp condition
   can still see. If the fact is inside that window it is not recalled, it is read.
2. **Empirical.** Run the task under `NoOpMemory` with that same window. It must fail. A
   task that succeeds is rejected, whatever the geometry claimed.

The static gate is not a substitute for the empirical one - it catches the specs that cannot
possibly force recall, so the expensive gate only spends model calls on plausible ones.

The caller supplies `run_under_noop`, which owns the model, the budget and the components.
That keeps this module free of any dependency on a model endpoint, so the rejection rule is
testable offline and identical whichever runner executes it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from harness.types import TaskSpec, Verdict

REASON_ACCEPTED = "fails under NoOp: the required fact is outside the visible window"
REASON_SOLVED_UNDER_NOOP = "solved under NoOp: the task does not force recall"
REASON_WINDOW_TOO_WIDE = (
    "recall_distance {distance} does not exceed the NoOp window of {window} observations"
)

RunUnderNoOp = Callable[[TaskSpec], Verdict]


@dataclass(frozen=True)
class ForcesRecallReport:
    task_id: str
    accepted: bool
    reason: str


def static_gate(spec: TaskSpec, noop_window: int) -> str | None:
    """Return a rejection reason, or None when the geometry could force recall."""
    if spec.recall_distance <= noop_window:
        return REASON_WINDOW_TOO_WIDE.format(distance=spec.recall_distance, window=noop_window)
    return None


def check(spec: TaskSpec, run_under_noop: RunUnderNoOp, noop_window: int) -> ForcesRecallReport:
    """Admit `spec` only if it fails under NoOp. Runs the model only when it might pass."""
    rejection = static_gate(spec, noop_window)
    if rejection is not None:
        return ForcesRecallReport(task_id=spec.task_id, accepted=False, reason=rejection)
    verdict = run_under_noop(spec)
    if verdict.success:
        return ForcesRecallReport(
            task_id=spec.task_id, accepted=False, reason=REASON_SOLVED_UNDER_NOOP
        )
    return ForcesRecallReport(task_id=spec.task_id, accepted=True, reason=REASON_ACCEPTED)


def filter_specs(
    specs: Iterable[TaskSpec],
    run_under_noop: RunUnderNoOp,
    noop_window: int,
) -> tuple[list[TaskSpec], list[ForcesRecallReport]]:
    """Partition a candidate set. Every report is kept, accepted and rejected alike.

    The rejection rate is itself a result: a task family that mostly fails this gate is a
    family whose design does not depend on memory, and that is worth knowing in week 3
    rather than week 10.
    """
    candidates = list(specs)
    reports = [check(spec, run_under_noop, noop_window) for spec in candidates]
    accepted = [spec for spec, report in zip(candidates, reports, strict=True) if report.accepted]
    return accepted, reports
