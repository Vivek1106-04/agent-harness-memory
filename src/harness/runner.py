"""Resumable sweep runner.

A sweep is a list of cells; each cell is executed once and its row committed immediately.
Losing four hours of completed runs to a crash is unacceptable, so resume is the default
behaviour and not a flag: a restart skips every `run_id` already in the store.

A cell that raises is left unrecorded on purpose, so that a rerun retries it. The summary
reports it rather than swallowing it.

Two conditions stop the sweep instead of moving on to the next cell. An `SweepAbortedError` - the
spend limit, for one - means every later cell would fail the same way. And a run of
consecutive failures almost always means the same thing without saying so: a revoked key or
a daily quota spent. Grinding through the remaining cells would record nothing and, on a
paid key, could still cost money.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from harness.results import ResultsStore, RunResult

DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


class SweepAbortedError(Exception):
    """Raised by a cell when no later cell can succeed either. Stops the sweep."""


@dataclass
class FailureStreak:
    """Consecutive failed cells. Shared across every config of a sweep, so a sweep of
    one-task configs still stops when the key or the quota is dead."""

    count: int = 0


@dataclass(frozen=True)
class RunPlanItem:
    """One cell of a sweep. `run_id` is the idempotency key and must be deterministic."""

    run_id: str
    task_id: str
    condition: str
    token_budget: int
    seed: int


@dataclass(frozen=True)
class RunSummary:
    executed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    aborted: list[str] = field(default_factory=list)


def build_run_id(prefix: str, condition: str, task_id: str, seed: int, token_budget: int) -> str:
    """Deterministic identity for a cell. The same cell always resolves to the same row."""
    return f"{prefix}:{condition}:{token_budget}:{task_id}:s{seed}"


class Runner:
    def __init__(
        self,
        store: ResultsStore,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        streak: FailureStreak | None = None,
    ) -> None:
        self._store = store
        self._max_consecutive_failures = max_consecutive_failures
        self._streak = streak if streak is not None else FailureStreak()

    def pending(self, plan: Sequence[RunPlanItem]) -> list[RunPlanItem]:
        done = self._store.completed_run_ids()
        return [item for item in plan if item.run_id not in done]

    def run_all(
        self,
        plan: Iterable[RunPlanItem],
        execute: Callable[[RunPlanItem], RunResult],
    ) -> RunSummary:
        summary = RunSummary()
        for item in plan:
            if self._store.has(item.run_id):
                summary.skipped.append(item.run_id)
                continue
            try:
                result = execute(item)
            except SweepAbortedError as error:
                summary.failed.append((item.run_id, f"{type(error).__name__}: {error}"))
                summary.aborted.append(str(error))
                return summary
            except Exception as error:
                summary.failed.append((item.run_id, f"{type(error).__name__}: {error}"))
                self._streak.count += 1
                if self._streak.count >= self._max_consecutive_failures:
                    summary.aborted.append(f"{self._streak.count} consecutive cells failed")
                    return summary
                continue
            self._streak.count = 0
            if result.run_id != item.run_id:
                summary.failed.append(
                    (item.run_id, f"execute returned a result for {result.run_id!r}")
                )
                continue
            self._store.record(result)
            summary.executed.append(item.run_id)
        return summary
