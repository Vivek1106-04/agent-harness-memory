"""Pilot analysis: success, headroom recovered, and paired differences, with task-level CIs.

The decision rule in SPEC.md section 11 is fixed before any data is seen, and this module
implements the parts of it the pilot reports:

* **Unit of analysis is the task.** Seeds are averaged within a task first, so a task run
  twice does not count as two independent observations.
* **Cluster bootstrap over tasks.** One set of resampled task indices is drawn and shared by
  every condition, which is what makes a paired difference paired.
* **Headroom recovered** = (condition - floor) / (ceiling - floor), bootstrapped as a ratio
  so its interval reflects the uncertainty in the bracket too.
* **Forces-recall screen.** A task the floor solves in any replicate did not need memory,
  and is struck from every condition before anything else is read (week 3 memo).

Only tasks present in every condition are analysed; the rest are counted, never silently
dropped. The pilot is reported as preliminary and makes no significance claim - these
intervals are what the slide shows, not a test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from harness.results import RunResult

FLOOR = "noop"
CEILING = "oracle"
PRIMARY = ("graph", "vector")
DEFAULT_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 0
CI_LOW_PERCENTILE = 2.5
CI_HIGH_PERCENTILE = 97.5

Matrix = npt.NDArray[np.float64]


class AnalysisError(Exception):
    """Rows that cannot be analysed as one pilot. Never a silently partial report."""


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class ConditionSummary:
    condition: str
    runs: int
    success: Interval
    headroom: Interval | None
    tokens_read: float
    tokens_write: float
    retrieval_precision: float | None
    retrieval_recall: float | None


@dataclass(frozen=True)
class PilotReport:
    token_budget: int
    tasks: tuple[str, ...]
    struck: tuple[str, ...]
    incomplete: tuple[str, ...]
    conditions: tuple[ConditionSummary, ...]
    primary: tuple[str, str, Interval] | None


def analyse(
    rows: Sequence[RunResult],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PilotReport:
    _refuse_duplicates(rows)
    budgets = sorted({row.token_budget for row in rows})
    if len(budgets) != 1:
        raise AnalysisError(f"expected rows from exactly one token budget, found {budgets}")
    conditions = sorted({row.condition for row in rows})
    complete, incomplete = _complete_tasks(rows, conditions)
    struck = _floor_solved(rows, complete)
    tasks = [task for task in complete if task not in struck]
    if not tasks:
        raise AnalysisError("no task is left once the forces-recall screen is applied")

    kept = [row for row in rows if row.task_id in tasks]
    success = {c: _task_means(kept, c, tasks) for c in conditions}
    draws = np.random.default_rng(seed).integers(0, len(tasks), size=(resamples, len(tasks)))
    boot = {c: success[c][draws].mean(axis=1) for c in conditions}

    return PilotReport(
        token_budget=budgets[0],
        tasks=tuple(tasks),
        struck=tuple(struck),
        incomplete=tuple(incomplete),
        conditions=tuple(_summary(kept, c, success, boot) for c in conditions),
        primary=_primary(success, boot),
    )


def _refuse_duplicates(rows: Sequence[RunResult]) -> None:
    """Two rows for one (condition, budget, task, seed) - the same cell under two run-id
    prefixes - would be averaged as if they were two replicates. Refuse rather than inflate n."""
    seen: set[tuple[str, int, str, int]] = set()
    for row in rows:
        cell = (row.condition, row.token_budget, row.task_id, row.seed)
        if cell in seen:
            raise AnalysisError(f"duplicate rows for condition/budget/task/seed {cell}")
        seen.add(cell)


def _complete_tasks(
    rows: Sequence[RunResult], conditions: list[str]
) -> tuple[list[str], list[str]]:
    by_condition = {c: {row.task_id for row in rows if row.condition == c} for c in conditions}
    every = sorted({row.task_id for row in rows})
    complete = [task for task in every if all(task in seen for seen in by_condition.values())]
    return complete, [task for task in every if task not in complete]


def _floor_solved(rows: Sequence[RunResult], tasks: list[str]) -> list[str]:
    solved = {row.task_id for row in rows if row.condition == FLOOR and row.success}
    return [task for task in tasks if task in solved]


def _task_means(rows: Sequence[RunResult], condition: str, tasks: list[str]) -> Matrix:
    """Per-task success rate averaged over replicates, in `tasks` order."""
    means = []
    for task in tasks:
        outcomes = [
            row.success for row in rows if row.condition == condition and row.task_id == task
        ]
        means.append(sum(outcomes) / len(outcomes))
    return np.asarray(means, dtype=np.float64)


def _interval(point: float, draws: Matrix) -> Interval:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return Interval(point, float("nan"), float("nan"))
    low, high = np.percentile(finite, [CI_LOW_PERCENTILE, CI_HIGH_PERCENTILE])
    return Interval(point, float(low), float(high))


def _headroom(
    condition: str, success: dict[str, Matrix], boot: dict[str, Matrix]
) -> Interval | None:
    if FLOOR not in success or CEILING not in success:
        return None
    span = success[CEILING].mean() - success[FLOOR].mean()
    if span == 0:
        return None
    point = float((success[condition].mean() - success[FLOOR].mean()) / span)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = (boot[condition] - boot[FLOOR]) / (boot[CEILING] - boot[FLOOR])
    return _interval(point, draws)


def _summary(
    rows: Sequence[RunResult],
    condition: str,
    success: dict[str, Matrix],
    boot: dict[str, Matrix],
) -> ConditionSummary:
    mine = [row for row in rows if row.condition == condition]
    return ConditionSummary(
        condition=condition,
        runs=len(mine),
        success=_interval(float(success[condition].mean()), boot[condition]),
        headroom=_headroom(condition, success, boot),
        tokens_read=float(np.mean([row.tokens_read for row in mine])),
        tokens_write=float(np.mean([row.tokens_write for row in mine])),
        retrieval_precision=_mean_defined([row.retrieval_precision for row in mine]),
        retrieval_recall=_mean_defined([row.retrieval_recall for row in mine]),
    )


def _primary(
    success: dict[str, Matrix], boot: dict[str, Matrix]
) -> tuple[str, str, Interval] | None:
    """The one comparison declared in advance: graph minus vector, paired on tasks."""
    first, second = PRIMARY
    if first not in success or second not in success:
        return None
    point = float((success[first] - success[second]).mean())
    return first, second, _interval(point, boot[first] - boot[second])


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def render(report: PilotReport) -> str:
    """Markdown for the sweep memo. Preliminary by construction, and it says so."""
    lines = [
        f"Pilot at token budget {report.token_budget}: {len(report.tasks)} tasks analysed, "
        f"{len(report.struck)} struck by the forces-recall screen, "
        f"{len(report.incomplete)} incomplete. Preliminary - no significance claimed.",
        "",
        "| Condition | Runs | Success [95% CI] | Headroom [95% CI] | Read tok | Write tok "
        "| Ret. P | Ret. R |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_row(summary) for summary in report.conditions)
    if report.primary is not None:
        first, second, diff = report.primary
        lines += ["", f"Primary comparison, {first} - {second} (paired): {_fmt(diff)}"]
    return "\n".join(lines) + "\n"


def _row(s: ConditionSummary) -> str:
    headroom = _fmt(s.headroom) if s.headroom is not None else "-"
    return (
        f"| {s.condition} | {s.runs} | {_fmt(s.success)} | {headroom} | "
        f"{s.tokens_read:.0f} | {s.tokens_write:.0f} | {_opt(s.retrieval_precision)} | "
        f"{_opt(s.retrieval_recall)} |"
    )


def _fmt(interval: Interval) -> str:
    return f"{interval.estimate:.2f} [{interval.low:.2f}, {interval.high:.2f}]"


def _opt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"
