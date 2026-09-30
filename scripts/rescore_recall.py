"""Backfill `retrieval_recall_final` and `retrieval_recall_dropped` for runs recorded
before those columns existed.

Each run is replayed through the real loop: the planner is replaced by one that returns the
actions recorded in the run's trace, so the tools produce the same observations, the memory
under test stores the same chunks, and the loop scores retrieval exactly as it did live.
Nothing calls a model.

A row is updated only when the replay reproduces the stored `retrieval_recall`, `steps` and
`success`. A mismatch means the replay is not the run that was recorded, and its final
recall would be a number about some other run, so the row is left alone and reported.

    uv run python scripts/rescore_recall.py results/runs.db results/traces
    uv run python scripts/rescore_recall.py results/pilot-t1.db \
        results/pilot-t1-traces --write
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

from harness.build import build_loop
from harness.cli import suite_spec
from harness.config import HarnessConfig, load_config
from harness.interfaces import Planner
from harness.results import ResultsStore
from harness.tasks.generator import GeneratedTask, generate
from harness.tasks.suite import build_suite
from harness.tokens import approx_token_count
from harness.trace import MemoryTraceSink, TraceBus
from harness.types import Action, ContextWindow

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
#: run_id prefix -> config, and for the pilot prefix, condition -> config.
BRACKET = {
    "week3_bracket_noop": "week3_bracket_noop",
    "week3_bracket_oracle": "week3_bracket_oracle",
}
PILOT_PREFIX = "pilot"
ENV = {"HARNESS_LLM_BASE_URL": "http://replay.invalid/v1"}
TOLERANCE = 1e-9


class ReplayPlanner(Planner):
    def __init__(self, actions: list[Action]) -> None:
        self._actions = list(actions)

    def next_action(self, window: ContextWindow) -> Action | None:
        return self._actions.pop(0) if self._actions else None


class NoModel:
    """build_loop wants a client for the planner and the token counter; the planner is
    replaced, and the counter is the same arithmetic count the live client uses."""

    def complete(self, window: ContextWindow, max_output_tokens: int) -> str:
        raise RuntimeError("replay must not call a model")

    def count_tokens(self, text: str) -> int:
        return approx_token_count(text)


def config_for(prefix: str, condition: str) -> HarnessConfig:
    name = BRACKET.get(prefix) or (f"pilot_{condition}_2k" if prefix == PILOT_PREFIX else None)
    if name is None:
        raise ValueError(f"no config known for run prefix {prefix!r}")
    return load_config(CONFIGS / f"{name}.yaml", env=ENV)


def recorded_actions(trace: Path) -> list[Action]:
    """Actions of the last attempt only. A cell that failed on a transport error and was
    retried appends a second attempt to the same file; the row is the last attempt's."""
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    last_header = max(i for i, event in enumerate(events) if event["kind"] == "header")
    actions = []
    for event in events[last_header:]:
        if event["kind"] == "action":
            payload = event["payload"]
            actions.append(Action(tool_name=payload["tool_name"], arguments=payload["arguments"]))
    return actions


def task_for(
    config: HarnessConfig, task_id: str, cache: dict[str, dict[str, Any]]
) -> GeneratedTask:
    key = config.suite.model_dump_json()
    if key not in cache:
        cache[key] = {spec.task_id: spec for spec in build_suite(suite_spec(config))}
    return generate(cache[key][task_id])


def replay(row: sqlite3.Row, traces: Path, cache: dict[str, dict[str, Any]]) -> tuple[Any, str]:
    run_id = row["run_id"]
    prefix = run_id.split(":", 1)[0]
    config = config_for(prefix, row["condition"])
    trace = traces / f"{run_id.replace(':', '_')}.jsonl"
    if not trace.exists():
        return None, f"no trace at {trace}"
    built = build_loop(config, TraceBus(run_id, MemoryTraceSink()), NoModel())
    built.loop._planner = ReplayPlanner(recorded_actions(trace))
    outcome = built.loop.run(task_for(config, row["task_id"], cache))
    mismatch = _mismatch(row, outcome)
    return (outcome, mismatch)


def _mismatch(row: sqlite3.Row, outcome: Any) -> str:
    problems = []
    if outcome.steps != row["steps"]:
        problems.append(f"steps {outcome.steps} != {row['steps']}")
    if bool(outcome.verdict.success) != bool(row["success"]):
        problems.append(f"success {outcome.verdict.success} != {bool(row['success'])}")
    stored, replayed = row["retrieval_recall"], outcome.retrieval_recall
    if (stored is None) != (replayed is None) or (
        stored is not None and not math.isclose(stored, replayed, abs_tol=TOLERANCE)
    ):
        problems.append(f"recall {replayed} != {stored}")
    return "; ".join(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--write", action="store_true", help="update rows; default is a dry run")
    args = parser.parse_args()

    ResultsStore(args.db).close()  # adds the column to a database that predates it
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    rows = connection.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
    cache: dict[str, dict[str, Any]] = {}
    matched, refused = 0, []
    for row in rows:
        outcome, mismatch = replay(row, args.traces, cache)
        if outcome is None or mismatch:
            refused.append(f"{row['run_id']}: {mismatch}")
            continue
        matched += 1
        if args.write:
            connection.execute(
                "UPDATE runs SET retrieval_recall_final = ?, retrieval_recall_dropped = ? "
                "WHERE run_id = ?",
                (outcome.retrieval_recall_final, outcome.retrieval_recall_dropped, row["run_id"]),
            )
    connection.commit()
    connection.close()
    print(f"{args.db}: {matched} reproduced, {len(refused)} refused, write={args.write}")
    for line in refused:
        print(f"  REFUSED {line}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
