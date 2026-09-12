"""Results store. One row per run, in SQLite.

Transactional, so checkpoint and resume are free, and no service to run. Writes are
idempotent per `run_id`: a retry after a crash must never produce a second row, because a
duplicated row silently corrupts every downstream statistic. See SECURITY.md section 2.

Token counts are stored split by read path and write path. A memory system that wins on
read while costing fifty times as much on write has not won.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import TracebackType

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,
    task_id            TEXT NOT NULL,
    condition          TEXT NOT NULL,
    token_budget       INTEGER NOT NULL,
    seed               INTEGER NOT NULL,
    success            INTEGER NOT NULL,
    score              REAL NOT NULL,
    tokens_read        INTEGER NOT NULL,
    tokens_write       INTEGER NOT NULL,
    steps              INTEGER NOT NULL,
    wall_ms            REAL NOT NULL,
    config_hash        TEXT NOT NULL,
    git_commit         TEXT NOT NULL,
    payload_version    TEXT NOT NULL,
    model_id           TEXT NOT NULL,
    model_revision     TEXT NOT NULL,
    embedding_model    TEXT NOT NULL,
    embedding_revision TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_condition ON runs (condition, task_id);
"""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    task_id: str
    condition: str
    token_budget: int
    seed: int
    success: bool
    score: float
    tokens_read: int
    tokens_write: int
    steps: int
    wall_ms: float
    config_hash: str
    git_commit: str
    payload_version: str
    model_id: str
    model_revision: str
    embedding_model: str
    embedding_revision: str


class ResultsStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if self._path.parent != Path(""):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def record(self, result: RunResult) -> bool:
        """Insert one row. Returns False when the run was already recorded."""
        columns = [f.name for f in fields(RunResult)]
        placeholders = ", ".join(f":{name}" for name in columns)
        # The interpolated names come from the RunResult dataclass, never from input, and
        # every value is bound as a named parameter.
        cursor = self._connection.execute(
            f"INSERT OR IGNORE INTO runs ({', '.join(columns)}) VALUES ({placeholders})",  # noqa: S608
            asdict(result),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def has(self, run_id: str) -> bool:
        row = self._connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row is not None

    def completed_run_ids(self) -> set[str]:
        rows = self._connection.execute("SELECT run_id FROM runs").fetchall()
        return {row["run_id"] for row in rows}

    def get(self, run_id: str) -> RunResult | None:
        row = self._connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["success"] = bool(data["success"])
        return RunResult(**data)

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ResultsStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
