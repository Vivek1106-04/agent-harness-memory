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
    embedding_revision TEXT NOT NULL,
    retrieval_precision REAL,
    retrieval_recall   REAL,
    retrieval_recall_final REAL,
    cost_usd           REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_by_condition ON runs (condition, task_id);
"""

#: Columns added after databases already existed. `CREATE TABLE IF NOT EXISTS` leaves an old
#: table as it was, so each is added on open when missing. Nullable, so old rows read None.
ADDED_COLUMNS: tuple[tuple[str, str], ...] = (("retrieval_recall_final", "REAL"),)


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
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    cost_usd: float = 0.0
    retrieval_recall_final: float | None = None


class ResultsStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if self._path.parent != Path(""):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._add_missing_columns()
        self._connection.commit()

    def _add_missing_columns(self) -> None:
        present = {row["name"] for row in self._connection.execute("PRAGMA table_info(runs)")}
        for name, sql_type in ADDED_COLUMNS:
            if name not in present:
                # Both names come from ADDED_COLUMNS above, never from input.
                self._connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {sql_type}")

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
        return _result(row)

    def all(self) -> list[RunResult]:
        rows = self._connection.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
        return [_result(row) for row in rows]

    def total_cost_usd(self) -> float:
        """Spend already recorded. Seeds the spend limit, so a resumed sweep keeps counting."""
        row = self._connection.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS c FROM runs"
        ).fetchone()
        return float(row["c"])

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


def _result(row: sqlite3.Row) -> RunResult:
    data = dict(row)
    data["success"] = bool(data["success"])
    return RunResult(**data)
