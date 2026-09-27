"""One task, start to finish, reproducible from a config hash - with no model endpoint.

The end-to-end deliverable, driven by a scripted client. What it proves is that config,
registry, loop, trace, provenance and results agree with one another; what a real model
adds is the only thing left, and that is what the gate run is for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from harness.cli import RunPaths, main, run_config, run_sweep
from harness.config import load_config
from harness.results import ResultsStore
from harness.tasks.generator import generate
from harness.tasks.suite import SuiteSpec, build_suite
from tests.fakes import FakeLLMClient
from tests.test_build import write_config

SUITE = {
    "count": 2,
    "chain_length": 2,
    "recall_distance": 3,
    "distractor_density": 0.0,
    "horizon": 8,
    "seed": 1000,
}


def _paths(tmp_path: Path) -> RunPaths:
    return RunPaths(
        results=tmp_path / "runs.db",
        traces=tmp_path / "traces",
        sandbox=tmp_path / "sandbox",
    )


def _first_trace_event(trace_dir: Path, run_id: str) -> Any:
    path = trace_dir / f"{run_id.replace(':', '_')}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _reading_script(task_index: int = 0) -> list[str]:
    """Replies that walk the workspace and then answer: a competent agent, scripted."""
    spec = build_suite(SuiteSpec(**SUITE))[task_index]  # type: ignore[arg-type]
    task = generate(spec)
    reads = [
        json.dumps({"tool_name": "read_file", "arguments": {"path": name}})
        for name, _ in task.documents
    ]
    return [*reads, json.dumps({"tool_name": "answer", "arguments": {"text": task.answer}})]


def test_a_run_records_one_row_per_task(tmp_path: Path) -> None:
    # Arrange
    config = load_config(write_config(tmp_path, suite=SUITE, run={"max_steps": 12}))
    paths = _paths(tmp_path)

    # Act
    summary = run_config(config, paths, llm=FakeLLMClient(["{}"]))

    # Assert
    assert len(summary.executed) == 2
    with ResultsStore(paths.results) as store:
        assert store.count() == 2


def test_a_row_carries_the_provenance_its_trace_carries(tmp_path: Path) -> None:
    """A row and its trace can never disagree about which run produced them."""
    # Arrange
    config = load_config(write_config(tmp_path, suite=SUITE, run={"max_steps": 12}))
    paths = _paths(tmp_path)

    # Act
    summary = run_config(config, paths, llm=FakeLLMClient(["{}"]))

    # Assert
    with ResultsStore(paths.results) as store:
        row = store.get(summary.executed[0])
    header = _first_trace_event(paths.traces, summary.executed[0])
    assert row is not None
    assert header["kind"] == "header"
    assert header["payload"]["config_hash"] == row.config_hash
    assert header["payload"]["run_id"] == row.run_id
    assert header["payload"]["git_commit"] == row.git_commit


def test_a_scripted_agent_that_reads_the_documents_succeeds(tmp_path: Path) -> None:
    """The bracket instrument end to end: oracle memory, a reader, a correct answer."""
    # Arrange
    config = load_config(
        write_config(
            tmp_path,
            suite={**SUITE, "count": 1},
            memory={"type": "oracle"},
            run={"max_steps": 12},
            planner={"max_steps": 12},
        )
    )
    paths = _paths(tmp_path)

    # Act
    summary = run_config(config, paths, llm=FakeLLMClient(_reading_script()))

    # Assert
    with ResultsStore(paths.results) as store:
        row = store.get(summary.executed[0])
    assert row is not None
    assert row.success is True


def test_a_second_invocation_skips_what_is_already_recorded(tmp_path: Path) -> None:
    """Resume is the default, not a flag. Four hours of runs is not a thing to lose."""
    # Arrange
    config = load_config(write_config(tmp_path, suite=SUITE, run={"max_steps": 12}))
    paths = _paths(tmp_path)
    run_config(config, paths, llm=FakeLLMClient(["{}"]))

    # Act
    second = run_config(config, paths, llm=FakeLLMClient(["{}"]))

    # Assert
    assert second.executed == []
    assert len(second.skipped) == 2


def test_the_condition_column_is_the_memory_type(tmp_path: Path) -> None:
    # Arrange
    config = load_config(
        write_config(
            tmp_path, suite={**SUITE, "count": 1}, memory={"type": "episodic"}, run={"max_steps": 6}
        )
    )
    paths = _paths(tmp_path)

    # Act
    summary = run_config(config, paths, llm=FakeLLMClient(["{}"]))

    # Assert
    with ResultsStore(paths.results) as store:
        row = store.get(summary.executed[0])
    assert row is not None
    assert row.condition == "episodic"


def test_a_failing_cell_is_reported_and_left_unrecorded(tmp_path: Path) -> None:
    """Unrecorded on purpose: a rerun retries it rather than inheriting a bad row."""
    # Arrange
    config = load_config(
        write_config(tmp_path, suite={**SUITE, "recall_distance": 99}, run={"max_steps": 6})
    )
    paths = _paths(tmp_path)

    # Act
    summary = run_config(config, paths, llm=FakeLLMClient(["{}"]))

    # Assert
    assert summary.executed == []
    assert len(summary.failed) == 2
    with ResultsStore(paths.results) as store:
        assert store.count() == 0


def test_main_runs_a_config_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    config_path = write_config(tmp_path, suite={**SUITE, "count": 1}, run={"max_steps": 6})
    monkeypatch.setattr("harness.cli.build_llm", lambda config, env=None: FakeLLMClient(["{}"]))
    paths = _paths(tmp_path)

    # Act
    code = main(
        [
            "run",
            str(config_path),
            "--out",
            str(paths.results),
            "--traces",
            str(paths.traces),
            "--sandbox",
            str(paths.sandbox),
        ]
    )

    # Assert
    assert code == 0
    assert "executed=1" in capsys.readouterr().out


def test_main_exits_non_zero_when_a_cell_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    config_path = write_config(
        tmp_path, suite={**SUITE, "count": 1, "recall_distance": 99}, run={"max_steps": 6}
    )
    monkeypatch.setattr("harness.cli.build_llm", lambda config, env=None: FakeLLMClient(["{}"]))
    paths = _paths(tmp_path)

    # Act
    code = main(
        [
            "run",
            str(config_path),
            "--out",
            str(paths.results),
            "--traces",
            str(paths.traces),
            "--sandbox",
            str(paths.sandbox),
        ]
    )

    # Assert
    assert code == 1
    assert "FAILED" in capsys.readouterr().err


def test_a_run_id_with_colons_becomes_a_usable_trace_filename(tmp_path: Path) -> None:
    """Colons are a path separator on some filesystems, and run ids are full of them."""
    # Arrange
    config = load_config(write_config(tmp_path, suite={**SUITE, "count": 1}, run={"max_steps": 6}))
    paths = _paths(tmp_path)

    # Act
    run_config(config, paths, llm=FakeLLMClient(["{}"]))

    # Assert
    written = list(paths.traces.glob("*.jsonl"))
    assert len(written) == 1
    assert ":" not in written[0].name


def test_a_second_run_seed_is_a_new_replicate_not_a_duplicate(tmp_path: Path) -> None:
    """Replicates differ only in `run.seed`. Keying rows on the task seed skipped them."""
    # Arrange
    first = load_config(write_config(tmp_path, suite=SUITE))
    second = first.model_copy(update={"run": first.run.model_copy(update={"seed": 2})})
    paths = _paths(tmp_path)

    # Act
    run_config(first, paths, llm=FakeLLMClient(_reading_script()))
    summary = run_config(second, paths, llm=FakeLLMClient(_reading_script()))

    # Assert
    assert len(summary.executed) == SUITE["count"]
    with ResultsStore(paths.results) as store:
        assert store.count() == 2 * SUITE["count"]


def test_sweep_runs_every_config_under_every_seed(tmp_path: Path) -> None:
    # Arrange
    noop = load_config(write_config(tmp_path, suite=SUITE, memory={"type": "noop"}))
    oracle = noop.model_copy(update={"memory": noop.memory.model_copy(update={"type": "oracle"})})
    paths = _paths(tmp_path)

    # Act
    code = run_sweep([noop, oracle], paths, seeds=[1, 2], llm=FakeLLMClient(_reading_script()))

    # Assert
    assert code == 0
    with ResultsStore(paths.results) as store:
        rows = store.all()
    assert len(rows) == 2 * 2 * SUITE["count"]
    assert {(row.condition, row.seed) for row in rows} == {
        (c, s) for c in ("noop", "oracle") for s in (1, 2)
    }


def test_main_sweep_and_report_produce_a_pilot_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    config_path = write_config(tmp_path, suite=SUITE)
    monkeypatch.setattr("harness.cli.build_llm", lambda _: FakeLLMClient(_reading_script()))
    db = tmp_path / "runs.db"
    common = ["--out", str(db), "--traces", str(tmp_path / "t"), "--sandbox", str(tmp_path)]

    # Act
    swept = main(["sweep", str(config_path), "--seeds", "1", "2", *common])
    reported = main(["report", "--db", str(db), "--out", str(tmp_path / "pilot.md")])
    printed = main(["report", "--db", str(db), "--budget", "4000"])

    # Assert
    assert (swept, reported, printed) == (0, 0, 0)
    assert "Preliminary" in (tmp_path / "pilot.md").read_text()
    assert "| noop | 4 |" in capsys.readouterr().out


def test_report_on_an_empty_database_fails_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act
    code = main(["report", "--db", str(tmp_path / "empty.db")])

    # Assert
    assert code == 1
    assert "report:" in capsys.readouterr().err
