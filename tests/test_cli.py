"""One task, start to finish, reproducible from a config hash - with no model endpoint.

The end-to-end deliverable, driven by a scripted client. What it proves is that config,
registry, loop, trace, provenance and results agree with one another; what a real model
adds is the only thing left, and that is what the gate run is for.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from harness.cli import (
    EXIT_ABORTED,
    RunPaths,
    _set_aside_failed_attempt,
    main,
    run_config,
    run_sweep,
    suite_spec,
)
from harness.config import HarnessConfig, load_config
from harness.llm_http import SpendLimitError
from harness.results import ResultsStore
from harness.tasks.generator import generate
from harness.tasks.suite import build_suite
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


def _config_for_script() -> HarnessConfig:
    """The suite the scripts are written against: SUITE on the shared template."""
    with tempfile.TemporaryDirectory() as scratch:
        return load_config(write_config(Path(scratch), suite=SUITE))


def _reading_script(task_index: int = 0) -> list[str]:
    """Replies that walk the workspace and then answer: a competent agent, scripted."""
    spec = build_suite(suite_spec(_config_for_script()))[task_index]
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
    monkeypatch.setattr("harness.cli.build_llm", lambda config, **_: FakeLLMClient(["{}"]))
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
    monkeypatch.setattr("harness.cli.build_llm", lambda config, **_: FakeLLMClient(["{}"]))
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


def test_a_retried_run_sets_its_failed_attempt_aside(tmp_path: Path) -> None:
    """A run that failed mid-way left a partial trace. The retry's trace must be the run the
    row describes, and the failed attempt is kept beside it rather than lost or mixed in."""
    # Arrange
    config = load_config(write_config(tmp_path, suite={**SUITE, "count": 1}, run={"max_steps": 6}))
    paths = _paths(tmp_path)
    paths.traces.mkdir(parents=True)
    run_config(config, paths, llm=FakeLLMClient(["{}"]))
    (trace,) = paths.traces.glob("*.jsonl")
    partial = trace.read_text().splitlines()[0] + "\n"
    trace.write_text(partial)
    paths.results.unlink()

    # Act
    run_config(config, paths, llm=FakeLLMClient(["{}"]))

    # Assert
    headers = [line for line in trace.read_text().splitlines() if '"kind": "header"' in line]
    assert len(headers) == 1
    assert trace.with_name(trace.stem + ".failed-1.jsonl").read_text() == partial


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
    monkeypatch.setattr("harness.cli.build_llm", lambda _, **__: FakeLLMClient(_reading_script()))
    db = tmp_path / "runs.db"
    common = ["--out", str(db), "--traces", str(tmp_path / "t"), "--sandbox", str(tmp_path)]

    # Act
    swept = main(["sweep", str(config_path), "--seeds", "1", "2", *common])
    reported = main(["report", "--db", str(db), "--out", str(tmp_path / "pilot.md")])
    printed = main(["report", "--db", str(db), "--budget", "4000"])

    # Assert
    assert (swept, reported, printed) == (0, 0, 0)
    assert "Preliminary" in (tmp_path / "pilot.md").read_text()
    assert "| noop |" in capsys.readouterr().out


def test_report_on_an_empty_database_fails_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act
    code = main(["report", "--db", str(tmp_path / "empty.db")])

    # Assert
    assert code == 1
    assert "report:" in capsys.readouterr().err


def test_a_spend_abort_stops_the_whole_sweep_with_its_own_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    first = load_config(write_config(tmp_path, suite=SUITE))
    second = first.model_copy(update={"memory": first.memory.model_copy(update={"type": "oracle"})})

    class Broke(FakeLLMClient):
        def complete(self, window: object, max_output_tokens: int) -> str:
            raise SpendLimitError("spend limit reached: $25.00 of $25.00")

    # Act
    code = run_sweep([first, second], _paths(tmp_path), llm=Broke(["x"]))

    # Assert
    assert code == EXIT_ABORTED
    assert "sweep stopped: spend limit reached" in capsys.readouterr().err


def test_one_metered_client_per_seed_serves_every_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed is sent with every request, so a replicate needs its own client."""
    # Arrange
    built: list[int] = []

    def fake_build(config: HarnessConfig, already_spent_usd: float = 0.0) -> FakeLLMClient:
        built.append(config.run.seed)
        return FakeLLMClient(["{}"])

    monkeypatch.setattr("harness.cli.build_llm", fake_build)
    first = load_config(write_config(tmp_path, suite=SUITE))
    second = load_config(write_config(tmp_path, suite=SUITE, run={"run_id_prefix": "b"}))

    # Act
    run_sweep([first, second], _paths(tmp_path), seeds=[1, 2])

    # Assert
    assert built == [1, 2]


def test_a_later_seed_starts_its_meter_at_what_the_store_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Separate clients would each start the spend meter at zero."""

    # Arrange
    class Metered(FakeLLMClient):
        @property
        def spent_usd(self) -> float:
            return 0.01 * len(self.calls)

    built: list[float] = []

    def fake_build(config: HarnessConfig, already_spent_usd: float = 0.0) -> FakeLLMClient:
        built.append(already_spent_usd)
        return Metered(_reading_script())

    monkeypatch.setattr("harness.cli.build_llm", fake_build)
    config = load_config(write_config(tmp_path, suite=SUITE))

    # Act
    run_sweep([config], _paths(tmp_path), seeds=[1, 2])

    # Assert
    assert built[0] == 0.0
    assert built[1] > 0.0


def test_a_row_records_what_its_run_spent(tmp_path: Path) -> None:
    # Arrange
    class Metered(FakeLLMClient):
        @property
        def spent_usd(self) -> float:
            return 0.01 * len(self.calls)

    config = load_config(write_config(tmp_path, suite=SUITE))
    paths = _paths(tmp_path)

    # Act
    run_config(config, paths, llm=Metered(_reading_script()))

    # Assert
    with ResultsStore(paths.results) as store:
        rows = store.all()
    assert all(row.cost_usd > 0 for row in rows)
    assert store_total(paths) == pytest.approx(sum(row.cost_usd for row in rows))


def store_total(paths: RunPaths) -> float:
    with ResultsStore(paths.results) as store:
        return store.total_cost_usd()


def test_task_ids_carry_the_suite_so_two_suites_never_share_a_task(tmp_path: Path) -> None:
    # Arrange
    base = load_config(write_config(tmp_path, suite=SUITE))
    other = base.model_copy(
        update={"suite": base.suite.model_copy(update={"document_tokens": 900})}
    )

    # Act
    first = build_suite(suite_spec(base))[0].task_id
    again = build_suite(suite_spec(base))[0].task_id
    second = build_suite(suite_spec(other))[0].task_id

    # Assert
    assert first == again
    assert first != second
    assert first.startswith("suite")


def test_suite_size_does_not_change_task_identity(tmp_path: Path) -> None:
    base = load_config(write_config(tmp_path, suite=SUITE))
    bigger = base.model_copy(update={"suite": base.suite.model_copy(update={"count": 9})})
    assert build_suite(suite_spec(base))[0] == build_suite(suite_spec(bigger))[0]


def test_the_failure_streak_spans_every_config_of_a_sweep(tmp_path: Path) -> None:
    """One-task configs each failing once must still stop the sweep at the limit."""
    # Arrange
    base = load_config(write_config(tmp_path, suite={**SUITE, "count": 1}))
    configs = [
        base.model_copy(update={"memory": base.memory.model_copy(update={"type": kind})})
        for kind in ("noop", "oracle", "episodic", "noop")
    ]

    class Dead(FakeLLMClient):
        def complete(self, window: object, max_output_tokens: int) -> str:
            raise RuntimeError("401")

    # Act
    code = run_sweep(configs, _paths(tmp_path), llm=Dead(["x"]))

    # Assert
    assert code == EXIT_ABORTED


def test_a_second_failed_attempt_does_not_overwrite_the_first(tmp_path: Path) -> None:
    # Arrange
    trace = tmp_path / "run.jsonl"
    trace.write_text("first\n")
    _set_aside_failed_attempt(trace)
    trace.write_text("second\n")

    # Act
    _set_aside_failed_attempt(trace)

    # Assert
    assert (tmp_path / "run.failed-1.jsonl").read_text() == "first\n"
    assert (tmp_path / "run.failed-2.jsonl").read_text() == "second\n"
    assert not trace.exists()
