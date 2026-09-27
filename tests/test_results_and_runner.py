from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from harness.results import ResultsStore, RunResult
from harness.runner import Runner, RunPlanItem, SweepAbortedError, build_run_id


def _result(run_id: str = "exp1:vector:8000:t1:s1", score: float = 1.0) -> RunResult:
    return RunResult(
        run_id=run_id,
        task_id="t1",
        condition="vector",
        token_budget=8000,
        seed=1,
        success=True,
        score=score,
        tokens_read=1200,
        tokens_write=300,
        steps=14,
        wall_ms=2500.0,
        config_hash="9f2b1c4e",
        git_commit="abc123",
        payload_version="1.0",
        model_id="qwen3-8b",
        model_revision="def456",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_revision="ghi789",
    )


def _plan(n: int = 3) -> list[RunPlanItem]:
    return [
        RunPlanItem(
            run_id=build_run_id("exp1", "vector", f"t{i}", seed=1, token_budget=8000),
            task_id=f"t{i}",
            condition="vector",
            token_budget=8000,
            seed=1,
        )
        for i in range(n)
    ]


def test_recorded_result_round_trips(tmp_path: Path) -> None:
    # Arrange
    store = ResultsStore(tmp_path / "results.db")

    # Act
    inserted = store.record(_result())

    # Assert
    assert inserted is True
    assert store.get(_result().run_id) == _result()
    store.close()


def test_recording_the_same_run_twice_writes_one_row(tmp_path: Path) -> None:
    """A duplicated row silently corrupts every downstream statistic."""
    # Arrange
    store = ResultsStore(tmp_path / "results.db")

    # Act
    store.record(_result())
    second = store.record(_result(score=0.0))

    # Assert
    assert second is False
    assert store.count() == 1
    assert store.get(_result().run_id).score == 1.0
    store.close()


def test_results_survive_reopening_the_database(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "results.db"
    with ResultsStore(path) as store:
        store.record(_result())

    # Act
    with ResultsStore(path) as reopened:
        recovered = reopened.get(_result().run_id)

    # Assert
    assert recovered is not None
    assert recovered.success is True


def test_missing_run_returns_none(tmp_path: Path) -> None:
    with ResultsStore(tmp_path / "results.db") as store:
        assert store.get("nope") is None
        assert store.has("nope") is False


def test_runner_executes_every_pending_cell(tmp_path: Path) -> None:
    # Arrange
    store = ResultsStore(tmp_path / "results.db")
    runner = Runner(store)
    plan = _plan()

    # Act
    summary = runner.run_all(plan, lambda item: _result(item.run_id))

    # Assert
    assert len(summary.executed) == 3
    assert store.count() == 3
    store.close()


def test_restart_after_a_crash_skips_completed_runs(tmp_path: Path) -> None:
    """Kill it mid-sweep, restart, no duplicate rows and no repeated work."""
    # Arrange
    path = tmp_path / "results.db"
    plan = _plan()
    executed_first: list[str] = []

    def crashing(item: RunPlanItem) -> RunResult:
        if len(executed_first) == 2:
            raise KeyboardInterrupt("simulated kill")
        executed_first.append(item.run_id)
        return _result(item.run_id)

    store = ResultsStore(path)
    with contextlib.suppress(KeyboardInterrupt):
        Runner(store).run_all(plan, crashing)
    store.close()

    # Act
    reopened = ResultsStore(path)
    executed_second: list[str] = []

    def normal(item: RunPlanItem) -> RunResult:
        executed_second.append(item.run_id)
        return _result(item.run_id)

    summary = Runner(reopened).run_all(plan, normal)

    # Assert
    assert executed_second == [plan[2].run_id]
    assert summary.skipped == [plan[0].run_id, plan[1].run_id]
    assert reopened.count() == 3
    reopened.close()


def test_pending_lists_only_unrecorded_cells(tmp_path: Path) -> None:
    # Arrange
    store = ResultsStore(tmp_path / "results.db")
    plan = _plan()
    store.record(_result(plan[0].run_id))

    # Act
    pending = Runner(store).pending(plan)

    # Assert
    assert [item.run_id for item in pending] == [plan[1].run_id, plan[2].run_id]
    store.close()


def test_a_failing_cell_is_reported_and_left_for_a_retry(tmp_path: Path) -> None:
    # Arrange
    store = ResultsStore(tmp_path / "results.db")
    plan = _plan(2)

    def flaky(item: RunPlanItem) -> RunResult:
        if item.run_id == plan[0].run_id:
            raise RuntimeError("endpoint refused the connection")
        return _result(item.run_id)

    # Act
    summary = Runner(store).run_all(plan, flaky)

    # Assert
    assert summary.executed == [plan[1].run_id]
    assert summary.failed[0][0] == plan[0].run_id
    assert "endpoint refused" in summary.failed[0][1]
    assert store.has(plan[0].run_id) is False
    store.close()


def test_a_result_for_the_wrong_run_is_refused(tmp_path: Path) -> None:
    """An execute function that returns another cell's result would mislabel the sweep."""
    # Arrange
    store = ResultsStore(tmp_path / "results.db")
    plan = _plan(1)

    # Act
    summary = Runner(store).run_all(plan, lambda item: _result("some-other-run"))

    # Assert
    assert summary.executed == []
    assert summary.failed[0][0] == plan[0].run_id
    assert store.count() == 0
    store.close()


def test_run_id_is_deterministic_for_a_cell() -> None:
    first = build_run_id("exp1", "graph", "t7", seed=2, token_budget=4000)
    second = build_run_id("exp1", "graph", "t7", seed=2, token_budget=4000)
    assert first == second
    assert first != build_run_id("exp1", "graph", "t7", seed=3, token_budget=4000)


def test_an_abort_stops_the_sweep_at_that_cell(tmp_path: Path) -> None:
    # Arrange
    attempted: list[str] = []

    def execute(item: RunPlanItem) -> RunResult:
        attempted.append(item.run_id)
        raise SweepAbortedError("spend limit reached")

    # Act
    with ResultsStore(tmp_path / "r.db") as store:
        summary = Runner(store).run_all(_plan(3), execute)

    # Assert
    assert len(attempted) == 1
    assert summary.aborted == ["spend limit reached"]


@pytest.mark.parametrize("limit", [2, 3])
def test_consecutive_failures_stop_the_sweep(tmp_path: Path, limit: int) -> None:
    """A dead key or a spent daily quota fails every cell; stop rather than grind on."""
    # Arrange
    attempted: list[str] = []

    def execute(item: RunPlanItem) -> RunResult:
        attempted.append(item.run_id)
        raise RuntimeError("429")

    # Act
    with ResultsStore(tmp_path / "r.db") as store:
        summary = Runner(store, max_consecutive_failures=limit).run_all(_plan(5), execute)

    # Assert
    assert len(attempted) == limit
    assert summary.aborted == [f"{limit} consecutive cells failed"]


def test_a_success_resets_the_failure_streak(tmp_path: Path) -> None:
    # Arrange
    plan = _plan(4)
    outcomes = iter([RuntimeError("x"), None, RuntimeError("y"), None])

    def execute(item: RunPlanItem) -> RunResult:
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome
        return _result(item.run_id)

    # Act
    with ResultsStore(tmp_path / "r.db") as store:
        summary = Runner(store, max_consecutive_failures=2).run_all(plan, execute)

    # Assert
    assert summary.aborted == []
    assert len(summary.executed) == 2


def test_total_cost_sums_recorded_runs(tmp_path: Path) -> None:
    from dataclasses import replace

    with ResultsStore(tmp_path / "r.db") as store:
        assert store.total_cost_usd() == 0.0
        store.record(replace(_result("a"), cost_usd=0.25))
        store.record(replace(_result("b"), cost_usd=0.5))
        assert store.total_cost_usd() == 0.75
