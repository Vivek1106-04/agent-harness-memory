from __future__ import annotations

import json
from pathlib import Path

from harness.redaction import PLACEHOLDER
from harness.trace import HEADER_KIND, JsonlTraceSink, MemoryTraceSink, TraceBus
from harness.types import PAYLOAD_VERSION, RunProvenance


def _provenance() -> RunProvenance:
    return RunProvenance(
        run_id="run-1",
        config_hash="9f2b1c4e",
        git_commit="abc123",
        payload_version=PAYLOAD_VERSION,
        model_id="qwen3-8b",
        model_revision="def456",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_revision="ghi789",
        seeds={"run": 1},
    )


def test_emitted_event_carries_the_run_id_and_fields() -> None:
    # Arrange
    sink = MemoryTraceSink()
    bus = TraceBus("run-1", sink)

    # Act
    bus.emit("memory", "retrieval", step=3, payload={"k": 5}, token_counts={"read": 120})

    # Assert
    event = sink.events[0]
    assert event.run_id == "run-1"
    assert (event.layer, event.kind, event.step) == ("memory", "retrieval", 3)
    assert event.token_counts == {"read": 120}


def test_credentials_are_redacted_before_the_sink_sees_them() -> None:
    """Redaction happens at emit time, not read time. SECURITY.md section 1."""
    # Arrange
    sink = MemoryTraceSink()
    bus = TraceBus("run-1", sink)

    # Act
    bus.emit("planner", "action", step=0, payload={"prompt": "sk-abcdefghijklmnopqrstuvwxyz0123"})

    # Assert
    assert sink.events[0].payload["prompt"] == PLACEHOLDER


def test_header_records_full_provenance() -> None:
    # Arrange
    sink = MemoryTraceSink()
    bus = TraceBus("run-1", sink)

    # Act
    bus.emit_header(_provenance())

    # Assert
    event = sink.events[0]
    assert event.kind == HEADER_KIND
    assert event.payload["config_hash"] == "9f2b1c4e"
    assert event.payload["payload_version"] == PAYLOAD_VERSION


def test_jsonl_sink_appends_one_json_object_per_line(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "traces" / "run-1.jsonl"

    # Act
    with JsonlTraceSink(path) as sink:
        bus = TraceBus("run-1", sink)
        bus.emit_header(_provenance())
        bus.emit("tools", "observation", step=1, payload={"content": "ok"})

    # Assert
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["payload"]["content"] == "ok"


def test_jsonl_sink_appends_rather_than_truncating(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "run-1.jsonl"
    with JsonlTraceSink(path) as sink:
        TraceBus("run-1", sink).emit("runtime", "error", step=0, payload={"m": "first"})

    # Act
    with JsonlTraceSink(path) as sink:
        TraceBus("run-1", sink).emit("runtime", "error", step=1, payload={"m": "second"})

    # Assert
    assert len(path.read_text().splitlines()) == 2
