"""The trace bus. Every layer emits through it and no layer logs any other way.

One event type, one stream, one redaction point. The evaluator and the results store read
only from this stream, which is what makes a run reconstructible from its trace alone.

Traces are JSONL, one file per run: append-only, greppable, and streamable without loading
the whole file. Never pickle. See SECURITY.md sections 1 and 6.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from harness.redaction import redact_payload
from harness.types import RunProvenance, TraceEvent

HEADER_KIND = "header"
RUNTIME_LAYER = "runtime"


class TraceSink(Protocol):
    def write(self, event: TraceEvent) -> None: ...


class MemoryTraceSink:
    """Collects events in memory. What unit tests assert against."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def write(self, event: TraceEvent) -> None:
        self.events.append(event)


class JsonlTraceSink:
    """One JSONL file per run, opened for append."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    def write(self, event: TraceEvent) -> None:
        self._handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlTraceSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class TraceBus:
    """Emits redacted `TraceEvent`s for one run.

    Redaction runs here, at emit time, so there is exactly one place where a credential
    could reach disk and exactly one place that prevents it.
    """

    def __init__(self, run_id: str, sink: TraceSink) -> None:
        self._run_id = run_id
        self._sink = sink

    @property
    def run_id(self) -> str:
        """The run this bus stamps. Layers use it to build provenance strings."""
        return self._run_id

    def emit(
        self,
        layer: str,
        kind: str,
        step: int,
        payload: dict[str, Any],
        token_counts: dict[str, int] | None = None,
        wall_ms: float = 0.0,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self._run_id,
            step=step,
            layer=layer,
            kind=kind,
            payload=redact_payload(payload),
            token_counts=dict(token_counts or {}),
            wall_ms=wall_ms,
        )
        self._sink.write(event)
        return event

    def emit_header(self, provenance: RunProvenance) -> TraceEvent:
        """A result whose provenance cannot be reconstructed is not a result."""
        return self.emit(RUNTIME_LAYER, HEADER_KIND, step=0, payload=asdict(provenance))
