"""Print what one run did, for the live swap demo: the verdict, the answer given, and which
documents the memory handed back on the final step - the retrieval the answer was made on.

    uv run python scripts/demo_summary.py <trace.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def summarise(events: list[dict[str, Any]]) -> list[str]:
    acted = {
        event["step"]: event["payload"]["arguments"].get("path") or event["payload"]["tool_name"]
        for event in events
        if event["kind"] == "action"
    }
    answers = [
        event["payload"]["arguments"].get("text")
        for event in events
        if event["kind"] == "action" and event["payload"]["tool_name"] == "answer"
    ]
    retrievals = [event for event in events if event["kind"] == "retrieval"]
    verdict = next(event["payload"] for event in events if event["kind"] == "verdict")
    header = next(event["payload"] for event in events if event["kind"] == "header")

    last = retrievals[-1]["payload"]["returned"] if retrievals else []
    steps = sorted({int(provenance.rsplit(":", 1)[1]) for provenance in last})
    returned = [str(acted.get(step, f"step {step}")) for step in steps]
    return [
        f"  memory      : {header['run_id'].split(':')[1]}",
        f"  steps       : {events[-1]['step']}",
        f"  answer      : {answers[-1] if answers else '(none)'}",
        f"  verdict     : {'SUCCESS' if verdict['success'] else 'FAIL'} - {verdict['reason']}",
        f"  memory gave : {', '.join(returned) if returned else '(nothing)'}",
    ]


def main() -> int:
    lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    print("\n".join(summarise([json.loads(line) for line in lines])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
