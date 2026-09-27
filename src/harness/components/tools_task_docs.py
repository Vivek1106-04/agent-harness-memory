"""TaskDocumentRuntime - serves a generated task's documents as read-only files.

The task environment for family `multihop_qa`: a workspace of documents, one fact each,
most of them distractors. Documents live in memory rather than on disk - nothing here
writes a file - but every path is validated through the same confinement helper the real
runtime uses. A mock more permissive than what ships hides the bug it exists to catch.
See SECURITY.md section 5.

Document content is model-facing untrusted text. It is returned as an observation and never
interpreted, and the system instructions tell the agent the same thing.
"""

from __future__ import annotations

from pathlib import Path

from harness.interfaces import ToolRuntime
from harness.prompts import TARGET_KEY
from harness.registry import register
from harness.sandbox import SandboxViolationError, resolve_within
from harness.tasks.generator import ANSWER_TOOL, generate
from harness.types import Action, Observation, TaskSpec

MAX_OUTPUT_CHARS = 4000

READ_TOOL = "read_file"
LIST_TOOL = "list_dir"
PATH_TOOLS = frozenset({READ_TOOL})
TEXT_TOOLS = frozenset({ANSWER_TOOL})


@register("tools", name="task_docs")
class TaskDocumentRuntime(ToolRuntime):
    def __init__(self, sandbox_root: Path, max_output_chars: int = MAX_OUTPUT_CHARS) -> None:
        self._sandbox_root = Path(sandbox_root)
        self._max_output_chars = max_output_chars
        self._documents: dict[str, str] = {}

    def prepare(self, task: TaskSpec) -> None:
        """Re-derive the task's workspace. Generation is pure, so the runtime, the oracle
        and the evaluator all see the identical document set."""
        self._documents = dict(generate(task).documents)

    def execute(self, action: Action, step: int) -> Observation:
        if action.tool_name in PATH_TOOLS:
            return self._read(action, step)
        if action.tool_name == LIST_TOOL:
            return self._ok(step, LIST_TOOL, "\n".join(self._documents))
        if action.tool_name in TEXT_TOOLS:
            return self._answer(action, step)
        return self._error(step, action.tool_name, f"unknown tool {action.tool_name!r}")

    def _read(self, action: Action, step: int) -> Observation:
        path = action.arguments.get("path")
        if not isinstance(path, str) or not path:
            return self._error(step, action.tool_name, "missing required argument 'path'")
        try:
            resolved = resolve_within(self._sandbox_root, path)
        except SandboxViolationError as violation:
            return self._error(step, action.tool_name, str(violation))
        content = self._documents.get(resolved.name)
        if content is None:
            return self._error(step, action.tool_name, f"no such document {resolved.name!r}")
        return self._ok(step, READ_TOOL, content, target=path)

    def _answer(self, action: Action, step: int) -> Observation:
        text = action.arguments.get("text")
        if not isinstance(text, str):
            return self._error(step, action.tool_name, "missing required argument 'text'")
        return self._ok(step, ANSWER_TOOL, text)

    def _ok(
        self, step: int, tool_name: str, content: str, target: str | None = None
    ) -> Observation:
        return Observation(
            step=step,
            content=content[: self._max_output_chars],
            tool_name=tool_name,
            metadata={TARGET_KEY: target} if target is not None else {},
        )

    @staticmethod
    def _error(step: int, tool_name: str | None, message: str) -> Observation:
        return Observation(step=step, content="", tool_name=tool_name, error=message)
