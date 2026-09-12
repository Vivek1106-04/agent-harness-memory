"""MockToolRuntime - deterministic canned tools, no filesystem, no network.

The trivial second implementation of the tool layer and what every unit test uses. It
touches no filesystem, but it validates paths through exactly the same confinement helper
as the real runtime: a mock that is more permissive than what ships hides the bug it exists
to catch. See SECURITY.md section 5.
"""

from __future__ import annotations

from pathlib import Path

from harness.interfaces import ToolRuntime
from harness.registry import register
from harness.sandbox import SandboxViolationError, resolve_within
from harness.types import Action, Observation

CANNED_FILE_CONTENT = "mock file content"
MAX_OUTPUT_CHARS = 4000

#: Tool allowlist. Anything outside it is an error observation, never an execution.
PATH_TOOLS = frozenset({"read_file", "write_file", "list_dir"})
TEXT_TOOLS = frozenset({"echo", "answer"})


@register("tools", name="mock")
class MockToolRuntime(ToolRuntime):
    def __init__(self, sandbox_root: Path, max_output_chars: int = MAX_OUTPUT_CHARS) -> None:
        self._sandbox_root = Path(sandbox_root)
        self._max_output_chars = max_output_chars

    def execute(self, action: Action, step: int) -> Observation:
        if action.tool_name in PATH_TOOLS:
            return self._execute_path_tool(action, step)
        if action.tool_name in TEXT_TOOLS:
            return self._execute_text_tool(action, step)
        return self._error(step, action.tool_name, f"unknown tool {action.tool_name!r}")

    def _execute_path_tool(self, action: Action, step: int) -> Observation:
        path = action.arguments.get("path")
        if not isinstance(path, str) or not path:
            return self._error(step, action.tool_name, "missing required argument 'path'")
        try:
            resolve_within(self._sandbox_root, path)
        except SandboxViolationError as violation:
            return self._error(step, action.tool_name, str(violation))
        return Observation(
            step=step,
            content=self._truncate(f"{action.tool_name}({path}): {CANNED_FILE_CONTENT}"),
            tool_name=action.tool_name,
        )

    def _execute_text_tool(self, action: Action, step: int) -> Observation:
        text = action.arguments.get("text")
        if not isinstance(text, str):
            return self._error(step, action.tool_name, "missing required argument 'text'")
        return Observation(step=step, content=self._truncate(text), tool_name=action.tool_name)

    def _truncate(self, content: str) -> str:
        return content[: self._max_output_chars]

    @staticmethod
    def _error(step: int, tool_name: str | None, message: str) -> Observation:
        return Observation(step=step, content="", tool_name=tool_name, error=message)
