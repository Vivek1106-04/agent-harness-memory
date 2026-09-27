"""LocalToolRuntime - the real filesystem, confined to a per-run sandbox.

`TaskDocumentRuntime` serves documents from a dict and never touches disk, which is right
for speed but leaves the confinement code untested against a real filesystem: a symlink
cannot escape a dictionary. This runtime writes the task's documents into the sandbox root
and reads them back through `resolve_within`, so the path checks in SECURITY.md section 5
are exercised against the thing they defend.

Both runtimes refuse exactly the same paths. That is the property under test, not an
implementation detail - a mock more permissive than what ships hides the bug it exists to
catch.

Two hazards a path check alone does not cover, and what is done about each:

* **Reads that never return.** A FIFO under the sandbox root blocks forever and no timeout
  on the loop can interrupt a blocking read. So only regular files are opened; anything
  else is refused before the open.
* **Reads that never end.** A large file would exhaust the context budget and the wallet.
  Reads stop at `max_read_bytes`, ahead of the character cap on the observation.
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
#: Read cap, applied before the character cap. Documents are one sentence; anything near
#: this is a file that does not belong in a task workspace.
MAX_READ_BYTES = 64 * 1024

READ_TOOL = "read_file"
LIST_TOOL = "list_dir"

ERROR_NOT_A_FILE = "path {name!r} is not a regular file"
ERROR_UNREADABLE = "document {name!r} could not be read"


@register("tools", name="local")
class LocalToolRuntime(ToolRuntime):
    def __init__(
        self,
        sandbox_root: Path,
        max_output_chars: int = MAX_OUTPUT_CHARS,
        max_read_bytes: int = MAX_READ_BYTES,
    ) -> None:
        self._sandbox_root = Path(sandbox_root)
        self._max_output_chars = max_output_chars
        self._max_read_bytes = max_read_bytes
        #: Per-task subdirectory of the root, set by `prepare`. Held separately so that a
        #: second `prepare` re-roots from the sandbox rather than nesting inside the first.
        self._workspace = self._sandbox_root
        self._document_names: tuple[str, ...] = ()

    def prepare(self, task: TaskSpec) -> None:
        """Materialise the task's workspace on disk under the sandbox root.

        Generation is pure, so the runtime, the oracle and the evaluator all see the same
        documents. Files are rewritten every run rather than reused: a document left over
        from a previous task is cross-run contamination that would look like a real effect.
        """
        generated = generate(task)
        workspace = self._sandbox_root / task.task_id
        workspace.mkdir(parents=True, exist_ok=True)
        for name, content in generated.documents:
            (workspace / name).write_text(content, encoding="utf-8")
        self._workspace = workspace
        self._document_names = tuple(name for name, _ in generated.documents)

    def execute(self, action: Action, step: int) -> Observation:
        if action.tool_name == READ_TOOL:
            return self._read(action, step)
        if action.tool_name == LIST_TOOL:
            return self._ok(step, LIST_TOOL, "\n".join(self._document_names))
        if action.tool_name == ANSWER_TOOL:
            return self._answer(action, step)
        return self._error(step, action.tool_name, f"unknown tool {action.tool_name!r}")

    def _read(self, action: Action, step: int) -> Observation:
        path = action.arguments.get("path")
        if not isinstance(path, str) or not path:
            return self._error(step, action.tool_name, "missing required argument 'path'")
        try:
            resolved = resolve_within(self._workspace, path)
        except SandboxViolationError as violation:
            return self._error(step, action.tool_name, str(violation))
        if not resolved.is_file():
            return self._error(step, READ_TOOL, ERROR_NOT_A_FILE.format(name=resolved.name))
        try:
            content = self._read_capped(resolved)
        except OSError:
            return self._error(step, READ_TOOL, ERROR_UNREADABLE.format(name=resolved.name))
        return self._ok(step, READ_TOOL, content, target=path)

    def _read_capped(self, path: Path) -> str:
        with path.open("rb") as handle:
            raw = handle.read(self._max_read_bytes)
        return raw.decode("utf-8", errors="replace")

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
