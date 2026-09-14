from __future__ import annotations

from pathlib import Path

from harness.components.tools_mock import CANNED_FILE_CONTENT, MockToolRuntime
from harness.interfaces import ToolRuntime
from harness.types import Action
from tests.contracts.tools import ToolRuntimeContract


def _build(sandbox_root: Path) -> ToolRuntime:
    return MockToolRuntime(sandbox_root)


class TestMockToolRuntimeContract(ToolRuntimeContract):
    runtime_factory = staticmethod(_build)


def test_read_inside_sandbox_returns_canned_content(tmp_path: Path) -> None:
    # Arrange
    runtime = MockToolRuntime(tmp_path)
    action = Action(tool_name="read_file", arguments={"path": "notes/day1.txt"})

    # Act
    observation = runtime.execute(action, step=0)

    # Assert
    assert observation.error is None
    assert CANNED_FILE_CONTENT in observation.content


def test_output_is_capped(tmp_path: Path) -> None:
    # Arrange
    runtime = MockToolRuntime(tmp_path, max_output_chars=10)
    action = Action(tool_name="echo", arguments={"text": "x" * 500})

    # Act
    observation = runtime.execute(action, step=0)

    # Assert
    assert len(observation.content) == 10


def test_text_tool_without_text_is_refused(tmp_path: Path) -> None:
    # Arrange
    runtime = MockToolRuntime(tmp_path)
    action = Action(tool_name="echo", arguments={})

    # Act
    observation = runtime.execute(action, step=0)

    # Assert
    assert observation.error is not None


def test_non_string_path_argument_is_refused(tmp_path: Path) -> None:
    # Arrange
    runtime = MockToolRuntime(tmp_path)
    action = Action(tool_name="read_file", arguments={"path": {"nested": "object"}})

    # Act
    observation = runtime.execute(action, step=0)

    # Assert
    assert observation.error is not None


def test_nothing_is_written_to_the_sandbox(tmp_path: Path) -> None:
    """The mock touches no filesystem at all. SECURITY.md section 5."""
    # Arrange
    runtime = MockToolRuntime(tmp_path)

    # Act
    runtime.execute(Action(tool_name="write_file", arguments={"path": "out.txt"}), step=0)

    # Assert
    assert list(tmp_path.iterdir()) == []
