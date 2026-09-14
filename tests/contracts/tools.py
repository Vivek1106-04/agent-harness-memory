"""ToolRuntimeContract - the admission criterion for the tool layer.

The highest-risk layer in the harness: it executes actions chosen by a language model.
Every implementation, mock included, refuses exactly the same paths - a mock more permissive
than the real runtime hides the bug it exists to catch. See SECURITY.md section 5.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from harness.interfaces import ToolRuntime
from harness.types import Action, Observation

#: Every tool runtime in this study supports at least this tool, taking a `path` argument
#: relative to the sandbox root. It is what the path-confinement test exercises.
READ_TOOL = "read_file"

ESCAPING_PATHS = [
    "../outside.txt",
    "../../etc/passwd",
    "/etc/passwd",
    "nested/../../outside.txt",
]


class ToolRuntimeContract:
    runtime_factory: Callable[[Path], ToolRuntime]

    @pytest.fixture
    def runtime(self, tmp_path: Path) -> ToolRuntime:
        return type(self).runtime_factory(tmp_path)

    def test_unknown_tool_returns_error_observation_not_exception(
        self, runtime: ToolRuntime
    ) -> None:
        # Arrange
        action = Action(tool_name="definitely_not_a_tool", arguments={})

        # Act
        observation = runtime.execute(action, step=0)

        # Assert
        assert isinstance(observation, Observation)
        assert observation.error is not None

    def test_observation_step_is_monotonic(self, runtime: ToolRuntime) -> None:
        # Arrange
        actions = [Action(tool_name=READ_TOOL, arguments={"path": "notes.txt"}) for _ in range(3)]

        # Act
        steps = [runtime.execute(action, step=i).step for i, action in enumerate(actions)]

        # Assert
        assert steps == sorted(steps)
        assert steps == [0, 1, 2]

    @pytest.mark.parametrize("path", ESCAPING_PATHS)
    def test_path_outside_sandbox_is_refused(self, runtime: ToolRuntime, path: str) -> None:
        # Arrange
        action = Action(tool_name=READ_TOOL, arguments={"path": path})

        # Act
        observation = runtime.execute(action, step=0)

        # Assert
        assert observation.error is not None

    def test_missing_required_argument_returns_error_observation(
        self, runtime: ToolRuntime
    ) -> None:
        """Arguments are validated before execution, and a failure is data, not an exception."""
        # Arrange
        action = Action(tool_name=READ_TOOL, arguments={})

        # Act
        observation = runtime.execute(action, step=0)

        # Assert
        assert observation.error is not None
