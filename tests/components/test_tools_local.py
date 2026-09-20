from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from harness.components.tools_local import LocalToolRuntime
from harness.interfaces import ToolRuntime
from harness.registry import resolve
from harness.tasks.generator import generate
from harness.types import Action
from tests.contracts.tools import ToolRuntimeContract
from tests.tasks.specs import spec


def _build(sandbox_root: Path) -> ToolRuntime:
    return LocalToolRuntime(sandbox_root=sandbox_root)


def _read(name: str) -> Action:
    return Action(tool_name="read_file", arguments={"path": name})


class TestLocalToolRuntimeContract(ToolRuntimeContract):
    runtime_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("tools", "local") is LocalToolRuntime


def test_prepare_writes_the_task_documents_to_disk(tmp_path: Path) -> None:
    """The point of this runtime: the confinement code meets a real filesystem."""
    # Arrange
    subject = spec()
    task = generate(subject)
    runtime = LocalToolRuntime(sandbox_root=tmp_path)

    # Act
    runtime.prepare(subject)

    # Assert
    workspace = tmp_path / subject.task_id
    for name, content in task.documents:
        assert (workspace / name).read_text(encoding="utf-8") == content


def test_reads_back_a_document_it_wrote(tmp_path: Path) -> None:
    # Arrange
    subject = spec()
    task = generate(subject)
    runtime = LocalToolRuntime(sandbox_root=tmp_path)
    runtime.prepare(subject)
    name, content = task.documents[0]

    # Act
    observation = runtime.execute(_read(name), step=1)

    # Assert
    assert observation.error is None
    assert observation.content == content


def test_list_dir_names_every_document(tmp_path: Path) -> None:
    # Arrange
    subject = spec()
    task = generate(subject)
    runtime = LocalToolRuntime(sandbox_root=tmp_path)
    runtime.prepare(subject)

    # Act
    observation = runtime.execute(Action(tool_name="list_dir", arguments={}), step=1)

    # Assert
    assert observation.content.splitlines() == [name for name, _ in task.documents]


def test_a_second_prepare_re_roots_rather_than_nesting(tmp_path: Path) -> None:
    """Two runs in one process must not leave the second reading through the first."""
    # Arrange
    runtime = LocalToolRuntime(sandbox_root=tmp_path)
    first = spec(task_id="task-1")
    second = spec(task_id="task-2")

    # Act
    runtime.prepare(first)
    runtime.prepare(second)
    observation = runtime.execute(_read(generate(second).documents[0][0]), step=1)

    # Assert
    assert observation.error is None
    assert (tmp_path / "task-2").is_dir()


def test_answer_returns_the_text_it_was_given(tmp_path: Path) -> None:
    runtime = LocalToolRuntime(sandbox_root=tmp_path)

    observation = runtime.execute(Action(tool_name="answer", arguments={"text": "4271"}), step=2)

    assert observation.error is None
    assert observation.content == "4271"


def test_answer_without_text_is_an_error_observation(tmp_path: Path) -> None:
    runtime = LocalToolRuntime(sandbox_root=tmp_path)

    observation = runtime.execute(Action(tool_name="answer", arguments={}), step=2)

    assert observation.error is not None


def test_non_string_path_is_an_error_observation(tmp_path: Path) -> None:
    runtime = LocalToolRuntime(sandbox_root=tmp_path)

    observation = runtime.execute(Action(tool_name="read_file", arguments={"path": 7}), step=1)

    assert observation.error is not None


def test_a_directory_is_refused_rather_than_opened(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "subdir").mkdir()
    runtime = LocalToolRuntime(sandbox_root=tmp_path)

    # Act
    observation = runtime.execute(_read("subdir"), step=1)

    # Assert
    assert observation.error is not None
    assert "regular file" in observation.error


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFOs only")
def test_a_fifo_is_refused_before_the_open_blocks_forever(tmp_path: Path) -> None:
    """No timeout on the loop can interrupt a blocking read, so the open never happens."""
    # Arrange
    os.mkfifo(tmp_path / "pipe")
    runtime = LocalToolRuntime(sandbox_root=tmp_path)

    # Act
    observation = runtime.execute(_read("pipe"), step=1)

    # Assert
    assert observation.error is not None
    assert "regular file" in observation.error


def test_a_read_stops_at_the_byte_cap(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")
    runtime = LocalToolRuntime(sandbox_root=tmp_path, max_read_bytes=100)

    # Act
    observation = runtime.execute(_read("big.txt"), step=1)

    # Assert
    assert len(observation.content) == 100


def test_the_character_cap_applies_after_the_byte_cap(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")
    runtime = LocalToolRuntime(sandbox_root=tmp_path, max_output_chars=50, max_read_bytes=4000)

    # Act
    observation = runtime.execute(_read("big.txt"), step=1)

    # Assert
    assert len(observation.content) == 50


def test_undecodable_bytes_are_replaced_rather_than_raising(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "raw.bin").write_bytes(b"\xff\xfe valid tail")
    runtime = LocalToolRuntime(sandbox_root=tmp_path)

    # Act
    observation = runtime.execute(_read("raw.bin"), step=1)

    # Assert
    assert observation.error is None
    assert "valid tail" in observation.content


def test_an_unreadable_file_is_an_error_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    (tmp_path / "locked.txt").write_text("secret", encoding="utf-8")
    runtime = LocalToolRuntime(sandbox_root=tmp_path)

    def _refuse(*args: object, **kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "open", _refuse)

    # Act
    observation = runtime.execute(_read("locked.txt"), step=1)

    # Assert
    assert observation.error is not None
    assert "could not be read" in observation.error


def test_a_symlink_out_of_the_sandbox_is_refused(tmp_path: Path) -> None:
    """Both sides are fully resolved before the descendant check, so a link cannot escape."""
    # Arrange
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("not yours", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.txt").symlink_to(outside)
    runtime = LocalToolRuntime(sandbox_root=root)

    # Act
    observation = runtime.execute(_read("link.txt"), step=1)

    # Assert
    assert observation.error is not None
