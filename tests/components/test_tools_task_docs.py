from __future__ import annotations

from pathlib import Path

from harness.components.tools_task_docs import TaskDocumentRuntime
from harness.interfaces import ToolRuntime
from harness.registry import resolve
from harness.tasks.generator import generate
from harness.types import Action
from tests.contracts.tools import ToolRuntimeContract
from tests.tasks.specs import spec


def _build(sandbox_root: Path) -> ToolRuntime:
    return TaskDocumentRuntime(sandbox_root)


def _prepared(sandbox_root: Path) -> tuple[TaskDocumentRuntime, list[tuple[str, str]]]:
    subject = spec()
    runtime = TaskDocumentRuntime(sandbox_root)
    runtime.prepare(subject)
    return runtime, list(generate(subject).documents)


class TestTaskDocumentRuntimeContract(ToolRuntimeContract):
    runtime_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("tools", "task_docs") is TaskDocumentRuntime


def test_reading_a_prepared_document_returns_its_fact(tmp_path: Path) -> None:
    # Arrange
    runtime, documents = _prepared(tmp_path)
    name, content = documents[0]

    # Act
    observation = runtime.execute(Action(tool_name="read_file", arguments={"path": name}), step=1)

    # Assert
    assert observation.error is None
    assert observation.content == content


def test_listing_returns_every_document_in_canonical_order(tmp_path: Path) -> None:
    # Arrange
    runtime, documents = _prepared(tmp_path)

    # Act
    observation = runtime.execute(Action(tool_name="list_dir", arguments={}), step=0)

    # Assert
    assert observation.content.splitlines() == [name for name, _ in documents]


def test_unknown_document_is_an_error_observation(tmp_path: Path) -> None:
    # Arrange
    runtime, _ = _prepared(tmp_path)

    # Act
    observation = runtime.execute(
        Action(tool_name="read_file", arguments={"path": "doc_99.txt"}), step=0
    )

    # Assert
    assert observation.error is not None
    assert "no such document" in observation.error


def test_answer_is_echoed_so_the_evaluator_can_read_it(tmp_path: Path) -> None:
    # Arrange
    runtime, _ = _prepared(tmp_path)

    # Act
    observation = runtime.execute(Action(tool_name="answer", arguments={"text": "4271"}), step=7)

    # Assert
    assert observation.error is None
    assert observation.content == "4271"
    assert observation.tool_name == "answer"


def test_answer_without_text_is_an_error_observation(tmp_path: Path) -> None:
    # Arrange
    runtime, _ = _prepared(tmp_path)

    # Act
    observation = runtime.execute(Action(tool_name="answer", arguments={}), step=0)

    # Assert
    assert observation.error is not None


def test_traversal_is_refused_before_any_lookup(tmp_path: Path) -> None:
    """SECURITY.md section 5 - the confinement check is the same one the real runtime uses."""
    # Arrange
    runtime, _ = _prepared(tmp_path)

    # Act
    observation = runtime.execute(
        Action(tool_name="read_file", arguments={"path": "../doc_00.txt"}), step=0
    )

    # Assert
    assert observation.error is not None
    assert "outside the sandbox" in observation.error


def test_output_is_capped(tmp_path: Path) -> None:
    # Arrange
    runtime = TaskDocumentRuntime(tmp_path, max_output_chars=4)
    runtime.prepare(spec())

    # Act
    observation = runtime.execute(Action(tool_name="answer", arguments={"text": "x" * 500}), 0)

    # Assert
    assert observation.content == "xxxx"


def test_an_unprepared_runtime_serves_nothing(tmp_path: Path) -> None:
    # Arrange
    runtime = TaskDocumentRuntime(tmp_path)

    # Act
    observation = runtime.execute(
        Action(tool_name="read_file", arguments={"path": "doc_00.txt"}), step=0
    )

    # Assert
    assert observation.error is not None


def test_a_read_records_its_target_for_the_prompt(tmp_path: Path) -> None:
    from harness.prompts import TARGET_KEY
    from harness.tasks.generator import generate
    from harness.types import Action
    from tests.tasks.specs import spec

    runtime = TaskDocumentRuntime(tmp_path)
    runtime.prepare(spec())
    name = generate(spec()).documents[0][0]
    observation = runtime.execute(Action("read_file", {"path": name}), step=1)
    assert observation.metadata == {TARGET_KEY: name}
