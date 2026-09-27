from __future__ import annotations

from harness.segments import PROTECTED_ROLES, dropped_entry, memory_id, observation_id


def test_identifiers_name_their_source() -> None:
    assert observation_id(7) == "obs:7"
    assert memory_id("run-1:step-3") == "mem:run-1:step-3"


def test_a_dropped_entry_carries_an_identifier_and_a_reason() -> None:
    entry = dropped_entry(observation_id(7), "budget")
    assert entry == "obs:7|budget"


def test_protected_roles_are_system_and_goal() -> None:
    assert frozenset({"system", "goal"}) == PROTECTED_ROLES


def test_a_read_is_rendered_with_the_document_it_read() -> None:
    """Without the target the agent cannot tell which documents it has already read."""
    from harness.prompts import TARGET_KEY, render_observation
    from harness.types import Observation

    observation = Observation(
        step=3, content="text", tool_name="read_file", metadata={TARGET_KEY: "doc_03.txt"}
    )
    assert render_observation(observation) == "[step 3] read_file(doc_03.txt): text"


def test_an_observation_without_a_target_renders_as_before() -> None:
    from harness.prompts import render_observation
    from harness.types import Observation

    assert render_observation(Observation(step=1, content="a", tool_name="list_dir")) == (
        "[step 1] list_dir: a"
    )


def test_a_memory_is_labelled_with_the_step_it_came_from() -> None:
    from harness.prompts import render_memory
    from harness.types import MemoryItem

    item = MemoryItem(content="fact.", score=1.0, provenance="run:a:12", kind="episodic")
    assert render_memory(item) == "[memory from step 12] fact."


def test_a_memory_without_a_step_is_labelled_plainly() -> None:
    from harness.prompts import render_memory
    from harness.types import MemoryItem

    item = MemoryItem(content="fact.", score=1.0, provenance="external", kind="semantic")
    assert render_memory(item) == "[memory] fact."


def test_the_action_log_lists_each_step_with_its_target_and_errors() -> None:
    from harness.prompts import TARGET_KEY, render_action_log
    from harness.types import Observation

    history = [
        Observation(step=1, content="doc_00.txt", tool_name="list_dir"),
        Observation(
            step=2, content="x", tool_name="read_file", metadata={TARGET_KEY: "doc_00.txt"}
        ),
        Observation(step=3, content="", tool_name=None, error="unparseable"),
    ]
    assert render_action_log(history) == (
        "Actions taken so far: 1 list_dir; 2 read_file(doc_00.txt); 3 none (error)"
    )


def test_no_history_means_no_action_log() -> None:
    from harness.prompts import render_action_log

    assert render_action_log([]) is None
