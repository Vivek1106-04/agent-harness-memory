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
