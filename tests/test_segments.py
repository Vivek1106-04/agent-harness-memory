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
