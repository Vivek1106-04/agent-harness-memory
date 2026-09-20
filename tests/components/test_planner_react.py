from __future__ import annotations

import pytest

from harness.components.planner_react import (
    INVALID_ACTION_TOOL,
    MAX_ECHOED_CHARS,
    ReActPlanner,
    parse_action,
)
from harness.interfaces import LLMClient, Planner
from harness.registry import resolve
from harness.types import ContextWindow
from tests.contracts.planner import PlannerContract
from tests.fakes import FakeLLMClient

READ_ACTION = '{"tool_name": "read_file", "arguments": {"path": "doc_00.txt"}}'


def _build(max_steps: int, llm: LLMClient) -> Planner:
    return ReActPlanner(llm=llm, max_steps=max_steps)


def _window() -> ContextWindow:
    return ContextWindow(
        segments=[("system", "you are an agent"), ("goal", "find the code")],
        token_count=10,
        dropped=[],
    )


class TestReActPlannerContract(PlannerContract):
    planner_factory = staticmethod(_build)


def test_registered_under_its_config_name() -> None:
    assert resolve("planner", "react") is ReActPlanner


def test_rejects_a_non_positive_step_budget() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        ReActPlanner(llm=FakeLLMClient(["ignored"]), max_steps=0)


def test_parses_an_action_wrapped_in_thought_prose() -> None:
    # Arrange
    reply = f"Thought: I should list the workspace first.\nAction: {READ_ACTION}"

    # Act
    action = parse_action(reply)

    # Assert
    assert action is not None
    assert action.tool_name == "read_file"
    assert action.arguments == {"path": "doc_00.txt"}
    assert action.rationale == "I should list the workspace first."


def test_parses_an_action_inside_a_fenced_code_block() -> None:
    action = parse_action(f"```json\n{READ_ACTION}\n```")

    assert action is not None
    assert action.tool_name == "read_file"


def test_parses_an_action_followed_by_trailing_prose_with_braces() -> None:
    """The greedy match is not valid JSON; retreating to the last brace recovers the step."""
    action = parse_action(f"{READ_ACTION}\nI will then answer with {{the code}}.")

    assert action is not None
    assert action.tool_name == "read_file"


def test_action_without_arguments_defaults_to_an_empty_mapping() -> None:
    action = parse_action('{"tool_name": "list_dir"}')

    assert action is not None
    assert action.arguments == {}


def test_rationale_is_none_when_no_thought_line_is_present() -> None:
    action = parse_action(READ_ACTION)

    assert action is not None
    assert action.rationale is None


def test_non_mapping_arguments_are_refused() -> None:
    """A list reaching the tool layer as `arguments` is unvalidated model output."""
    assert parse_action('{"tool_name": "read_file", "arguments": ["doc_00.txt"]}') is None


def test_blank_tool_name_is_refused() -> None:
    assert parse_action('{"tool_name": "   ", "arguments": {}}') is None


def test_non_string_tool_name_is_refused() -> None:
    assert parse_action('{"tool_name": 7, "arguments": {}}') is None


def test_json_that_is_not_an_object_is_refused() -> None:
    assert parse_action('{"a": [1, 2}') is None


def test_reply_with_no_braces_at_all_is_refused() -> None:
    assert parse_action("I am not going to answer that.") is None


def test_unparseable_reply_becomes_an_invalid_action_the_model_can_see() -> None:
    """The error observation enters the history and the model corrects itself next step."""
    # Arrange
    planner = ReActPlanner(llm=FakeLLMClient(["no action here"]), max_steps=5)

    # Act
    action = planner.next_action(_window())

    # Assert
    assert action is not None
    assert action.tool_name == INVALID_ACTION_TOOL
    assert action.arguments == {"raw": "no action here"}


def test_echoed_raw_reply_is_truncated() -> None:
    # Arrange
    planner = ReActPlanner(llm=FakeLLMClient(["prose " * 500]), max_steps=5)

    # Act
    action = planner.next_action(_window())

    # Assert
    assert action is not None
    assert len(str(action.arguments["raw"])) == MAX_ECHOED_CHARS


def test_terminates_after_consecutive_parse_failures() -> None:
    """A model stuck in a formatting loop must not spend the whole step budget failing."""
    # Arrange
    planner = ReActPlanner(llm=FakeLLMClient(["garbage"]), max_steps=35, max_parse_failures=3)

    # Act
    outcomes = [planner.next_action(_window()) for _ in range(3)]

    # Assert
    assert outcomes[0] is not None
    assert outcomes[1] is not None
    assert outcomes[2] is None


def test_a_good_reply_resets_the_failure_streak() -> None:
    # Arrange
    llm = FakeLLMClient(["garbage", "garbage", READ_ACTION, "garbage"])
    planner = ReActPlanner(llm=llm, max_steps=35, max_parse_failures=3)

    # Act
    outcomes = [planner.next_action(_window()) for _ in range(4)]

    # Assert
    assert outcomes[2] is not None
    assert outcomes[2].tool_name == "read_file"
    assert outcomes[3] is not None
    assert outcomes[3].tool_name == INVALID_ACTION_TOOL


def test_loops_rather_than_answering_once() -> None:
    """The difference from SingleStepPlanner, and the reason the bracket needs this one."""
    # Arrange
    planner = ReActPlanner(llm=FakeLLMClient([READ_ACTION]), max_steps=4)

    # Act
    actions = [planner.next_action(_window()) for _ in range(4)]

    # Assert
    assert all(action is not None and action.tool_name == "read_file" for action in actions)


def test_the_window_it_was_handed_is_what_reaches_the_model() -> None:
    """A planner that rebuilt the prompt would escape the context manager's budget."""
    # Arrange
    llm = FakeLLMClient([READ_ACTION])
    planner = ReActPlanner(llm=llm, max_steps=2)
    window = _window()

    # Act
    planner.next_action(window)

    # Assert
    assert llm.calls == [window]
