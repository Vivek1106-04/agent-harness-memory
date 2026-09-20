from __future__ import annotations

import pytest

from harness.tasks.generator import TaskGenerationError, generate
from harness.tasks.suite import (
    DEFAULT_POOL_PADDING,
    MIN_CHAIN_LENGTH,
    SuiteError,
    SuiteSpec,
    build_suite,
)


def _suite(**overrides: object) -> SuiteSpec:
    defaults: dict[str, object] = {
        "count": 5,
        "seed": 1000,
        "chain_length": 2,
        "recall_distance": 12,
        "distractor_density": 0.5,
        "horizon": 20,
    }
    return SuiteSpec(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_builds_the_requested_number_of_tasks() -> None:
    assert len(build_suite(_suite(count=30))) == 30


def test_the_same_suite_spec_yields_the_identical_task_set() -> None:
    """The set is re-derivable from the config hash, so a row identifies its own tasks."""
    assert build_suite(_suite()) == build_suite(_suite())


def test_every_task_has_a_distinct_id() -> None:
    specs = build_suite(_suite(count=10))

    assert len({spec.task_id for spec in specs}) == 10


def test_every_task_has_a_distinct_seed() -> None:
    """A shared seed would give every task in the suite the same access codes."""
    specs = build_suite(_suite(count=10))

    assert len({spec.seed for spec in specs}) == 10


def test_no_entity_is_shared_between_two_tasks() -> None:
    """A fact met in one run must not be scoreable in another. Cross-run contamination."""
    # Arrange
    specs = build_suite(_suite(count=5))

    # Act
    pools = [set(spec.entity_pool) for spec in specs]

    # Assert
    for index, pool in enumerate(pools):
        others = set().union(*pools[:index], *pools[index + 1 :])
        assert not pool & others


def test_a_chain_of_two_has_one_dependency_edge() -> None:
    spec = build_suite(_suite(chain_length=2))[0]

    assert spec.fact_dependency_graph == {"n_1": ["n_0"]}


def test_a_longer_chain_links_each_fact_to_its_predecessor() -> None:
    spec = build_suite(_suite(chain_length=4, recall_distance=6, horizon=20))[0]

    assert spec.fact_dependency_graph == {"n_1": ["n_0"], "n_2": ["n_1"], "n_3": ["n_2"]}


def test_every_generated_task_holds_one_required_fact_per_chain_link() -> None:
    """The suite and the generator have to agree, or recall_distance means nothing."""
    # Arrange
    specs = build_suite(_suite(count=3, chain_length=3, recall_distance=8, horizon=20))

    # Act
    tasks = [generate(spec) for spec in specs]

    # Assert
    assert [len(task.required) for task in tasks] == [3, 3, 3]


def test_the_entity_pool_covers_the_chain_with_room_for_distractors() -> None:
    """Too small a pool and the generator starts suffixing distractor names with `_2`."""
    spec = build_suite(_suite(chain_length=3, recall_distance=8, horizon=20))[0]

    assert len(spec.entity_pool) == 3 + DEFAULT_POOL_PADDING


def test_an_empty_suite_is_refused() -> None:
    with pytest.raises(SuiteError, match="at least 1"):
        build_suite(_suite(count=0))


def test_a_chain_too_short_to_force_a_hop_is_refused() -> None:
    with pytest.raises(SuiteError, match="force a hop"):
        build_suite(_suite(chain_length=MIN_CHAIN_LENGTH - 1))


def test_a_geometry_the_horizon_cannot_hold_fails_in_the_generator() -> None:
    """The suite builds the spec; the generator is what refuses an impossible one."""
    # Arrange
    spec = build_suite(_suite(chain_length=2, recall_distance=19, horizon=20))[0]

    # Act / Assert
    with pytest.raises(TaskGenerationError, match="cannot hold"):
        generate(spec)
