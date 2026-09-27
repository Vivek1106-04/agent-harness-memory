from __future__ import annotations

import pytest

from harness.tasks.generator import TaskGenerationError, generate
from tests.tasks.specs import spec


def test_generation_is_a_pure_function_of_the_spec() -> None:
    """Every layer re-derives the task from the spec. If that is not identical, nothing works."""
    # Arrange
    subject = spec()

    # Act
    first, second = generate(subject), generate(subject)

    # Assert
    assert first == second


def test_different_seeds_produce_different_answers() -> None:
    # Arrange / Act
    first, second = generate(spec(seed=1)), generate(spec(seed=2))

    # Assert
    assert first.answer != second.answer


def test_task_id_participates_in_the_seed() -> None:
    """Two tasks sharing a seed but not an id must not be the same task."""
    assert generate(spec(task_id="a")).answer != generate(spec(task_id="b")).answer


def test_document_count_reserves_one_step_for_the_answer() -> None:
    task = generate(spec(horizon=12))
    assert len(task.documents) == 11
    assert task.answer_step == 11


def test_the_terminal_fact_is_the_last_document() -> None:
    """The agent learns what it needs only at the end, so it cannot answer early."""
    # Act
    task = generate(spec(recall_distance=5, horizon=12))

    # Assert
    assert task.required[-1].index == len(task.documents) - 1


def test_last_supporting_fact_sits_recall_distance_before_the_terminal() -> None:
    # Arrange
    subject = spec(recall_distance=5, horizon=12)

    # Act
    task = generate(subject)

    # Assert
    terminal, supporting = task.required[-1], task.required[:-1]
    assert terminal.index - max(fact.index for fact in supporting) == 5


def test_supporting_facts_are_contiguous_and_in_dependency_order() -> None:
    # Arrange
    subject = spec(graph={"code": ["middle"], "middle": ["origin"]}, recall_distance=4, horizon=14)

    # Act
    task = generate(subject)

    # Assert
    assert [fact.fact_id for fact in task.required] == ["origin", "middle", "code"]
    indices = [fact.index for fact in task.required]
    assert indices[:2] == [indices[0], indices[0] + 1]
    assert indices[2] == len(task.documents) - 1


def test_answer_is_the_chain_resolved_through_its_dependencies() -> None:
    # Arrange
    subject = spec(graph={"code": ["left", "right"]}, recall_distance=3, horizon=14)

    # Act
    task = generate(subject)

    # Assert
    roots = [fact for fact in task.required if fact.fact_id in {"left", "right"}]
    literals = [fact.content.rsplit(" is ", maxsplit=1)[1].rstrip(".") for fact in roots]
    suffix = task.required[-1].content.rsplit(" followed by ", maxsplit=1)[1].rstrip(".")
    assert task.answer == "".join(literals) + suffix


def test_the_answer_appears_verbatim_in_no_single_document() -> None:
    """A guess of the nearest-looking code must not be scored as recall."""
    task = generate(spec())
    assert not any(task.answer in text for _, text in task.documents)


def test_the_goal_names_the_document_range() -> None:
    task = generate(spec(horizon=12))
    assert "11 documents, doc_00.txt to doc_10.txt" in task.goal


def test_derived_fact_names_its_dependencies_rather_than_their_values() -> None:
    """The chain is what forces multi-hop recall; a derived fact holding the literal would
    make every earlier document redundant."""
    # Arrange
    task = generate(spec())

    # Act
    derived = next(fact for fact in task.required if fact.fact_id == "code")

    # Assert
    assert "the access code for" in derived.content.split(" is ", maxsplit=1)[1]
    assert task.answer not in derived.content


def test_documents_match_their_facts_and_have_unique_names() -> None:
    # Arrange
    task = generate(spec())

    # Act
    names = [name for name, _ in task.documents]
    contents = {name: text for name, text in task.documents}

    # Assert
    assert len(set(names)) == len(names)
    for fact in (*task.required, *task.distractors):
        assert contents[fact.doc_name] == fact.content


def test_distractors_fill_every_remaining_document() -> None:
    # Arrange
    task = generate(spec())

    # Assert
    assert len(task.required) + len(task.distractors) == len(task.documents)
    assert all(fact.is_distractor for fact in task.distractors)


def test_distractors_never_name_a_required_entity() -> None:
    """A distractor naming a required entity is not a distractor, it is a contradiction."""
    # Arrange
    task = generate(spec())

    # Act
    required_entities = {fact.entity for fact in task.required}

    # Assert
    assert required_entities.isdisjoint(fact.entity for fact in task.distractors)


def test_distractor_entities_are_suffixed_once_the_pool_is_exhausted() -> None:
    # Arrange
    subject = spec(pool=["alpha", "beta"], recall_distance=5, horizon=12)

    # Act
    task = generate(subject)

    # Assert
    entities = [fact.entity for fact in task.distractors]
    assert len(set(entities)) == len(entities)
    assert any("_" in entity for entity in entities)


def test_a_suffixed_name_already_in_the_pool_is_skipped_rather_than_duplicated() -> None:
    # Arrange
    subject = spec(
        graph={"code": ["middle"], "middle": ["origin"]},
        pool=["alpha", "beta", "alpha_2"],
        recall_distance=4,
        horizon=14,
    )

    # Act
    task = generate(subject)

    # Assert
    entities = [fact.entity for fact in (*task.required, *task.distractors)]
    assert len(set(entities)) == len(entities)


def test_gold_labels_accumulate_to_the_full_required_set() -> None:
    # Arrange
    task = generate(spec())

    # Act
    labels = task.gold_labels

    # Assert
    assert len(labels) == len(task.documents) + 1
    assert labels[0] == frozenset()
    assert labels[-1] == {fact.fact_id for fact in task.required}
    assert all(labels[i] <= labels[i + 1] for i in range(len(labels) - 1))


def test_goal_names_the_terminal_entity_and_the_answer_tool() -> None:
    # Arrange
    task = generate(spec())

    # Act
    terminal = next(fact for fact in task.required if fact.fact_id == "code")

    # Assert
    assert terminal.entity in task.goal
    assert "answer" in task.goal


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"family": "repo_edit"}, "not implemented"),
        ({"graph": {}}, "empty"),
        ({"graph": {"a": ["b"], "b": ["a"]}}, "cycle"),
        ({"graph": {"one": ["origin"], "two": ["origin"]}}, "terminal"),
        ({"pool": ["alpha"]}, "entity_pool holds"),
        ({"pool": ["alpha", "alpha", "beta"]}, "duplicates"),
        ({"horizon": 1}, "horizon must leave"),
        ({"recall_distance": 0}, "recall_distance must be at least 1"),
        ({"distractor_density": 1.0}, "distractor_density must lie"),
        ({"distractor_density": -0.1}, "distractor_density must lie"),
        ({"recall_distance": 11, "horizon": 12}, "cannot hold"),
        ({"distractor_density": 0.95, "horizon": 12}, "needs 10 distractor documents"),
    ],
)
def test_incoherent_specs_fail_loudly(kwargs: dict[str, object], message: str) -> None:
    """A degraded task is worse than no task: it produces plausible-looking wrong numbers."""
    with pytest.raises(TaskGenerationError, match=message):
        generate(spec(**kwargs))  # type: ignore[arg-type]
