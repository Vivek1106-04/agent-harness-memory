"""Task generator v1 - one `TaskSpec` in, three artefacts out.

SPEC.md section 9: the generator emits the task, its checker and its per-step gold labels
from a single source. That single source is what makes `OracleMemory` and retrieval
precision/recall possible at all - a hand-written task has no gold labels, so the ceiling
condition cannot be constructed for it and retrieval quality cannot be separated from agent
quality.

Generation is a **pure function of the spec**. Nothing here reads a clock, a file or a
global. Any layer holding a `TaskSpec` can re-derive the identical task, which is why the
evaluator and the oracle need no side channel to the generator.

Family `multihop_qa` only in v1. `mock_fs` and `repo_edit` follow once the loop is real.

Geometry, and why it is fixed this way:

    documents = horizon - 1            one step is reserved for the answer
    the terminal fact - the one the goal names - is the LAST document
    the facts it depends on occupy a contiguous block ending `recall_distance`
    documents before it; every other document is a distractor

So `recall_distance` is the number of documents between the last supporting fact and the
moment it is needed: the agent only learns a supporting fact matters when it reads the
terminal one, at the end. That gap is the whole point: a task whose fact is still inside
the recent window is solvable without memory and contributes pure noise.

Why the terminal fact is last (v2, 2026-09-27). v1 placed every required fact in one
contiguous block, so an agent reading in order found the whole chain together and answered
at once - NoOp solved it in the first real run, and the recall distance only existed for an
agent that kept reading after it had the answer. `distractor_density` acts as
a floor on the fraction of documents that are irrelevant, and generation fails loudly when
the dependency graph is too large for the horizon to honour it.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from harness.tasks.padding import pad
from harness.types import TaskSpec

FAMILY_MULTIHOP_QA = "multihop_qa"
ATTRIBUTE = "access code"
ANSWER_TOOL = "answer"
VALUE_MIN = 1000
VALUE_MAX = 9999
SUFFIX_MIN = 10
SUFFIX_MAX = 99
#: Bumped whenever the same spec would generate a different task. Part of every task id,
#: so rows from two generator versions can never be merged as the same task.
GENERATOR_VERSION = 2
#: Characters of the spec digest used to seed the generator's RNG.
SEED_DIGEST_CHARS = 16


class TaskGenerationError(Exception):
    """A spec that cannot produce a coherent task. Never a silently degraded task."""


@dataclass(frozen=True)
class Fact:
    """One document's worth of content, and where it sits in the canonical trajectory."""

    fact_id: str
    entity: str
    content: str
    doc_name: str
    index: int
    is_distractor: bool


@dataclass(frozen=True)
class GeneratedTask:
    """The task, its answer (the checker's source of truth), and its gold labels.

    `gold_labels[i]` is the set of required fact ids a perfect memory would hold after
    document `i` of the canonical trajectory - documents read in listed order. It is the
    reference trajectory retrieval precision and recall are scored against.
    """

    spec: TaskSpec
    goal: str
    documents: tuple[tuple[str, str], ...]
    required: tuple[Fact, ...]
    distractors: tuple[Fact, ...]
    answer: str
    gold_labels: tuple[frozenset[str], ...]

    @property
    def answer_step(self) -> int:
        """Canonical step at which the answer is due, once every document is readable."""
        return len(self.documents)


def generate(spec: TaskSpec) -> GeneratedTask:
    """Build the task described by `spec`. Deterministic in the spec alone."""
    if spec.family != FAMILY_MULTIHOP_QA:
        raise TaskGenerationError(
            f"task {spec.task_id}: family {spec.family!r} is not implemented in generator v1; "
            f"only {FAMILY_MULTIHOP_QA!r} is"
        )
    order = _topological_order(spec)
    terminal = _terminal_node(spec, order)
    entities = _assign_entities(spec, order)
    rng = _rng(spec)
    values, suffixes = _resolve_values(spec, order, rng)

    total_docs, first_index = _geometry(spec, len(order))
    required = _required_facts(order, entities, values, suffixes, spec, first_index)
    distractors = _distractor_facts(spec, total_docs, required, set(entities.values()), rng)
    documents = _documents(spec, total_docs, required, distractors)

    return GeneratedTask(
        spec=spec,
        goal=_goal(entities[terminal], total_docs),
        documents=documents,
        required=required,
        distractors=distractors,
        answer=values[terminal],
        gold_labels=_gold_labels(total_docs, required),
    )


def _rng(spec: TaskSpec) -> random.Random:
    """Seeded from the spec digest, so the same spec yields the same values everywhere."""
    digest = hashlib.sha256(f"{spec.task_id}:{spec.seed}".encode()).hexdigest()
    return random.Random(int(digest[:SEED_DIGEST_CHARS], 16))  # noqa: S311 - not cryptographic


def _nodes(spec: TaskSpec) -> list[str]:
    seen: list[str] = []
    for node, dependencies in spec.fact_dependency_graph.items():
        for name in (node, *dependencies):
            if name not in seen:
                seen.append(name)
    return seen


def _topological_order(spec: TaskSpec) -> list[str]:
    """Dependencies before dependants. A cycle is a generation error, not a hang."""
    nodes = _nodes(spec)
    if not nodes:
        raise TaskGenerationError(f"task {spec.task_id}: fact_dependency_graph is empty")
    order: list[str] = []
    visiting: set[str] = set()

    def visit(node: str) -> None:
        if node in order:
            return
        if node in visiting:
            raise TaskGenerationError(
                f"task {spec.task_id}: fact_dependency_graph has a cycle through {node!r}"
            )
        visiting.add(node)
        for dependency in spec.fact_dependency_graph.get(node, []):
            visit(dependency)
        visiting.discard(node)
        order.append(node)

    for node in nodes:
        visit(node)
    return order


def _terminal_node(spec: TaskSpec, order: list[str]) -> str:
    """The one fact nothing else depends on. Two terminals means two questions."""
    depended_on = {dep for deps in spec.fact_dependency_graph.values() for dep in deps}
    terminals = [node for node in order if node not in depended_on]
    if len(terminals) != 1:
        raise TaskGenerationError(
            f"task {spec.task_id}: expected exactly one terminal fact, found {sorted(terminals)}"
        )
    return terminals[0]


def _assign_entities(spec: TaskSpec, order: list[str]) -> dict[str, str]:
    if len(spec.entity_pool) < len(order):
        raise TaskGenerationError(
            f"task {spec.task_id}: entity_pool holds {len(spec.entity_pool)} entities but the "
            f"dependency graph needs {len(order)}"
        )
    if len(set(spec.entity_pool)) != len(spec.entity_pool):
        raise TaskGenerationError(f"task {spec.task_id}: entity_pool contains duplicates")
    return {node: spec.entity_pool[index] for index, node in enumerate(order)}


def _resolve_values(
    spec: TaskSpec, order: list[str], rng: random.Random
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (values, suffixes). A root fact carries a literal code; a derived fact is its
    dependencies' codes concatenated, followed by a literal suffix of its own.

    Concatenation rather than arithmetic on purpose: a sum would measure the model's
    arithmetic alongside its recall, and that is a confound we are not funding.

    The suffix (v2) means the answer appears verbatim in no single document. Without it a
    two-fact chain's answer was the root's code, and a model that answered with the first
    code it found for a similarly named entity scored a success it had not earned - which
    is exactly what the first real run did.
    """
    values: dict[str, str] = {}
    suffixes: dict[str, str] = {}
    for node in order:
        dependencies = spec.fact_dependency_graph.get(node, [])
        if dependencies:
            suffixes[node] = str(rng.randint(SUFFIX_MIN, SUFFIX_MAX))
            values[node] = "".join(values[dep] for dep in dependencies) + suffixes[node]
        else:
            values[node] = str(rng.randint(VALUE_MIN, VALUE_MAX))
    return values, suffixes


def _geometry(spec: TaskSpec, required_count: int) -> tuple[int, int]:
    """Return (document count, index of the first required document)."""
    if spec.horizon < 2:
        raise TaskGenerationError(f"task {spec.task_id}: horizon must leave a step for the answer")
    if spec.recall_distance < 1:
        raise TaskGenerationError(f"task {spec.task_id}: recall_distance must be at least 1")
    if not 0.0 <= spec.distractor_density < 1.0:
        raise TaskGenerationError(
            f"task {spec.task_id}: distractor_density must lie in [0.0, 1.0), "
            f"got {spec.distractor_density}"
        )
    total_docs = spec.horizon - 1
    first_index = total_docs - spec.recall_distance - required_count + 1
    if first_index < 0:
        raise TaskGenerationError(
            f"task {spec.task_id}: horizon {spec.horizon} cannot hold {required_count} required "
            f"facts at recall_distance {spec.recall_distance}"
        )
    distractor_count = total_docs - required_count
    floor = round(spec.distractor_density * total_docs)
    if distractor_count < floor:
        raise TaskGenerationError(
            f"task {spec.task_id}: distractor_density {spec.distractor_density} needs {floor} "
            f"distractor documents but the horizon leaves room for {distractor_count}"
        )
    return total_docs, first_index


def _required_facts(
    order: list[str],
    entities: dict[str, str],
    values: dict[str, str],
    suffixes: dict[str, str],
    spec: TaskSpec,
    first_index: int,
) -> tuple[Fact, ...]:
    facts: list[Fact] = []
    terminal_index = spec.horizon - 2  # the last document; the step after it is the answer
    for offset, node in enumerate(order):
        is_terminal = offset == len(order) - 1
        index = terminal_index if is_terminal else first_index + offset
        dependencies = spec.fact_dependency_graph.get(node, [])
        if dependencies:
            chain = " followed by ".join(
                [*(f"the {ATTRIBUTE} for {entities[dep]}" for dep in dependencies), suffixes[node]]
            )
            content = f"The {ATTRIBUTE} for {entities[node]} is {chain}."
        else:
            content = f"The {ATTRIBUTE} for {entities[node]} is {values[node]}."
        facts.append(
            Fact(
                fact_id=node,
                entity=entities[node],
                content=content,
                doc_name=_doc_name(index),
                index=index,
                is_distractor=False,
            )
        )
    return tuple(facts)


def _distractor_facts(
    spec: TaskSpec,
    total_docs: int,
    required: tuple[Fact, ...],
    used_entities: set[str],
    rng: random.Random,
) -> tuple[Fact, ...]:
    taken = {fact.index for fact in required}
    indices = [index for index in range(total_docs) if index not in taken]
    entities = _distractor_entities(len(indices), spec.entity_pool, used_entities, rng)
    return tuple(
        Fact(
            fact_id=f"distractor_{index:02d}",
            entity=entity,
            content=f"The {ATTRIBUTE} for {entity} is {rng.randint(VALUE_MIN, VALUE_MAX)}.",
            doc_name=_doc_name(index),
            index=index,
            is_distractor=True,
        )
        for index, entity in zip(indices, entities, strict=True)
    )


def _distractor_entities(
    count: int, pool: list[str], used: set[str], rng: random.Random
) -> list[str]:
    """Unused pool entries first, then suffixed variants once the pool is exhausted.

    A distractor must never name a required entity, or it stops being a distractor and
    starts being a contradiction.
    """
    available = [entity for entity in pool if entity not in used]
    rng.shuffle(available)
    taken = set(used)
    names: list[str] = []
    suffix = 2
    while len(names) < count:
        if available:
            name = available.pop()
        else:
            name = f"{pool[len(names) % len(pool)]}_{suffix}"
            suffix += 1
        if name in taken:
            continue
        taken.add(name)
        names.append(name)
    return names


def _documents(
    spec: TaskSpec, total_docs: int, required: tuple[Fact, ...], distractors: tuple[Fact, ...]
) -> tuple[tuple[str, str], ...]:
    by_index = {fact.index: fact for fact in (*required, *distractors)}
    documents = []
    for index in range(total_docs):
        fact = by_index[index]
        key = f"{spec.task_id}:{spec.seed}:{fact.doc_name}"
        documents.append((fact.doc_name, pad(fact.content, spec.document_tokens, key)))
    return tuple(documents)


def _gold_labels(total_docs: int, required: tuple[Fact, ...]) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(fact.fact_id for fact in required if fact.index <= step)
        for step in range(total_docs + 1)
    )


def _doc_name(index: int) -> str:
    return f"doc_{index:02d}.txt"


def _goal(entity: str, total_docs: int) -> str:
    """Names the document range because the goal is never evicted and a listing is: an
    agent that has lost the listing otherwise cannot tell how far it has to read, and in
    the first real run it restarted from the first document instead of reading on."""
    return (
        f"Find the {ATTRIBUTE} for {entity}. The workspace holds {total_docs} documents, "
        f"{_doc_name(0)} to {_doc_name(total_docs - 1)}; together they hold the facts you "
        f"need. Read them and follow the references between them. When you know the code, "
        f"call {ANSWER_TOOL} with the code and nothing else."
    )
