"""Task sets. The generator makes one task; the bracket needs thirty of them.

A suite is a pure function of a `SuiteSpec`, for the same reason a task is a pure function
of a `TaskSpec`: the whole set can be re-derived from the config that is hashed into the
trace header, so "which thirty tasks was this run scored against" is answerable from a
results row alone rather than from a file someone kept.

Chain shape. A chain of length k is `n_0 <- n_1 <- ... <- n_{k-1}`, where `n_0` holds a
literal code and every later fact points at its predecessor. `n_{k-1}` is the terminal the
goal asks for, so answering it requires walking the whole chain. Length 2 is the shortest
chain that forces a hop at all.

Entities are named per task rather than drawn from a shared pool, so two tasks in a suite
never discuss the same entity. A shared entity across tasks is a route by which a run could
be scored on a fact it met in a different run, and cross-run contamination produces
plausible-looking wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.tasks.generator import FAMILY_MULTIHOP_QA
from harness.types import TaskSpec

#: Spare entities beyond the chain itself. The generator suffixes names once the pool runs
#: out, so this only has to keep the common case free of `_2` suffixes in distractors.
DEFAULT_POOL_PADDING = 8
DEFAULT_ENTITY_PREFIX = "site"
MIN_CHAIN_LENGTH = 2


class SuiteError(Exception):
    """A suite description that cannot produce tasks. Never a silently smaller suite."""


@dataclass(frozen=True)
class SuiteSpec:
    """Everything that varies across a task set, and nothing that varies within one."""

    count: int
    seed: int
    chain_length: int
    recall_distance: int
    distractor_density: float
    horizon: int
    family: str = FAMILY_MULTIHOP_QA
    task_id_prefix: str = "task"
    pool_padding: int = DEFAULT_POOL_PADDING


def build_suite(suite: SuiteSpec) -> list[TaskSpec]:
    """Return `suite.count` task specs. Deterministic in the suite spec alone."""
    if suite.count < 1:
        raise SuiteError(f"suite count must be at least 1, got {suite.count}")
    if suite.chain_length < MIN_CHAIN_LENGTH:
        raise SuiteError(
            f"chain_length must be at least {MIN_CHAIN_LENGTH} to force a hop, "
            f"got {suite.chain_length}"
        )
    return [_task_spec(suite, index) for index in range(suite.count)]


def _task_spec(suite: SuiteSpec, index: int) -> TaskSpec:
    return TaskSpec(
        task_id=f"{suite.task_id_prefix}-{index:03d}",
        family=suite.family,
        fact_dependency_graph=_chain(suite.chain_length),
        recall_distance=suite.recall_distance,
        distractor_density=suite.distractor_density,
        horizon=suite.horizon,
        entity_pool=_entity_pool(suite, index),
        #: Offset rather than shared, so the same suite gives every task different codes
        #: while the suite as a whole stays reproducible from one number.
        seed=suite.seed + index,
    )


def _chain(length: int) -> dict[str, list[str]]:
    return {f"n_{position}": [f"n_{position - 1}"] for position in range(1, length)}


def _entity_pool(suite: SuiteSpec, index: int) -> list[str]:
    size = suite.chain_length + suite.pool_padding
    return [f"{DEFAULT_ENTITY_PREFIX}_{index:03d}_{slot:02d}" for slot in range(size)]
