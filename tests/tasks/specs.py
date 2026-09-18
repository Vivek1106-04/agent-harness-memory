"""One place to build task specs for tests, so a geometry change updates one file."""

from __future__ import annotations

from harness.tasks.generator import FAMILY_MULTIHOP_QA
from harness.types import TaskSpec

DEFAULT_POOL = ["alpha", "beta", "gamma", "delta"]
#: {"code": ["origin"]} - a two-fact chain whose answer is the origin fact's literal code.
DEFAULT_GRAPH = {"code": ["origin"]}


def spec(
    *,
    task_id: str = "task-1",
    family: str = FAMILY_MULTIHOP_QA,
    graph: dict[str, list[str]] | None = None,
    recall_distance: int = 5,
    distractor_density: float = 0.5,
    horizon: int = 12,
    pool: list[str] | None = None,
    seed: int = 1,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        family=family,
        fact_dependency_graph=DEFAULT_GRAPH if graph is None else graph,
        recall_distance=recall_distance,
        distractor_density=distractor_density,
        horizon=horizon,
        entity_pool=DEFAULT_POOL if pool is None else pool,
        seed=seed,
    )
