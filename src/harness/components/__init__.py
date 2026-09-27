"""Implementations shipped with the harness.

Importing this package registers every built-in component. `registry.discover()` imports it
before consulting entry points, so a config can name a built-in without the harness naming
an implementation class anywhere in its own resolution path.
"""

from harness.components import (
    context_budgeted,
    context_full,
    evaluator_always_fail,
    evaluator_checker,
    memory_episodic,
    memory_graph,
    memory_hybrid,
    memory_noop,
    memory_oracle,
    memory_vector,
    planner_react,
    planner_single_step,
    tools_local,
    tools_mock,
    tools_task_docs,
)

__all__ = [
    "context_budgeted",
    "context_full",
    "evaluator_always_fail",
    "evaluator_checker",
    "memory_episodic",
    "memory_graph",
    "memory_hybrid",
    "memory_noop",
    "memory_oracle",
    "memory_vector",
    "planner_react",
    "planner_single_step",
    "tools_local",
    "tools_mock",
    "tools_task_docs",
]
