"""Implementations shipped with the harness.

Importing this package registers every built-in component. `registry.discover()` imports it
before consulting entry points, so a config can name a built-in without the harness naming
an implementation class anywhere in its own resolution path.
"""

from harness.components import (
    context_full,
    evaluator_always_fail,
    planner_single_step,
    tools_mock,
)

__all__ = [
    "context_full",
    "evaluator_always_fail",
    "planner_single_step",
    "tools_mock",
]
