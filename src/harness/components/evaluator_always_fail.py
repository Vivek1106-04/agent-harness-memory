"""AlwaysFailEvaluator - scores every trace zero.

The trivial second implementation of the evaluator layer. Its use is diagnostic: a sweep
run under it must report a flat zero everywhere, so any non-zero cell exposes a scoring
path that is not reading the evaluator's verdict.
"""

from __future__ import annotations

from harness.interfaces import Evaluator
from harness.registry import register
from harness.types import Observation, TaskSpec, Verdict

REASON = "always_fail baseline: no trace is ever accepted"


@register("evaluator", name="always_fail")
class AlwaysFailEvaluator(Evaluator):
    def judge(self, task: TaskSpec, trace: list[Observation]) -> Verdict:
        return Verdict(
            success=False,
            score=0.0,
            per_step=[False] * len(trace),
            reason=REASON,
        )
