"""CheckerEvaluator - deterministic scoring against the generated task's own answer.

The checker is the second of the three artefacts SPEC.md section 9 demands from one source.
It is derived, not written: the evaluator re-generates the task from the spec and compares
the agent's answer against the answer the generator resolved. No LLM-as-judge - it would
reintroduce the confound this study exists to remove, and a judge reading model-generated
traces is directly injectable (SECURITY.md section 8).

Scoring:

* `success` - an `answer` observation containing the resolved answer as a complete run of
  digits, so that "4271", "4271." and "the code is 4271" all count while "42718" does not.
  We are measuring recall, not output formatting, and equally not rewarding a near miss.
* `per_step` - did step *i* earn its place: surface a required fact for the first time, or
  give the correct answer. Fact discovery is what memory is supposed to enable, so this is
  the series Experiment 1 reads when a run fails.
* `score` - 1.0 on success, otherwise the fraction of required facts ever surfaced. Partial
  credit keeps a near-miss distinguishable from a run that never found anything.

Week 4 hardens this against the `mock_fs` and `repo_edit` families; v1 covers multihop_qa,
which is the only family the generator emits.
"""

from __future__ import annotations

import re

from harness.interfaces import Evaluator
from harness.registry import register
from harness.tasks.generator import ANSWER_TOOL, generate
from harness.types import Observation, TaskSpec, Verdict

REASON_SUCCESS = "answer matches the generated task answer"
REASON_NO_ANSWER = "no answer observation in the trace"
REASON_WRONG = "answer does not match the generated task answer"


@register("evaluator", name="checker")
class CheckerEvaluator(Evaluator):
    def judge(self, task: TaskSpec, trace: list[Observation]) -> Verdict:
        generated = generate(task)
        required = generated.required

        per_step: list[bool] = []
        surfaced: set[str] = set()
        answered = False
        success = False
        for observation in trace:
            useful = False
            for fact in required:
                if fact.fact_id not in surfaced and fact.content in observation.content:
                    surfaced.add(fact.fact_id)
                    useful = True
            if observation.tool_name == ANSWER_TOOL and observation.error is None:
                answered = True
                if generated.answer in _digit_runs(observation.content):
                    success = True
                    useful = True
            per_step.append(useful)

        score = 1.0 if success else (len(surfaced) / len(required) if required else 0.0)
        return Verdict(
            success=success,
            score=score,
            per_step=per_step,
            reason=REASON_SUCCESS if success else (REASON_WRONG if answered else REASON_NO_ANSWER),
        )


def _digit_runs(text: str) -> list[str]:
    return re.findall(r"\d+", text)
