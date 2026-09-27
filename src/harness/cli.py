"""`python -m harness run <config>` - one config, one task set, one row per task.

This is the command the week-3 gate is answered with. Run it against
`configs/week3_bracket_noop.yaml`, run it against `configs/week3_bracket_oracle.yaml`, and
the difference between the two `success` columns is the bracket.

Everything it does is already owned by a module: the suite comes from `tasks.suite`, the
components from `build`, the sequencing from `loop`, resume from `runner`, and the row from
`results`. What is left here is argument parsing and the provenance header, which is why
there is no logic in this file worth a second implementation.

Resume is the default and not a flag. A sweep interrupted at task 19 of 30 continues at 20.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from harness.build import BuiltRun, build_llm, build_loop
from harness.config import HarnessConfig, load_config
from harness.interfaces import LLMClient
from harness.loop import LoopOutcome
from harness.provenance import build_provenance
from harness.results import ResultsStore, RunResult
from harness.runner import Runner, RunPlanItem, RunSummary, build_run_id
from harness.tasks.generator import generate
from harness.tasks.suite import SuiteSpec, build_suite
from harness.trace import JsonlTraceSink, TraceBus
from harness.types import PAYLOAD_VERSION, RunProvenance, TaskSpec

DEFAULT_RESULTS_PATH = Path("results/runs.db")
DEFAULT_TRACE_DIR = Path("results/traces")
DEFAULT_SANDBOX_ROOT = Path(".sandbox")


@dataclass(frozen=True)
class RunPaths:
    results: Path
    traces: Path
    sandbox: Path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    paths = RunPaths(results=args.out, traces=args.traces, sandbox=args.sandbox)
    summary = run_config(config, paths)
    _report(summary, config)
    return 1 if summary.failed else 0


def run_config(
    config: HarnessConfig,
    paths: RunPaths,
    llm: LLMClient | None = None,
) -> RunSummary:
    """Execute every task in the config's suite, recording one row each.

    `llm` is injectable so the whole path is testable without a network. Left out, it is
    built from the config and the key named by `llm.api_key_env`.
    """
    specs = build_suite(_suite_spec(config))
    client = llm if llm is not None else build_llm(config)
    store = ResultsStore(paths.results)
    try:
        plan = [_plan_item(config, spec) for spec in specs]
        by_id = {item.run_id: spec for item, spec in zip(plan, specs, strict=True)}
        return Runner(store).run_all(
            plan,
            lambda item: _execute(config, paths, client, item, by_id[item.run_id]),
        )
    finally:
        store.close()


def _execute(
    config: HarnessConfig,
    paths: RunPaths,
    llm: LLMClient,
    item: RunPlanItem,
    spec: TaskSpec,
) -> RunResult:
    provenance = build_provenance(item.run_id, config)
    sink = JsonlTraceSink(paths.traces / f"{_safe_name(item.run_id)}.jsonl")
    try:
        bus = TraceBus(item.run_id, sink)
        bus.emit_header(provenance)
        built = build_loop(config, bus, llm, sandbox_root=paths.sandbox)
        outcome = built.loop.run(generate(spec))
    finally:
        sink.close()
    return _row(provenance, item, built, outcome)


def _row(
    provenance: RunProvenance,
    item: RunPlanItem,
    built: BuiltRun,
    outcome: LoopOutcome,
) -> RunResult:
    """Every identity column is taken from the header the trace already carries, so a row
    and its trace can never disagree about which run produced them."""
    return RunResult(
        run_id=item.run_id,
        task_id=item.task_id,
        condition=item.condition,
        token_budget=built.token_budget,
        seed=item.seed,
        success=outcome.verdict.success,
        score=outcome.verdict.score,
        tokens_read=outcome.tokens_read,
        tokens_write=outcome.tokens_write,
        steps=outcome.steps,
        wall_ms=outcome.wall_ms,
        config_hash=provenance.config_hash,
        git_commit=provenance.git_commit,
        payload_version=PAYLOAD_VERSION,
        model_id=provenance.model_id,
        model_revision=provenance.model_revision,
        embedding_model=provenance.embedding_model,
        embedding_revision=provenance.embedding_revision,
    )


def _plan_item(config: HarnessConfig, spec: TaskSpec) -> RunPlanItem:
    budget = int(config.context.model_dump().get("token_budget", 0))
    return RunPlanItem(
        run_id=build_run_id(
            config.run.run_id_prefix, config.memory.type, spec.task_id, spec.seed, budget
        ),
        task_id=spec.task_id,
        condition=config.memory.type,
        token_budget=budget,
        seed=spec.seed,
    )


def _suite_spec(config: HarnessConfig) -> SuiteSpec:
    return SuiteSpec(
        count=config.suite.count,
        seed=config.suite.seed,
        chain_length=config.suite.chain_length,
        recall_distance=config.suite.recall_distance,
        distractor_density=config.suite.distractor_density,
        horizon=config.suite.horizon,
    )


def _safe_name(run_id: str) -> str:
    """Run ids carry colons, which are a path separator on some filesystems."""
    return run_id.replace(":", "_").replace("/", "_")


def _report(summary: RunSummary, config: HarnessConfig) -> None:
    print(
        f"condition={config.memory.type} executed={len(summary.executed)} "
        f"skipped={len(summary.skipped)} failed={len(summary.failed)}"
    )
    for run_id, reason in summary.failed:
        print(f"  FAILED {run_id}: {reason}", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness", description="Run a harness config over its task suite."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one config over its task suite")
    run.add_argument("config", type=Path, help="path to a harness config YAML")
    run.add_argument("--out", type=Path, default=DEFAULT_RESULTS_PATH)
    run.add_argument("--traces", type=Path, default=DEFAULT_TRACE_DIR)
    run.add_argument("--sandbox", type=Path, default=DEFAULT_SANDBOX_ROOT)
    return parser


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m harness`
    raise SystemExit(main())
