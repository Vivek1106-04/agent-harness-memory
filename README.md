# Agent Harness — memory representation under context constraint

A component-substitutable LLM agent harness, and a controlled study run on it.

**Research question.** How much does the choice of memory representation affect a
long-horizon LLM agent's task success and token cost, and how does that effect depend
on the context budget the agent is given?

**Premise.** No single agent configuration is best for every task, so whoever runs the
harness must be able to swap any component. That is only worth building if the choice
actually matters — which is what the study measures.

---

## The two deliverables

| | What | Assessed by |
|---|---|---|
| **A — Engineering** | Five-layer harness where every layer meets the same bar: frozen payload contract, abstract interface, ≥2 implementations, conformance suite, config-registry entry, trace emission, named owner | Its own conformance suites — independent of B's outcome |
| **B — Research** | Experiment 1 (does representation matter?) and Experiment 2 (at what context budget does it stop mattering?) | The measured result, including a null |

---

## Where to start

| You are | Read |
|---|---|
| New to the repo | this file, then [PLAN.md](PLAN.md) for your week |
| Writing code | [SPEC.md](SPEC.md) — the interfaces are the contract |
| Using an AI coding agent | [AGENTS.md](AGENTS.md) |
| Opening a PR | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Touching the tool runtime, config, or plugin loading | [SECURITY.md](SECURITY.md) — required, not optional |
| Wondering why something is the way it is | [DECISIONS.md](DECISIONS.md) |

---

## Quickstart

```bash
# one-time
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras

# run the tests — this is the bar for every layer
uv run pytest

# lint, format, type-check (CI runs all three)
uv run ruff check .
uv run ruff format --check .
uv run mypy src/

# run one task end to end
uv run python -m harness.cli run --config configs/smoke.yaml
```

You need an LLM endpoint. Copy `.env.example` to `.env`, fill it in, and load it:

```
HARNESS_LLM_BASE_URL=http://localhost:8000/v1   # local vLLM — the default
GEMINI_API_KEY=...                               # the variable llm.api_key_env names

set -a; source .env; set +a                      # the harness reads the environment only
```

`.env` is gitignored. **Never commit a key.** See [SECURITY.md](SECURITY.md).

---

## Layer ownership

Every layer has exactly one owner, accountable for all seven conditions on it.

| Layer | Implementations | Owner |
|---|---|---|
| Spine (types, ABCs, runtime loop, trace bus, registry) | — | Ansh |
| Planner | `ReActPlanner` · `SingleStepPlanner` | Ansh |
| Context manager | `BudgetedContextManager` · `FullContextManager` | Muni |
| Memory A | `VectorMemory` · `EpisodicMemory` | Ansh + Vivek |
| Memory B | `GraphMemory` · `HybridMemory` | Muni + Jugal |
| Memory instruments | `NoOpMemory` · `OracleMemory` | Vivek |
| Environment | task generator, gold labels, runner, results store, stats | Vivek |
| Tool runtime | `LocalToolRuntime` · `MockToolRuntime` | Vivek |
| Evaluator | `CheckerEvaluator` · `AlwaysFailEvaluator` | Jugal |

Memory implementations are built in **pairs who review each other's code**. That is the
cheapest available control on the effort confound — the most serious threat in the study.

---

## Status

Week 2 spine in place: payload types, five interfaces, registry with entry-point discovery,
all five conformance suites, one trivial implementation each for planner, context, tools and
evaluator, config schema with hashing, trace bus with redaction, SQLite results store and a
resumable runner.

Week 3 instruments in place: task generator v1 (`multihop_qa`, a pure function of the spec,
emitting the task, its checker and its per-step gold labels), the forces-recall assertion,
`NoOpMemory` and `OracleMemory`, `BudgetedContextManager`, the task-document tool runtime,
a minimal `CheckerEvaluator`, and the runtime loop that runs one task through all five
layers.

Week 4 in place: `ReActPlanner` (pulled forward, since the week-3 gate needed it),
`HttpLLMClient` for hosted and local vLLM endpoints alike, `LocalToolRuntime` against a
real sandboxed filesystem, the first two memory providers under test (`VectorMemory` with
exact cosine, `EpisodicMemory` with time-windowed retrieval), task suites as a pure
function of the config, and a resumable CLI. 383 tests, 100% coverage on `src/harness/`.

Running a cell:

```bash
export GEMINI_API_KEY=...                       # never in a config file
uv run python -m harness run configs/week3_bracket_noop.yaml
uv run python -m harness run configs/week3_bracket_oracle.yaml
```

The week-3 gate is **not yet answered**: it now needs only a key and a model pinned in
`llm.model` / `llm.revision` (>= 128k context). Questions, method and decision rules were
recorded in [docs/sweep-memos/week3.md](docs/sweep-memos/week3.md) before any data was
seen, and the answers go in the same file the day the bracket runs.

Week 5 in place: `GraphMemory` (networkx, entity anchors, hop-bounded traversal) and
`HybridMemory` (cosine + graph fused by reciprocal rank), both passing the memory contract,
so all six conditions are registered. Retrieval precision/recall is scored per run against
the gold facts. `python -m harness sweep` runs several configs under several seeds and
`python -m harness report` produces the pilot table: success and headroom recovered with
cluster-bootstrap CIs over tasks, forces-recall screen applied, graph-vs-vector paired.

**Open before the pilot:** at `token_budget: 8000` the shipped suite does not force recall
(see DECISIONS.md, 2026-09-27). The task geometry or the pilot budget has to change first.

Not yet implemented: the `mock_fs` / `repo_edit` task families (after the panel).

`PLAN.md` is the schedule; the next hard gate is **code freeze for the panel**.
