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
| **A — Engineering** | Five-layer harness where every layer meets the same bar: frozen payload contract, abstract interface, ≥2 implementations, conformance suite, config-registry entry, trace emission | Its own conformance suites — independent of B's outcome |
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
uv run python -m harness run configs/smoke.yaml --out results/smoke.db --traces results/smoke-traces
```

You need an LLM endpoint. Copy `.env.example` to `.env`, fill it in, and load it:

```
HARNESS_LLM_BASE_URL=http://localhost:8000/v1   # local vLLM — the default
GEMINI_API_KEY=...                               # the variable llm.api_key_env names

set -a; source .env; set +a                      # the harness reads the environment only
```

`.env` is gitignored. **Never commit a key.** See [SECURITY.md](SECURITY.md).

---

## Layers

Every layer meets the same bar: frozen payload contract, abstract interface, at least two
implementations, a conformance suite, a registry entry and trace emission.

| Layer | Implementations |
|---|---|
| Spine (types, ABCs, runtime loop, trace bus, registry) | - |
| Planner | `ReActPlanner` · `SingleStepPlanner` |
| Context manager | `BudgetedContextManager` · `FullContextManager` |
| Memory | `VectorMemory` · `EpisodicMemory` · `GraphMemory` · `HybridMemory` |
| Memory instruments | `NoOpMemory` · `OracleMemory` |
| Environment | task generator, gold labels, runner, results store, statistics |
| Tool runtime | `TaskDocumentRuntime` · `LocalToolRuntime` · `MockToolRuntime` |
| Evaluator | `CheckerEvaluator` · `AlwaysFailEvaluator` |
| Outside the harness | `external/harness-keyword-memory`, registered by entry point only |

The repository is one continuous history, from the frozen interfaces through every layer,
experiment and memo; `git log` shows how each piece arrived and why, and DECISIONS.md
records every place the design bent against reality.

---

## Status - 30 September 2026

**Harness complete.** All five layers with at least two implementations each, every one
passing its conformance suite; 567 tests, 100% coverage on `src/harness/`, mypy strict.
Six memory conditions registered. A seventh, `keyword`, lives in a separate package and is
selected by config through an entry point, with no harness source naming it.

**Model.** `gemini-3.5-flash-lite` on Vertex AI (project `btp-vivek`, education credits),
`reasoning_effort: minimal`, temperature 1.0 with the run seed sent, so replicates are real
samples. Qwen3-Embedding-0.6B at sha `97b0c61`. Budget 2k tokens, documents padded to 220
tokens: about seven fit, so a fact read twelve documents earlier has left the window.

**Results.** Every sweep has a memo in [docs/sweep-memos/](docs/sweep-memos/); predictions
were written down before the paid runs that tested them.

| Sweep | Runs | Headline |
|---|---|---|
| Week-3 bracket | 60 | NoOp 0/60, Oracle 56/60: width 93.3pp [86.7, 98.3] |
| Two-fact pilot, 6 conditions x 2 seeds | 360 | Ceiling: graph, hybrid, vector all reach Oracle; graph - vector -0.05 [-0.12, 0.02] |
| Chain-3 suite, 6 conditions x 2 seeds | 360 | Ceiling broken: graph - vector **+0.62 [0.48, 0.75]**; hybrid 0.80, Oracle 0.82 |

Retrieval is reported as dropped-fact recall - did memory return what the window lost -
the one recall figure that tracks success (DECISIONS.md, 2026-09-30). Episodic memory is
NoOp by construction under goal-text queries; that is a finding about the loop, not about
temporal memory.

**Cost.** ~$17.50 for 905 runs, about $0.02 a run, measured per row from the endpoint's
own usage report. Experiment 1 is ~$30; Experiment 2 is estimated at ~$345 and needs a
decision before it runs ([PLAN.md](PLAN.md), section C).

**Demos.** Both rehearsed on the presenting laptop:

```bash
export HARNESS_LLM_BASE_URL=https://aiplatform.googleapis.com/v1/projects/btp-vivek/locations/global/endpoints/openapi
scripts/demo_swap.sh              # one config line, vector -> graph, ~90 s  (--replay offline)
scripts/demo_external.sh --live   # outside package, discovered and run, ~40 s
```

Runbooks: [docs/demo-swap.md](docs/demo-swap.md), [docs/demo-external.md](docs/demo-external.md).

**Reproducing a sweep** (Vertex needs `gcloud auth application-default login` once):

```bash
uv run python -m harness sweep configs/chain3_*_2k.yaml --seeds 1 2 \
    --out results/chain3.db --traces results/chain3-traces
uv run python -m harness report --db results/chain3.db
```

Raw results and traces are not in git. They are archived outside the repository as
`results-backup-<date>.tar.gz` with a SHA-256 manifest.

**Next.** Panel on 21 October. After it, Experiment 1 on the chain-3 suite (80 tasks x 3
seeds), then the week-12 go/no-go, then Experiment 2. Not yet implemented: the `mock_fs` /
`repo_edit` task families.
