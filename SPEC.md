# SPEC — the harness contract

Version 1.0. Frozen target date: **23 September 2026**.

This document is the specification four people build against in parallel. Once the
interfaces are frozen, changing anything in §2 or §3 requires an issue, agreement from all
four layer owners, and a version bump.

---

## 1. Stack

| Concern | Choice | Why this and not the alternative |
|---|---|---|
| Language | **Python 3.12** | The memory, embedding and graph ecosystem is Python, and external contributors will be Python people |
| Packaging | **uv** | Single binary replacing pip/pyenv/virtualenv; universal lockfile is the reproducibility story |
| Payloads | **frozen dataclasses** | Immutable by construction, cheap, no runtime validation cost on the hot path |
| Config | **YAML + pydantic v2** | Configs must be frozen, diffable and hash-stamped into every trace. Not Hydra — runtime composition defeats that |
| Vector search | **numpy, exact cosine** | Under 100k vectors brute force is sub-millisecond. **ANN indexes approximate, and approximation is a confound** — we would be comparing representations through different index error rates |
| Graph store | **networkx** | A run's graph is hundreds of nodes. Pure Python, deterministic, zero setup. Not Kùzu — archived Oct 2025, community forks only |
| Embeddings | **Qwen3-Embedding-0.6B**, pinned by revision hash | Apache-2.0, local, free, 70.7 MTEB-eng-v2. MiniLM-L6-v2 for the fast dev loop |
| Results | **SQLite (stdlib)** | Transactional, so checkpoint/resume is free. No service to run |
| Traces | **JSONL**, one file per run | Append-only, greppable, streams without loading |
| LLM access | thin `LLMClient` interface, two adapters | Local vLLM and OpenAI-compatible hosted. ~60 lines beats a framework dependency |
| Registration | `importlib.metadata` entry points | How a third party registers a component without forking |
| Testing | **pytest** | Contract suites are abstract base test classes |
| Lint / format / types | **ruff**, **mypy** strict on `src/harness/` | One tool for lint and format; no hand-formatting debates |

**Embedding model note.** MemDelta shows that swapping only the embedding model shifts
accuracy by 6.2pp — larger than the effect we are trying to detect. Pin the **revision
hash**, not the model name, and record it in every trace.

---

## 2. Payload types — the real contract

Interface breakage in agent systems is data-shape drift, not signature drift. These are
the frozen, versioned contract. See `src/harness/types.py`.

```python
PAYLOAD_VERSION = "1.0"

@dataclass(frozen=True)
class Observation:
    step: int
    content: str
    tool_name: str | None
    error: str | None
    metadata: dict[str, Any]

@dataclass(frozen=True)
class MemoryQuery:
    text: str | None = None
    k: int = 5
    time_range: tuple[int, int] | None = None
    entities: list[str] | None = None
    hops: int = 1
    filters: dict[str, Any] | None = None

@dataclass(frozen=True)
class MemoryItem:
    content: str
    score: float
    provenance: str          # originating trace id + step
    kind: str                # episodic | semantic | procedural | failure

@dataclass(frozen=True)
class Action:
    tool_name: str
    arguments: dict[str, Any]
    rationale: str | None

@dataclass(frozen=True)
class ContextWindow:
    segments: list[tuple[str, str]]   # (role, text)
    token_count: int
    dropped: list[str]                # what the budget evicted, and why

@dataclass(frozen=True)
class Verdict:
    success: bool
    score: float
    per_step: list[bool]
    reason: str
```

**Two fields are load-bearing and must not be treated as optional.**

`MemoryQuery` is deliberately wide. A narrow `retrieve(query: str, k: int)` suits vector
memory and handicaps graph memory, which needs entity anchors and hop depth, and episodic
memory, which needs time windows. Narrowing it means benchmarking crippled versions of
those representations, which invalidates the entire comparison.

`ContextWindow.dropped` records what the budget evicted and why. Without it, an
Experiment 2 result cannot distinguish *"memory failed to recover the fact"* from
*"the fact was never evicted in the first place."*

**Compatibility rule.** Within a major version, fields may be **added with a default**.
Fields are never removed, renamed, or retyped. External components depend on this.

---

## 3. Interfaces

Five abstract interfaces, in `src/harness/interfaces.py`. Every implementation declares
what it honours rather than pretending to support everything.

```python
class MemoryProvider(ABC):
    capabilities: frozenset[str]   # subset of {"temporal", "graph", "semantic"}

    @abstractmethod
    def write(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]: ...

    @abstractmethod
    def reset(self) -> None: ...
```

`reset()` is load-bearing. Without it, state leaks between runs and we lose a week to
phantom results.

```python
class Planner(ABC):
    @abstractmethod
    def next_action(self, window: ContextWindow) -> Action | None: ...   # None = terminate

class ContextManager(ABC):
    token_budget: int

    @abstractmethod
    def assemble(self, goal: str, history: list[Observation],
                 retrieved: list[MemoryItem]) -> ContextWindow: ...

class ToolRuntime(ABC):
    @abstractmethod
    def execute(self, action: Action, step: int) -> Observation: ...     # errors → Observation

class Evaluator(ABC):
    @abstractmethod
    def judge(self, task: TaskSpec, trace: list[Observation]) -> Verdict: ...
```

**Implementations ignore query fields they do not support; they must not crash on them.**
A vector memory receiving `hops=3` ignores it. A vector memory that raises is inadmissible.

---

## 4. Conformance suites — admission, not hygiene

Every interface ships an abstract contract test. An implementation subclasses it, sets a
factory, and inherits the tests. **This is the admission criterion**: a component — ours or
a stranger's — is admissible when it passes the suite for its layer, and not before.

| Layer | Suite | Implementations that must pass |
|---|---|---|
| Memory | `MemoryProviderContract` | 6 |
| Context manager | `ContextManagerContract` | 2 |
| Planner | `PlannerContract` | 2 |
| Tool runtime | `ToolRuntimeContract` | 2 |
| Evaluator | `EvaluatorContract` | 2 |

Minimum required tests per suite:

```python
class MemoryProviderContract:
    def test_write_then_retrieve_returns_item(self): ...
    def test_reset_clears_all_state(self): ...
    def test_retrieve_on_empty_returns_empty_list(self): ...
    def test_unsupported_query_fields_do_not_crash(self): ...
    def test_retrieve_never_exceeds_k(self): ...
    def test_results_are_deterministic_for_identical_input(self): ...

class ContextManagerContract:
    def test_never_exceeds_declared_budget(self): ...
    def test_dropped_list_accounts_for_every_evicted_segment(self): ...
    def test_system_instructions_and_goal_are_never_evicted(self): ...

class PlannerContract:
    def test_emits_valid_action_or_terminates(self): ...
    def test_respects_max_steps(self): ...
    def test_malformed_model_output_does_not_crash(self): ...

class ToolRuntimeContract:
    def test_unknown_tool_returns_error_observation_not_exception(self): ...
    def test_observation_step_is_monotonic(self): ...
    def test_path_outside_sandbox_is_refused(self): ...

class EvaluatorContract:
    def test_verdict_is_deterministic_for_identical_traces(self): ...
```

`ContextManagerContract.test_never_exceeds_declared_budget` deserves special note. The
entire context-budget argument and the whole of Experiment 2 rest on the budget actually
binding. That is an engineering guarantee on a layer we do not sweep, and the study is
worthless without it.

---

## 5. Registration without forking

A component defined outside this repository must be selectable by name in config. Nothing
in the harness may name an implementation class.

```python
from harness.registry import register

@register("memory", name="my_memory")
class MyMemory(MemoryProvider):
    capabilities = frozenset({"semantic"})
```

```yaml
memory: {type: my_memory, backend: whatever_they_want}
```

Third-party packages register through entry points in their own `pyproject.toml`:

```toml
[project.entry-points."harness.memory"]
my_memory = "my_package.memory:MyMemory"
```

⚠️ Loading third-party components executes third-party code. See [SECURITY.md](SECURITY.md) §7.

**Week 4–7 deliverable:** one memory provider written in a separate package, against the
published guide alone, by someone who did not read the harness internals. A claim of
extensibility nobody has exercised from outside is untested code.

---

## 6. Trace bus and results

Components do not log as they please. The runtime owns an emitter; there is one event type;
the evaluator and results store read only from that stream.

```python
@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    step: int
    layer: str                # planner | context | memory | tools | evaluator | runtime
    kind: str                 # action | observation | retrieval | eviction | verdict | error
    payload: dict[str, Any]
    token_counts: dict[str, int]
    wall_ms: float
```

**Every run records, in its trace header:** config hash, git commit, payload version,
model identifier and revision, embedding model revision, and every seed. A result whose
provenance cannot be reconstructed is not a result.

Results go to SQLite: one row per run, with `run_id`, condition, budget, task id, seed,
verdict, token counts split **read path and write path**, step count, wall time. Runner is
resumable — losing 400 completed runs to a crash at hour four is unacceptable.

---

## 7. Config

One YAML file fully describes a run. Swapping a component is never an import change.

```yaml
run:
  seed: 1
  max_steps: 35
  run_id_prefix: exp1

planner:   {type: react,    max_steps: 35}
memory:    {type: graph,    backend: networkx, hops: 2}
context:   {type: budgeted, token_budget: 8000}
tools:     {type: local,    sandbox_root: ./.sandbox}
evaluator: {type: checker}

embedding:
  model: Qwen/Qwen3-Embedding-0.6B
  revision: <pinned-sha>          # frozen across all conditions

llm:
  provider: local                 # local | openai_compatible
  base_url: ${HARNESS_LLM_BASE_URL}
  model: <pinned-model-id>
  revision: <pinned-sha>
  max_output_tokens: 300          # cost lever — see PLAN.md
  temperature: 0.0

limits:
  max_tokens_per_run: 500_000
  max_spend_usd: 25.0             # global kill-switch
```

Config is validated by pydantic on load, hashed, and the hash is written into every trace.
Unknown keys are an error, not a warning — a typo'd component name must fail loudly rather
than silently fall back to a default.

---

## 8. Frozen controls

Declared constants, pinned in config, reported in the methodology:

| Control | Value |
|---|---|
| Embedding model + revision | one, identical across Vector / Episodic / Graph / Hybrid |
| Chunking strategy | identical where applicable |
| Language model + revision | one pinned checkpoint |
| Planner, tool runtime, evaluator | one implementation each during the study |
| Tuning budget | fixed hours per memory implementation, declared in advance, actual hours logged |

Token cost is reported for the **read path and write path separately**. A memory system
that wins on read while costing 50× on write has not won.

---

## 9. Task environment

Tasks are generated from a specification, not hand-written.

```python
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str                # multihop_qa | mock_fs | repo_edit
    fact_dependency_graph: dict[str, list[str]]
    recall_distance: int
    distractor_density: float
    horizon: int
    entity_pool: list[str]
    seed: int
```

The generator emits **three artefacts from one source**: the task, its checker, and its
per-step gold labels. This is what makes `OracleMemory` and retrieval precision/recall
possible at all.

**The forces-recall assertion is a test, not an assumption.** Every generated task is run
under `NoOpMemory` with only the last *k* observations available and **must fail**. Tasks
that pass are rejected from the set automatically. Without this check, a task solvable from
recent context contributes pure noise and looks identical to a task where memory genuinely
did not matter.

---

## 10. Metrics

| Metric | Purpose |
|---|---|
| Task success rate | Primary outcome |
| **Headroom recovered** = (condition − NoOp) / (Oracle − NoOp) | Scale-free, comparable across task sets and models |
| Tokens per task, read and write paths separate | Cost |
| Retrieval precision / recall | Separates memory quality from agent quality |
| Steps to completion | Efficiency |
| Recovery rate after error | Reliability |
| Wall-clock latency | Practical cost |

---

## 11. Decision rule — declared before any data is seen

Written here so it cannot be chosen after seeing results.

- **Unit of analysis is the task**, not the run. Every task runs under all conditions with
  the same seeds — a paired design that blocks on task difficulty.
- **Inference:** cluster bootstrap resampling *tasks*, giving a CI on the paired difference
  in success rate. Cross-checked with mixed-effects logistic regression (condition fixed,
  random intercept per task). Report effect size with CI, never a bare p-value.
- **Multiple comparisons:** one primary comparison declared in advance — graph vs vector.
  Everything else is secondary, with Holm correction.
- **The rule:** A beats B for family F at budget C when the paired difference interval
  excludes zero after correction, the difference exceeds the MDE re-derived at week 3, and
  it holds in the same direction on the confirmation model.
- **"No difference" must also be earned.** If the interval lies entirely inside a
  pre-declared equivalence margin, we report no practical difference. ⚠️ The margin is set
  at week 3 from measured intra-task correlation. A naive bound at n=80 gives ≈20pp, so we
  cannot claim equivalence at 5pp — the margin follows the design's real precision.
- **Claims are scoped, never global.** Output is a table of family × budget → winner or
  "no detectable difference", not a leaderboard.
- **Cost is part of "best."** Report success against total tokens and identify the Pareto
  frontier. If one wins on success and loses on cost, say so rather than picking.

**External calibration.** Before trusting the internal comparison, run the four
implementations once on a public benchmark (LongMemEval-S) to show they are competent
versions of the published methods rather than strawmen. This anchors us to the literature
at the implementation level — we compare methods under frozen controls, never our numbers
against someone else's published numbers.

---

## 12. Out of scope

Named so the boundary is deliberate:

- Adaptive or learned component selection
- `MemoryProvider`s wrapping other `MemoryProvider`s
- Scheduling and recovery as layers
- Planner and context-strategy variants as *swept axes* — both implementations are built
  and tested, they are simply not run as experimental conditions
- LLM-as-judge evaluation
