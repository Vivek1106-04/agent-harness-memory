# Design decisions log

Every place an implementation did not fit an interface: what changed, what it cost.

This covers **all five interfaces**, not memory alone. There is no established answer to
what a memory contract should look like when it must serve vector, graph and episodic
backends without handicapping any of them, and we will hit that problem around week 5
regardless. Where the abstraction bends is a reportable result about modular agent design —
a finding independent of whether the study returns a positive result or a null.

Maintained from day one. Reconstructing this later does not work.

## Format

```
### YYYY-MM-DD · <layer> · <one-line summary>
**Owner:** <name>
**What did not fit:** ...
**What we changed:** ...
**What it cost:** <hours, and what we gave up>
**Reportable?** yes/no — why
```

---

## Decisions taken before implementation began

### 2026-09-10 · stack · Python 3.12 + uv
**Owner:** Vivek
**Why:** The memory, embedding and graph ecosystem is Python, and external contributors —
the point of §3.8 substitutability — will be Python people. uv for the universal lockfile,
which is the reproducibility story.
**Reportable?** no — ordinary engineering choice.

### 2026-09-10 · memory · exact cosine search, no ANN index
**Owner:** Ansh + Vivek
**What did not fit:** the obvious choice is FAISS or Chroma.
**What we changed:** brute-force numpy cosine instead.
**Why:** a run holds hundreds of vectors, where brute force is sub-millisecond — but the
real reason is validity. **ANN indexes approximate, and different implementations would
approximate differently.** We would be comparing representations through varying index
error rates, which is precisely the class of confound MemDelta documents.
**Reportable?** yes — a measurement-integrity argument for avoiding an obvious optimisation.

### 2026-09-10 · memory · networkx, not an embedded graph database
**Owner:** Muni + Jugal
**Why:** a run's graph is hundreds of nodes; pure Python is deterministic and needs no
setup. Kùzu was the natural candidate but was archived in October 2025 after Apple acquired
the company, leaving community forks only — not a dependency for a 16-week project.
**Reportable?** no.

### 2026-09-10 · spine · MemoryQuery is wide rather than narrow
**Owner:** Ansh
**What did not fit:** `retrieve(text, k)` is the clean minimal interface, and it is wrong.
**What we changed:** the query carries `entities`, `hops`, `time_range` and `filters`.
Implementations ignore what they do not support and declare capabilities.
**Why:** a narrow query suits vector memory and handicaps graph and episodic memory, which
would then have to smuggle structure through a string. We would be benchmarking crippled
versions of them, invalidating the comparison.
**What it cost:** a wider contract surface, and a contract test asserting that unsupported
fields do not crash.
**Reportable?** yes — this is the central interface-design tension of the whole harness.

### 2026-09-10 · evaluator · no LLM-as-judge
**Owner:** Jugal
**Why:** it reintroduces the confound the study exists to remove, and a judge reading
model-generated traces is directly injectable — a trace could argue for its own passing
verdict.
**Reportable?** yes — worth a line in threats to validity.

### 2026-09-10 · models · local pinned weights preferred over a hosted endpoint
**Owner:** Vivek
**What changed:** the proposal assumed DeepSeek V4 Pro via API. DeepSeek repriced on
16 August 2026 (input $0.435→$0.66/M, output $0.87→$1.98/M, cached $0.004→$0.022/M), taking
the cached estimate from ~$59 to ~$158 against $300 approved.
**What we changed:** default to local weights on institute GPU; free API tiers for weeks 1–7
development; the $300 held as fallback for the final clean sweep, run off-peak.
**Constraint this creates:** the chosen model must have **≥128k context**. On a 32k model,
Experiment 2's "uncapped" cell collapses into its 32k cell and the sweep loses its top end.
**Reportable?** yes — reproducibility argument strengthens, and the cost model in the
proposal needs correcting.

---

## Week 2 - spine, config, trace, results

### 2026-09-12 · context · token counting is an approximation until week 4
**Owner:** Vivek
**What did not fit:** the budget must bind deterministically on every machine, and a real
tokenizer means a model download, a revision to pin, and network in unit tests.
**What we changed:** `harness/tokens.py` counts ceil(chars / 4). A context manager takes the
counter as a constructor argument, so `LLMClient.count_tokens` is substituted for the real
runs without touching the eviction logic.
**What it cost:** the budget is approximate during weeks 2-3. Numbers reported from a sweep
must use the model counter, not this one.
**Reportable?** no - implementation detail, but it is a measurement caveat worth stating.

### 2026-09-12 · context · FullContextManager declares an unbounded budget
**Owner:** Vivek
**What did not fit:** the contract asks that a manager never exceed its declared budget. A
"keep everything" manager handed an 8k budget would violate that on every call.
**What we changed:** it declares `token_budget = sys.maxsize` and ignores a requested
budget rather than accepting one it will not honour. It is the "uncapped" cell of
Experiment 2, so this is the honest reading rather than a workaround.
**Reportable?** yes - the alternative (silently ignoring a declared budget) is exactly the
defect that would corrupt the budget sweep.

### 2026-09-12 · config · component blocks accept extra keys, everything else forbids them
**Owner:** Vivek
**What did not fit:** SPEC section 7 says unknown keys are an error, but component blocks
carry implementation-specific keys (`hops`, `backend`, `k`) that the harness cannot know.
**What we changed:** every fixed block (`run`, `llm`, `embedding`, `limits`) forbids extras.
Component blocks allow them and pass them to the named component, which refuses an argument
it does not accept. A typo still fails loudly, one layer later.
**Reportable?** no.

### 2026-09-12 · config · the config hash excludes `llm.base_url`
**Owner:** Vivek
**What did not fit:** hashing the whole config gives the same experiment two identities when
run against two endpoints, which breaks resume and splits results across machines.
**What we changed:** `base_url` is excluded from the hash. Model identity is carried by
`llm.model` and `llm.revision`, which are hashed.
**Reportable?** yes - provenance argument.

### 2026-09-12 · runtime · redaction happens at emit time, in one place
**Owner:** Vivek
**What we changed:** `TraceBus.emit` redacts every payload before the sink sees it, using the
same detector that refuses credentials found in config files. Hex runs are exempt, because
config hashes and commit ids are hex and are recorded on purpose.
**Reportable?** no - SECURITY.md section 1 made this a requirement, not a choice.

### 2026-09-12 · runner · run_id is the idempotency key and is derived, not generated
**Owner:** Vivek
**What we changed:** `build_run_id(prefix, condition, task_id, seed, token_budget)` is
deterministic, and the results table makes it the primary key with `INSERT OR IGNORE`. A
restart after a crash skips completed cells; a retry cannot write a second row.
**What it cost:** a cell that raises is deliberately left unrecorded so a rerun retries it,
which means the summary, not the database, is where failures are visible.
**Reportable?** no.

### 2026-09-12 · repo · ASCII only in source, and Python pinned to 3.12
**Owner:** Vivek
**What we changed:** source files carry ASCII characters only (AGENTS.md code style), and
`.python-version` pins 3.12 so that the interpreter matches SPEC section 1 rather than
whatever uv finds on a contributor's machine.
**Reportable?** no.

### 2026-09-14 · memory + tools · `prepare(TaskSpec)` on two interfaces, defaulting to no-op
**Owner:** Vivek
**What did not fit:** `OracleMemory` is the ceiling of the bracket, and a ceiling has to know
what perfect retrieval means. `TaskDocumentRuntime` has to know which documents exist. Both
are per-run facts, so neither fits a constructor argument set from a static config, and
threading a task object through the registry would have put a run-specific object into
component construction.
**What we changed:** `MemoryProvider.prepare(task)` and `ToolRuntime.prepare(task)`, both
defaulting to no-op, called once per run by the loop after `reset`. Generation is a pure
function of the spec, so each component re-derives what it needs without a side channel.
**What it cost:** an interface addition eight days before the freeze, and a rule that has to
be enforced by review: a memory implementation *under test* that reads gold labels in
`prepare` is cheating by construction. Baselines only.
**Reportable?** yes - "where does the task environment enter a layered agent" is exactly the
kind of seam this study is meant to document.

### 2026-09-14 · memory · the contract suite gained a `stores_writes` flag
**Owner:** Vivek
**What did not fit:** `MemoryProviderContract.test_write_then_retrieve_returns_item` cannot
hold for `NoOpMemory`, which stores nothing by design and is not a memory that merely
performs badly.
**What we changed:** the suite declares `stores_writes = True` by default; `NoOpMemory` sets
it False and the same test then asserts retrieval stays empty. Every other test in the suite
applies unchanged to both.
**What it cost:** the contract is now slightly weaker as a specification - a lazy
implementation could claim the exemption. Review catches that; the alternative was either an
inadmissible baseline or a contract nobody can satisfy honestly.
**Reportable?** yes - the trivial implementation revealed the contract was a guess, which is
the argument for building two implementations per layer in the first place.

### 2026-09-14 · tasks · the generator derives document geometry from the horizon
**Owner:** Vivek
**What did not fit:** `recall_distance`, `distractor_density` and `horizon` can contradict
each other - a graph too large for the horizon, or a density that leaves no room.
**What we changed:** documents = `horizon - 1`; required facts occupy a contiguous block
ending `recall_distance` documents before the answer; everything else is a distractor, and
`distractor_density` acts as a *floor* on the distractor fraction. Any contradiction raises
`TaskGenerationError` rather than generating a degraded task.
**What it cost:** the achieved density is usually above the requested one. A degraded task
would have been worse: it produces plausible-looking wrong numbers.
**Reportable?** no.

### 2026-09-14 · memory · an unprepared `OracleMemory` falls back to total recall
**Owner:** Vivek
**What did not fit:** with no task bound, the oracle has no notion of gold and would return
nothing - failing its own contract suite, which knows nothing about tasks.
**What we changed:** unprepared, it returns the most recently written items, newest first.
Still an upper bound on any real memory, and in a run it is always prepared.
**Reportable?** no.

### 2026-09-14 · context · a budget too small for the instructions raises
**Owner:** Muni
**What did not fit:** system instructions and the goal are never evictable, so a budget
smaller than the two of them cannot be honoured.
**What we changed:** `BudgetTooSmallError` at assembly time. Truncating the instructions
would quietly run a different experiment than the one written down.
**Reportable?** no.

### 2026-09-14 · evaluator · `CheckerEvaluator` landed a week early, in minimal form
**Owner:** Vivek (Jugal owns the layer and hardens it in week 4)
**What did not fit:** the forces-recall assertion needs a verdict to reject a task on, and
`AlwaysFailEvaluator` rejects everything. Week 3 could not be built without a real checker.
**What we changed:** a minimal checker for `multihop_qa` only: digit-run match against the
answer the generator resolved, partial credit by required facts surfaced. Week 4 extends it
to the remaining families.
**Reportable?** no - schedule note, recorded so the week-4 owner inherits it deliberately.

---

## Week 4 - the model endpoint and the first two memories under test

### 2026-09-20 · planner · `ReActPlanner` pulled forward into the week-3 gate
**Owner:** Ansh (implemented early by Vivek; Ansh owns the layer)
**What did not fit:** the week-3 gate needs a planner that can read a document across many
steps, and `SingleStepPlanner` answers in one call. The gate therefore depended on a
week-4 deliverable.
**What we changed:** built it now rather than deferring the gate by a week. The scheduling
defect stays recorded in `docs/sweep-memos/week3.md` rather than being silently absorbed.
**Reportable?** no - schedule note.

### 2026-09-20 · planner · an unparseable reply becomes an action, not a stop
**Owner:** Vivek
**What did not fit:** `Planner.next_action` returns an `Action` or `None`, and `None`
terminates the run. A single malformed reply would therefore throw away an otherwise good
run, which at 1,440 runs is a measurable loss of data and money.
**What we changed:** an unparseable reply becomes `Action(tool_name="invalid_action")`. No
runtime implements that tool, so the tool layer returns its ordinary unknown-tool error,
the error enters the history, and the model reads its own mistake on the next step. A cap
of `max_parse_failures` consecutive failures still terminates, so a model stuck in a
formatting loop cannot spend all 35 steps failing the same way.
**Reportable?** yes - "how a modular harness should absorb malformed model output" is a
finding about the planner interface, not an implementation detail.

### 2026-09-20 · context · the action protocol lives in the system text, not the planner
**Owner:** Muni (context layer) / Vivek
**What did not fit:** a ReAct planner has to tell the model how to write an action, but a
planner that appended its own instructions after assembly would push the window past
`token_budget` and the Experiment 2 guarantee would stop being true.
**What we changed:** `ACTION_PROTOCOL` is part of `prompts.SYSTEM_INSTRUCTIONS`, so the
context manager counts it against the budget like any other protected segment. The cost is
that `SingleStepPlanner` now sees protocol text it does not use; it is a baseline, and the
checker matches digit runs, so a JSON-wrapped answer still scores.
**Reportable?** yes - where the layer boundary bent, and why the budget won.

### 2026-09-20 · llm · one adapter for both providers, and an approximate tokenizer
**Owner:** Vivek
**What did not fit:** PLAN.md week 1 asked for a local vLLM adapter *and* an
OpenAI-compatible adapter. Both speak the same chat-completions shape, so two adapters
would have been two things to keep in step for no measured difference.
**What we changed:** one `HttpLLMClient`; `provider` selects a base URL and whether a key
is sent. `count_tokens` stays the arithmetic approximation rather than the provider's
tokenizer - a provider-specific tokenizer would make the 8k cell mean one thing on Gemini
and another on a local model, and Experiment 2 would be comparing two different budgets.
**Reportable?** yes - the budget is a harness property, not a model property, and this is
where that is enforced.

### 2026-09-20 · memory · `EpisodicMemory` reads its clock from `provenance`
**Owner:** Muni / Vivek
**What did not fit:** the layer honours `MemoryQuery.time_range`, but `MemoryItem` has no
timestamp field and `types.py` is frozen. This is the open question recorded at the week-2
freeze, now answered by an implementation rather than by speculation.
**What we changed:** the step is parsed from the `"<run_id>:<step>"` provenance string the
runtime already writes, falling back to write order. No payload change. Wall-clock time is
deliberately not used: the evaluator must return the same verdict for the same trace in a
different timezone.
**Reportable?** yes - a wide `MemoryQuery` met a narrow `MemoryItem`, and the asymmetry is
a real finding about payload design for substitutable memory.

### 2026-09-20 · build · components are wired by parameter name, not by type
**Owner:** Vivek
**What did not fit:** components need things a YAML file cannot hold - an HTTP client, an
encoder, a sandbox path. Resolving that with `if config.memory == "vector"` would have
made the substitutability claim false (AGENTS.md rule 4).
**What we changed:** the builder offers a fixed set of services and each component
receives the ones it names in its constructor signature. Config values win over services.
The rule is generic, so a third-party component installed through an entry point is wired
by it with no change to harness source.
**Reportable?** yes - this is the substitutability claim surviving contact with dependency
injection, which is exactly the kind of place the claim usually dies.

### 2026-09-20 · tools · `LocalToolRuntime` refuses non-regular files instead of timing out
**Owner:** Vivek
**What did not fit:** PLAN.md asks for timeouts. For filesystem reads the real hazard is a
FIFO under the sandbox root, which blocks forever - and no timeout on the loop can
interrupt a blocking read that has already started.
**What we changed:** only regular files are opened, and reads stop at `max_read_bytes`. The
`timeout_s` key was removed from `configs/exp1_vector_8k.yaml` rather than left as a
setting that does nothing. A timeout returns when a tool can actually block, which is when
a shell tool lands - and no shell tool is in scope.
**Reportable?** no.

---

## Week 5 - memory B, retrieval scoring, pilot tooling

### 2026-09-27 · memory · graph entities are extracted lexically, and only identifier-shaped
**Owner:** Vivek
**What did not fit:** `GraphMemory` needs entity anchors, and the runtime loop queries with
the goal text alone - `MemoryQuery.entities` is never filled by any caller.
**What we changed:** anchors come from `query.entities` when given, else from the query
text, via a deterministic regex that recognises identifier-shaped names (`site_003_01`).
An LLM extractor was rejected: model tokens on the write path and nondeterminism in the
index would both land inside the measured number.
**What it cost:** the extractor matches the surface form `tasks.suite` gives entities.
Plain-word entities (`alpha`) are invisible to it, and it will need replacing for
LongMemEval-S. Graph memory is therefore advantaged by knowing the entity format while the
vector encoder is generic.
**Reportable?** yes - threat to validity; state it next to any graph-vs-vector number.

### 2026-09-27 · memory · the configured `hops` is a floor the query can only raise
**Owner:** Vivek
**What did not fit:** the loop always sends the default `hops=1`, so a query-only depth
would make the config's `hops` a dead setting.
**What we changed:** depth = max(query.hops, configured hops).
**Reportable?** no.

### 2026-09-27 · memory · HybridMemory fuses by reciprocal rank, not a weighted sum
**Owner:** Vivek
**What did not fit:** cosine and hop count are on incomparable scales, and any weight is a
tuning knob spent on one condition only (effort confound). Wrapping VectorMemory and
GraphMemory was out of scope (SPEC section 12).
**What we changed:** hybrid owns both indices, sharing `harness/entity_graph.py` with
`GraphMemory`, and fuses with RRF at the literature constant k=60, untuned.
**Reportable?** yes - methodology.

### 2026-09-27 · runner · replicates were keyed on the task seed and silently collided
**Owner:** Vivek
**What did not fit:** `run_id` used the task seed, which `run.seed` does not change. A
second replicate of a config resolved to the same run ids as the first and was skipped
as already done - a pilot at "2 seeds" would have been 1 seed with no error.
**What we changed:** `run_id` and the `seed` column carry `run.seed`. `harness sweep
--seeds 1 2` overrides it per replicate; the seed is hashed, so provenance records it.
**What it cost:** at `temperature: 0.0`, and with no seed sent to the endpoint, two
replicates differ only by provider nondeterminism. Question 4 of the week-3 memo
(intra-task correlation across seeds) cannot be answered honestly until that is settled.
**Reportable?** yes - the harness caught a measurement bug before any data existed.

### 2026-09-27 · evaluator · retrieval precision and recall are scored by the loop
**Owner:** Vivek
**What we changed:** `RetrievalTally` scores each step once a required fact has been read,
by content containment - the same test `OracleMemory` applies, so the ceiling is 1.0 on
both by construction. Stored per run as nullable columns.
**Reportable?** yes - it separates memory quality from agent quality (SPEC section 10).

### 2026-09-27 · tasks · at 8k, the shipped suite geometry does not force recall
**Owner:** Vivek
**What did not fit:** an offline run with a perfect reader and no memory solves 30/30
tasks at `token_budget: 8000`. Twenty one-sentence documents fit in the window many
times over, so the bracket is zero and the forces-recall screen would strike every task.
The static gate in `tasks/forces_recall.py` would have said so, but no runner calls it.
**What we changed:** documents are padded with inert filler to `suite.document_tokens`
(900 in every bracket and pilot config), the fact buried at a seeded position. Offline at
8k with the perfect reader: NoOp 0/30, Oracle 30/30; NoOp stops solving between 550 and
700, so 900 leaves margin. Padding uses its own RNG stream, so `document_tokens: 0`
reproduces every earlier suite byte for byte, and codes and entities never change.
**Payload change:** `TaskSpec.document_tokens: int = 0` was added and `PAYLOAD_VERSION`
bumped 1.0 -> 1.1. Adding a defaulted field is allowed within a major version, but
AGENTS.md rule 1 asks for an issue and all four owners' agreement. **Pending ratification.**
**What it cost:** filler dilutes a document's embedding but not its entity links, so
padding widens the gap between vector and graph retrieval. Realistic, and it compounds the
extractor caveat above - report them together.
**Reportable?** yes - the week-3 gate doing its job before a single paid call.

### 2026-09-27 · process · all four real memories were written by one author
**Owner:** Vivek
**What did not fit:** PLAN.md pairs memory owners to control the effort confound. Graph
and Hybrid were written in the same session as Vector and Episodic, by the same author.
**What it cost:** the pairing control is gone for this pilot. Paired review of
GraphMemory/HybridMemory by Muni + Jugal before the numbers are reported restores part of it.
**Reportable?** yes - threats to validity.

### 2026-09-27 · models · gemini-3.5-flash-lite, with thinking pinned to minimal
**Owner:** Vivek
**What did not fit:** `gemini-2.5-flash` and `-flash-lite` return 404 to new keys ("no
longer available to new users"). Gemini 3 models cannot turn thinking off, and the API
counts thinking tokens against `max_output_tokens` - with the 300-token cap, unpinned
thinking can consume the whole reply and leave an empty completion, which the planner
would score as malformed output. That would be a model-configuration failure recorded as
a memory result.
**What we changed:** `llm.reasoning_effort` (sent verbatim when set) and pinned
`minimal`. Probed 2026-09-27 on a two-fact prompt: 3.5-flash-lite answered correctly with
no hidden thinking tokens; 3.1-flash-lite ignored facts already in its prompt; 3.6-flash was
correct but thinks by default and doubles in price on 1 Jan 2027. Hosted models have no
revision sha, so `llm.revision` records the stable code's release month.
**Reportable?** yes - model selection is week-3 question 3.

### 2026-09-27 · limits · the spend and token limits are enforced, not declared
**Owner:** Vivek
**What did not fit:** `limits.max_spend_usd` and `limits.max_tokens_per_run` were
validated and hashed, and read by nothing.
**What we changed:** the HTTP client prices every call from the endpoint's `usage` block
(thinking billed as output via `total_tokens`), falls back to the arithmetic count when
usage is absent, and refuses the next call at the limit with `SpendLimitError`. Spend is
seeded from the results store, so a resumed sweep keeps counting; each row records
`cost_usd`. The loop stops a run before the next step once `max_tokens_per_run` is used.
The runner stops the sweep on `SweepAbortedError` and after 3 consecutive failed cells - a
revoked key or a spent daily quota fails every cell, and grinding on records nothing. The
streak spans every config of a sweep (a real run of six one-task configs kept going
because it reset per config). A 429 naming a per-day quota raises `DailyQuotaError` at
once instead of retrying: flash-lite's free tier is 500 requests a day, about 24 runs.
**What it cost:** spend inside a run that later raises is counted in-process but not
recorded, so after a crash-and-resume the meter under-counts by at most one run.
Prices are paid-tier list prices even on the free tier, so the meter overstates there.
**Reportable?** no - operational.

### 2026-09-27 · tasks · task ids carry a digest of the suite
**Owner:** Vivek
**What did not fit:** every suite numbered its tasks `task-000` upward, so two suites in one
results store - a smoke run, a padded and an unpadded suite - would be merged by the report
as the same tasks.
**What we changed:** task ids are `suite<digest>-NNN`, digest over the validated suite
block minus `count` (task N is the same task at any suite size). The report also refuses
two rows for one (condition, budget, task, seed).
**Reportable?** no - but it is the class of bug that produces plausible wrong numbers.

### 2026-09-27 · memory · queries are encoded with the embedder's own query prompt
**Owner:** Vivek
**What did not fit:** Qwen3-Embedding is instruction-tuned and ships a `query` prompt;
encoding queries like documents benchmarks a degraded vector memory.
**What we changed:** `Encoder.encode_queries` applies the model's shipped `query` prompt
(part of the pinned revision, not tuned here). Vector and Hybrid use it for queries only.
**Reportable?** yes - fairness to the baseline.

### 2026-09-27 · context · observations show what the action acted on
**Owner:** Vivek
**What did not fit:** the first real smoke run re-read `doc_00.txt` 25 times and hit the
step cap. The prompt showed `[step 3] read_file: <content>` with no file name, and the agent
sees no other record of its actions - once the listing was evicted it could not tell what
it had read.
**What we changed:** tool runtimes put the path in `Observation.metadata["target"]` and
the shared renderer shows `read_file(doc_03.txt)`. Memory still stores `content` only.
**Reportable?** yes - a harness defect only a real model could expose.

### 2026-09-27 · llm · empty completions are asked again, twice
**Owner:** Vivek
**What did not fit:** gemini-3.5-flash-lite sometimes returns `finish_reason: stop` with
no content and zero output tokens (1 call in 11 in a probe). Each became an invalid action,
and three in a row end a run.
**What we changed:** the client repeats an empty completion up to twice; every attempt is
metered. Identical in every condition.
**Reportable?** no - transport robustness.

### 2026-09-27 · tasks · generator v2: the terminal fact is the last document
**Owner:** Vivek
**What did not fit:** v1 put all required facts in one contiguous block. Reading in order,
the real model met both facts together and answered at step 10 - NoOp succeeded, Oracle
(one sample) did not. `recall_distance` only existed for an agent that kept reading after
it had the answer.
**What we changed:** the fact the goal names is the last document; its supporting facts
end `recall_distance` documents before it. The agent learns what matters only at the end.
`max_steps` tightened 35 -> 25 (19 reads + list + answer = 21, plus 4) so NoOp cannot
re-read the corpus to recover what it forgot.
**What it cost:** an agent that reads out of order can still shorten the gap; the NoOp
screen strikes any task where that happens.
**Reportable?** yes - task design; the week-3 gate answered with a real model.

### 2026-09-27 · memory · observations are written as sentence chunks, by the loop
**Owner:** Vivek
**What did not fit:** a 900-token document stored whole is retrieved whole. Five items took
about 4,500 of 8,000 tokens, and since the budget protects memory ahead of history the
memory conditions kept about three recent steps, lost their place and looped. That
penalised memory for being present - the opposite of the measurement.
**What we changed:** `harness/chunking.py` splits every observation into sentences before
writing, identical for every condition (SPEC section 8 lists chunking as a frozen control).
Retrieved items are about 20 tokens. Memory segments are labelled `[memory from step N]` so
the agent can tell a recollection from a fresh read; Oracle now reuses the provenance of
the read that saw a fact, so its labels match every other condition's.
**Reportable?** yes - context allocation between memory and history is a design result.

### 2026-09-27 · tasks · the terminal fact adds a suffix; the goal names the documents
**Owner:** Vivek
**What did not fit:** in the second real run, graph and episodic answered after reading
only the supporting fact and were scored correct: in a two-fact chain the answer *was* the
supporting fact's code, and `site_000_00` / `site_000_01` differ by one digit, so a guess
landed. Oracle, vector and hybrid instead looped over `doc_00`-`doc_08`: with the listing
evicted, nothing told the agent how many documents there were.
**What we changed:** a derived fact is its dependencies' codes followed by a two-digit
literal of its own, so the answer appears verbatim in no document and a guess fails. The
goal - never evicted - names the range (`19 documents, doc_00.txt to doc_18.txt`).
`GENERATOR_VERSION = 2` is part of every task id, so rows from v1 and v2 never merge.
**Reportable?** yes - a scoring path that rewarded guessing, caught before any data.

### 2026-09-27 · context · a compact action log survives eviction
**Owner:** Vivek
**What did not fit:** with the step targets visible, every condition still looped. When the
budget evicted the oldest observation, it took with it the only record that the action had
happened: the agent saw reads of doc_01..doc_08, concluded doc_00 was unread, re-read it -
which evicted doc_01 - and so on. Memory stores document content, not the action record,
so no condition could escape it.
**What we changed:** both context managers add one `actions` segment listing every step
(`2 read_file(doc_00.txt); 3 ...`, about six tokens a step), admitted after memory and
before observations, counted in the budget and named in `dropped` if it cannot fit. The
context contract now accounts for it. What the agent did is kept; what it saw is what
memory and the budget contend over - which is the thing measured.
**Reportable?** yes - separating action history from observation content is a design result.

### 2026-09-27 · tasks · the pilot runs at 2k with 220-token documents, not 8k with 900
**Owner:** Vivek
**What did not fit:** at 8k with 900-token documents, gemini-3.5-flash-lite lost its place
in about 7k tokens of filler and wandered back even as Oracle - a ceiling that fails
measures the model, not memory. gemini-3.6-flash and 3.8-flash are capped at 20 requests
a day on the free tier, so they cannot carry a pilot without billing.
**What we changed:** the same geometry at a quarter of the scale - about seven documents
fit either way, so recall pressure is unchanged. Real run, one task: Oracle read doc_00 to
doc_18 in order and answered correctly in 21 steps; NoOp read the same, learned at doc_18
what it needed, went back, and ran out of steps. About $0.015 a run.
**What it cost:** the pilot says nothing about 8k. Experiment 2's budget sweep needs either
documents scaled with the budget or a stronger model on a paid tier.
**Reportable?** yes - week-3 question 3 (model capability) answered with evidence.

### 2026-09-29 · llm · replicates are sampled: temperature 1.0, run seed sent to the endpoint
**Owner:** Vivek, agreed by the team
**What did not fit:** `run.seed` reached only the run id. At `temperature: 0.0`, with no
seed sent, seed 2 would differ from seed 1 only by provider nondeterminism, so week-3
question 4 (intra-task correlation) would have measured Vertex's noise, not replicate
variance. The 2026-09-27 runner entry left this open.
**What we changed:** the client sends `run.seed` as `seed`; the sweep builds one client per
seed (each starts its spend meter at the store total, so the limit still spans the sweep);
the bracket and pilot configs sample at `temperature: 1.0`, Gemini's default. Checked on
Vertex, 29 September: at 1.0 the same seed gave identical output three times out of three
and a different seed gave different output.
**What it cost:** the bracket and pilot are re-run at 1.0 into `results/pilot-t1.db`. The
temperature-0 rows in `runs.db` stay as a deterministic baseline and are not mixed in - run
ids do not carry the config hash, so a shared database would skip every sampled cell as
already done. The week-3 answers must be re-read on the sampled bracket.
**Reportable?** yes - methodology: replicates are real samples, not repeated greedy decodes.

---

## Open questions to resolve by the week-2 freeze

- ~~Does `MemoryItem` need a `timestamp` distinct from the step index in `provenance`?~~
  **Answered 2026-09-20: no.** `EpisodicMemory` parses the step out of `provenance` and
  honours `time_range` without a payload change. Wall-clock ordering was rejected outright
  - it would make the evaluator's verdict depend on the machine's timezone.
- Should `capabilities` be advisory or enforced? Currently advisory — the contract suite
  only asserts unsupported fields do not crash, not that declared capabilities work.
- Does the trace bus need backpressure at 117,600 calls, or is append-only JSONL enough?
