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

---

## Open questions to resolve by the week-2 freeze

- Does `MemoryItem` need a `timestamp` distinct from the step index in `provenance`?
  Episodic memory may want wall-clock ordering; adding a field later is allowed, removing
  one is not, so decide before the freeze.
- Should `capabilities` be advisory or enforced? Currently advisory — the contract suite
  only asserts unsupported fields do not crash, not that declared capabilities work.
- Does the trace bus need backpressure at 117,600 calls, or is append-only JSONL enough?
