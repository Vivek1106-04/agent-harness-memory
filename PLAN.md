# PLAN — week by week, layer by layer

**Today:** 10 September 2026 · **Next panel review:** 26–28 October 2026 · **Thesis deadline:** 16 weeks

Two horizons in this file. §A is the run to the October panel (6½ weeks). §B is the rest of
the thesis. Do not read §B until §A is delivered.

---

## What we are showing on 26 October

**We are not showing Experiment 1.** Its 1,440 runs sit in weeks 8–12 and no honest
compression gets us there. Claiming otherwise and missing is worse than scoping honestly.

We show four things instead:

1. **The bracket, measured.** Oracle scores X, NoOp scores Y, therefore our task set has Z
   points of headroom and *can* detect a difference. This is a real result and it is the
   direct answer to the panel's question about scope.
2. **A live component swap.** Change one config line — `memory: {type: vector}` →
   `{type: graph}` — rerun, show a different trace. Sixty seconds, and substitutability
   stops being a claim.
3. **CI green:** five contract suites, every implementation passing. Substitutability
   tested, not asserted.
4. **A pilot sweep** — 30 tasks × 2 seeds × 6 conditions — reported with error bars and the
   words *"preliminary, n=30, no significance claimed; Experiment 1 runs weeks 8–12."*

Plus the revised cost model (§C). Catching a provider reprice is the clearest possible
evidence that we are actually running this.

---

# §A — The run to 26 October

## Week 1 · 10–16 Sep · Foundations

**Gate: none. Everything else depends on this week not slipping.**

| Owner | Deliverable | Done when |
|---|---|---|
| Ansh | `types.py` — all payload dataclasses, `PAYLOAD_VERSION = "1.0"` | `mypy` clean, frozen, no mutable defaults |
| Ansh | `interfaces.py` — five ABCs with declared capabilities | Imports nothing from implementations |
| Ansh | `registry.py` — `@register` + entry-point discovery | A dummy component registers and resolves by name |
| Vivek | Repo hygiene: `pyproject.toml`, `uv.lock`, ruff/mypy config, CI workflow | CI green on an empty test suite |
| Vivek | `LLMClient` interface + local vLLM adapter + OpenAI-compatible adapter | One real completion from each path |
| Muni | Read MemGym (2605.20833) and MemDelta (2606.29914) **in full**, write both into `docs/citation-ledger.md` with quoted sentences | Ledger entry per claim we make about them |
| Jugal | Read MemoryAgentBench (2507.05257) and Evaluating Memory Structure (2602.11243) in full, same treatment | Same |
| **All four** | Agree the topic-selection story in one paragraph | Written in `docs/` — four people cannot give four answers |

**Two things that must happen in week 1 or they cannot happen at all:**

- **Ansh + Vivek: email the supervisor about institute GPU access.** A decent GPU makes the
  full sweep 20–40 hours and marginal cost zero. This is the single highest-value email of
  the project.
- **Vivek: apply for academic compute credits** — Google Cloud research credits, AWS Cloud
  Credit for Research, Azure for Students, NVIDIA academic, GitHub Student Pack. These take
  weeks to process. Applying late is the same as not applying.

Also get free development keys: Gemini (1,500 req/day, best default), Groq (fast iteration),
Cerebras (1M tokens/day batch). Free tiers **cannot** carry the sweep — 157 days at best —
but they are sized exactly right for weeks 1–7 debugging, where early runs surface bugs
rather than results.

---

## Week 2 · 17–23 Sep · Interfaces frozen

### 🚩 GATE — interfaces frozen 23 September

After this date, changing anything in SPEC.md §2 or §3 requires an issue, all four owners
agreeing, and a version bump. The spine blocks everyone; this deadline is hard.

| Owner | Deliverable | Done when |
|---|---|---|
| Ansh | `PlannerContract` + `SingleStepPlanner` | Contract green |
| Muni | `ContextManagerContract` + `FullContextManager` | Contract green, budget test passing |
| Vivek | `ToolRuntimeContract` + `MockToolRuntime` | Contract green, sandbox test passing |
| Jugal | `EvaluatorContract` + `AlwaysFailEvaluator` | Contract green |
| Ansh | `MemoryProviderContract` + config registry + pydantic config schema | A config typo fails loudly |
| Ansh | Trace bus + `TraceEvent` + trace redactor | No layer can log outside the bus |
| Vivek | SQLite results store + resumable runner skeleton | Kill it mid-run, restart, no duplicate rows |

**Why the trivial second implementation of every layer, in week 2, before the real one.**
An interface satisfied by one class is a guess; one satisfied by two is a contract. Each
costs about an hour and each doubles as a free ablation baseline.

---

## Week 3 · 24–30 Sep · The checkpoint that decides everything

### 🚩 GATE — bracket check · model selection · throughput verification

This is the highest-value week in the schedule. We run **NoOp and Oracle before building
any real memory**, and answer four questions in two days.

| Owner | Deliverable |
|---|---|
| Vivek | Task generator v1 — emits task, checker, per-step gold labels from one `TaskSpec` |
| Vivek | Forces-recall assertion running automatically; tasks that pass under NoOp are rejected |
| Vivek | `NoOpMemory` + `OracleMemory` |
| Ansh | Minimal runtime loop — enough to run a task end to end |
| Muni | `BudgetedContextManager` with the budget guarantee actually binding |

**The four questions:**

1. **Is the bracket wide enough to detect a difference?** Wide → proceed. Narrow → perfect
   memory barely beats none, no implementation can beat that spread, and we redesign the
   tasks now, in week 3, rather than discovering it in week 10.
2. **Is Oracle high?** Low Oracle means the model cannot use information even when handed
   it perfectly — the tasks are too hard or badly posed. Different problem, different fix,
   and invisible without the ceiling.
3. **Can the chosen model sustain 35-step tool use?** Week 3 is now a **model selection
   gate**, not only a task gate. ⚠️ Whatever model we pick must have **≥128k context** —
   on a 32k model, Experiment 2's "uncapped" cell collapses into its 32k cell and the sweep
   loses its top end.
4. **What is the real intra-task correlation?** Re-derive the MDE from measured variance and
   set the equivalence margin from it. A naive bound at n=80 gives ≈20pp; we state what the
   design actually supports.

Write the answers into `docs/sweep-memos/week3.md` the same day. Do not carry them in
someone's head.

---

## Week 4 · 1–7 Oct · The loop runs for real

| Owner | Deliverable | Done when |
|---|---|---|
| Ansh | `ReActPlanner` | Contract green; malformed model output does not crash it |
| Muni | `BudgetedContextManager` hardened; eviction recorded in `dropped` | Every evicted segment accounted for |
| Vivek | `LocalToolRuntime` — sandboxed, path-checked, timeouts | SECURITY.md §5 checklist satisfied |
| Jugal | `CheckerEvaluator` — deterministic scoring against gold labels | Same trace ⇒ same verdict, always |
| Ansh | Full end-to-end run with trace + verdict + results row | One task, start to finish, reproducible from config hash |

**End of week 4 the harness is real.** Everything after this is memory implementations and
measurement.

---

## Week 5 · 8–14 Oct · Memory A

| Owner | Deliverable |
|---|---|
| Ansh + Vivek | `VectorMemory` — numpy exact cosine, pinned embedding revision |
| Ansh + Vivek | `EpisodicMemory` — time-windowed retrieval, honours `time_range` |
| Muni + Jugal | `GraphMemory` started — networkx, honours `entities` and `hops` |
| Vivek | Retrieval precision/recall wired to gold labels |
| All | **Tuning-hours log started.** Declared budget per implementation, actual hours recorded |

The tuning log is not bureaucracy. The effort confound is the most serious threat in this
study, and the log plus paired peer review is the only mitigation we have.

**Pairs review each other's memory code.** Ansh and Vivek review one another; Muni and
Jugal review one another. A single owner working alone on an implementation is exactly
where the effort confound starts.

---

## Week 6 · 15–21 Oct · Memory B and the pilot

### 🚩 GATE — code freeze for the panel, 21 October

| Owner | Deliverable |
|---|---|
| Muni + Jugal | `GraphMemory` + `HybridMemory` green against contract |
| Vivek | **Pilot sweep**: 30 tasks × 2 seeds × 6 conditions = 360 runs (~12,600 calls) |
| Vivek | Analysis: headroom recovered per condition, CIs from cluster bootstrap |
| Ansh | External component demo — one provider in a separate package, registered without forking |
| All | Design-decisions log written up: where each interface bent against reality |

If Graph and Hybrid slip, **run the pilot with four conditions instead of six.** The pilot
is the should-have; the bracket and the harness are the must-have. Do not let a slipping
implementation take the demo down with it.

---

## Week 7 · 22–25 Oct · Rehearsal

| Owner | Deliverable |
|---|---|
| All | Slides: bracket chart, CI screenshot, pilot table with error bars, revised cost model |
| Vivek | **Rehearse the live config swap on the actual machine you will present from.** Twice |
| All | Q&A drill — every question in the talking-points doc, out loud |
| Ansh | Repo tidy: README status updated, no stale TODOs, CI green on `main` |

**Panel: 26–28 October.**

---

# §B — After the panel

| Weeks | Work | Gate |
|---|---|---|
| 8–10 | Experiment 1 proper: 6 conditions × 80 tasks × 3 seeds = 1,440 runs. Debugging why numbers look strange | — |
| 11–12 | Clean Experiment 1 results, external calibration run on LongMemEval-S | **GO / NO-GO** |
| 13–15 | Experiment 2 budget sweep (4k/8k/16k/32k/uncapped, 1,920 runs) + cross-model confirmation | — |
| 16 | Write-up | — |

**Week 12 go/no-go.** Trustworthy Experiment 1 numbers → proceed to Experiment 2 and
confirmation runs. Otherwise → ship the harness plus Experiment 1 as an ablation study and
stop. That is a complete project, and having a defined stopping condition is what keeps the
scope honest.

---

# §C — Cost, revised

DeepSeek repriced on 16 August 2026 and now bills peak/off-peak. The proposal's numbers are
stale.

| | Proposal assumed | Actual, off-peak | Peak |
|---|---|---|---|
| Input | $0.435/M | **$0.66/M** | $1.32/M |
| Output | $0.87/M | **$1.98/M** | $3.96/M |
| Cached input | $0.004/M | **$0.022/M** | $0.044/M |

Against 1.88B input / 59M output:

| Scenario | Off-peak | Peak |
|---|---|---|
| Cached | **~$158** | ~$316 |
| Uncached | ~$1,358 | ~$2,715 |

The $59 estimate becomes ~$158 — still inside the $300 approved, but the margin is gone.
**Output is not cacheable**, and $117 of that total is irreducible output cost.

Three levers, in order of preference:

1. **Run off-peak only.** Peak is 01:00–04:00 and 06:00–10:00 UTC = **06:30–09:30 and
   11:30–15:30 IST**. Simply not launching sweeps in those windows halves the bill.
2. **Cap output at 300 tokens/step**, not 500. A ReAct rationale plus an action JSON fits.
   Takes output cost from $117 to ~$69.
3. Reduce seeds. Last resort — it costs statistical power, which we have little of.

**Preferred path: local weights on institute GPU, free API tiers for development, $300 held
as the fallback for the final clean sweep.** That is also a stronger reproducibility story
than a hosted endpoint, since the proposal already requires pinned open weights.

---

# §D — Standing rules

**Weekly.** Every Monday, each owner posts in the team channel: what landed, what is
blocked, what they need from whom. Every Friday, one person mails the supervisor a
five-line summary. The last review went badly because she heard things late; that does not
happen twice.

**Sweep memos.** After every sweep, one page in `docs/sweep-memos/`: what we expected, what
happened, what surprised us. Ten of these by the end and the findings section writes itself
out of the surprises instead of being reconstructed from memory.

**Design-decisions log.** Every place an implementation did not fit an interface, what
changed, what it cost. All five interfaces, not memory alone. Where the abstraction bends
is a reportable result about modular agent design.

**Citation ledger.** No citation enters any document until someone on the team has opened
the paper and pasted the sentence that supports the claim. This file becomes the literature
review, so the cost is recovered.
