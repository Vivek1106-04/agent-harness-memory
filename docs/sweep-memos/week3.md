# Week 3 memo - the checkpoint that decides everything

**Status: all four questions answered 30 September, on the sampled rerun (temperature 1.0,
two seeds). The pilot results and what they do and do not show are in the last section.**
Written the day the instruments landed, so the questions and the method are
recorded before any data is seen. Answers are filled in below, the same day the bracket
ran, and not carried in anyone's head.

---

## What is in place

| Deliverable | State |
|---|---|
| Task generator v1 (`harness.tasks.generator`) | Done - `multihop_qa`, pure function of `TaskSpec` |
| Per-step gold labels | Done - `GeneratedTask.gold_labels`, indexed by the canonical trajectory |
| Checker | Done - `CheckerEvaluator` re-derives the answer from the spec |
| Forces-recall assertion (`harness.tasks.forces_recall`) | Done - static gate plus empirical NoOp gate |
| `NoOpMemory` / `OracleMemory` | Done - floor and ceiling of the bracket |
| `BudgetedContextManager` | Done - budget binds, evictions named in `dropped` |
| Runtime loop (`harness.loop`) | Done - one task end to end through all five layers |
| Bracket cells | `configs/week3_bracket_noop.yaml`, `configs/week3_bracket_oracle.yaml` |
| `ReActPlanner` (`harness.components.planner_react`) | Done - pulled forward from week 4 |
| Model client (`harness.llm_http`) | Done - OpenAI-compatible, covers hosted and local vLLM |
| Task suite (`harness.tasks.suite`) | Done - a task set is a pure function of the config |
| Runner CLI (`harness.cli`) | Done - `python -m harness run <config>`, resumable |

The bracket mechanism is verified offline in `tests/test_loop.py`: the same task, budget and
planner fail under `NoOpMemory` and pass under `OracleMemory`. That proves the instrument
moves. It says nothing about whether a **model** can use it, which is what the gate asks.

---

## What blocked the gate, and what is left

Both blockers recorded here on 14 September are now cleared in code:

1. ~~**No looping planner.**~~ `ReActPlanner` was pulled forward from week 4 rather than
   deferring the gate. The scheduling defect it exposed - a week-3 gate depending on a
   week-4 deliverable - stands as a finding about PLAN.md, not something to quietly fix.
2. ~~**No model endpoint.**~~ `HttpLLMClient` speaks the OpenAI chat-completions shape, so
   a hosted endpoint and a local vLLM server are the same code path. The key is read from
   the environment variable named by `llm.api_key_env` and the client refuses to start
   without it.

**What is left is not code.** Two settings, and they are the gate:

- `GEMINI_API_KEY` (or whichever provider is selected) exported in the shell.
- `llm.model` and `llm.revision` pinned in both bracket configs, to a model with
  **>= 128k context**. They currently read `PIN_ME`, which is wrong rather than absent, so
  a run against an unpinned config is recorded as a run against a model named `PIN_ME` and
  is obvious in the results rather than silent.

Then, in order:

```
uv run python -m harness run configs/week3_bracket_noop.yaml
uv run python -m harness run configs/week3_bracket_oracle.yaml
```

The floor cell doubles as the forces-recall screen: a task the NoOp cell **solves** did not
need memory and is struck from the set before the ceiling cell is read.

Neither cleared blocker touches the four questions below. They stand exactly as written on
14 September, before any data was seen.

---

## Finding before any paid call (27 September)

An offline run of the shipped suite (30 tasks, horizon 20, recall_distance 12), driven by
the scripted perfect reader in `tests/fakes.py` rather than a model:

| token_budget | NoOp | Episodic | Graph | Oracle |
|---|---|---|---|---|
| 8000 | 30/30 | 30/30 | 30/30 | 30/30 |
| 400 | 30/30 | 0/30 | 30/30 | 30/30 |
| 300 | 0/30 | 0/30 | 30/30 | 30/30 |

At 8k every document fits in the window, so NoOp solves everything and the bracket is
zero by construction. Running the bracket as configured would have spent the quota to learn
this. Question 1 answered in advance for that geometry: **narrow**, and the decision rule
above says redesign the tasks now.

**Redesign taken: longer documents.** Every document is padded with inert filler to
`suite.document_tokens`. Same suite, 8k budget, same scripted reader (vector and hybrid use
the hashing test encoder, not Qwen3, so treat those two rows as indicative only):

| document_tokens | NoOp | Episodic | Vector* | Graph | Hybrid* | Oracle |
|---|---|---|---|---|---|---|
| 400 | 30/30 | - | - | - | - | 30/30 |
| 550 | 30/30 | - | - | - | - | 30/30 |
| 700 | 0/30 | 0/30 | 8/30 | 30/30 | 30/30 | 30/30 |
| **900 (shipped)** | **0/30** | 0/30 | 6/30 | 30/30 | 30/30 | **30/30** |

Re-run on the shipped pilot suite (suite-digest task ids) with the **real** pinned
Qwen3-Embedding-0.6B for vector and hybrid, same scripted reader, 8k:

| NoOp | Episodic | Vector | Graph | Hybrid | Oracle |
|---|---|---|---|---|---|
| 0/30 | 0/30 | 10/30 | 30/30 | 30/30 | 30/30 |
| recall 0.00 | 0.42 | 0.46 | 1.00 | 0.98 | 1.00 |

The bracket is full width with a perfect reader and the conditions separate. Whether a
real model reaches the ceiling is still question 2, and still needs the paid run. Graph's
lead here is partly the entity-extractor advantage in DECISIONS.md, 2026-09-27.

---

## The four questions

### 1. Is the bracket wide enough to detect a difference?

**Method.** Run the full task set under both cells. Bracket width = Oracle success rate
minus NoOp success rate, with a cluster bootstrap CI over tasks.

**Decision rule, declared now.** Wide -> proceed to week 4. Narrow -> perfect memory barely
beats none, no implementation can beat that spread, and the tasks are redesigned in week 3
rather than discovered in week 10. Narrow means the CI on the difference does not clear the
MDE from question 4.

**Answer (30 September, sampled rerun, 30 tasks x 2 seeds): wide. Proceed.**

At `temperature: 1.0` with `run.seed` sent to the endpoint (DECISIONS.md, 2026-09-29):
NoOp 0/60, Oracle 56/60. Bracket width **93.3pp, 95% cluster bootstrap CI [86.7, 98.3]**
over tasks, seeds averaged within a task. The temperature-0 run below gave the same width.

**First run (29 September, temperature 0, n=30, seed 1): wide.**

Run on Vertex AI, `google/gemini-3.5-flash-lite`, 2k budget, 220-token documents,
`max_steps: 25`, one seed. Suite `suite714605e5`, 30 tasks.

| Cell | Solved | Mean steps | Cost |
|---|---|---|---|
| NoOp (floor) | 0/30 | 24.4 | $0.66 |
| Oracle (ceiling) | 28/30 | 20.8 | $0.48 |

Bracket width **93.3pp, 95% cluster bootstrap CI [83.3, 100.0]** (2000 resamples over
tasks, the bootstrap seed in `harness.analysis`). The lower bound clears the naive 20pp
MDE by a factor of four. The MDE from question 4 is still owed, but no plausible value
for it reaches 83pp.

**How the floor fails.** The n=1 run predicted "read, lost the fact, ran out of steps".
At n=30 the floor is 0/30 by three routes:

| NoOp outcome | Runs |
|---|---|
| Ran out of steps (25) | 24 |
| Planner terminated after 3 consecutive unparseable replies | 5 |
| Answered, wrong | 1 |

NoOp produced 138 unparseable replies across 29 of 30 runs (Oracle: 10 across 4 runs).
Once the fact it needs has been evicted, the model does not only forget - its output
degrades, down to fragments such as `"hormone\nThought: ..."`. Both routes are downstream
of the missing fact, so the floor is a memory effect, but part of it is model breakdown
rather than clean exhaustion, and a reader of the floor number should know that.

NoOp rows mostly carry `score: 1.0` with `success: false`. That is by design: `score` is
the fraction of required facts ever surfaced. NoOp read every fact and lost it, which is
exactly the pressure the task geometry is built to apply.

### 2. Is Oracle high?

**Method.** Oracle success rate alone, by task family and by chain length.

**Why it is asked separately.** A low ceiling means the model cannot use information even
when handed it perfectly - the tasks are too hard or badly posed. That is a different
problem with a different fix, and it is invisible without the ceiling.

**Answer (30 September, sampled rerun): high - 56/60 (93%).** 2 runs answered wrongly with
the facts in hand; 2 stopped without answering (planner terminated on unparseable replies).
The temperature-0 run gave 28/30.

**First run (29 September, temperature 0): high - 28/30 (93%).** One family (`multihop_qa`) and one
chain length (2) are in the suite, so the by-family and by-length breakdown has one cell.

Both misses (`suite714605e5-009`, `-020`) reached an `answer` action at step 15 and 17 with
every required fact surfaced, and gave the wrong value. That is the model composing the
answer wrongly with the facts in hand, not a retrieval or harness failure.

### 3. Can the chosen model sustain 35-step tool use?

**Method.** Oracle cell at `max_steps: 35`, counting runs that reach a verdict without
malformed output, tool-call collapse, or repetition loops.

**Hard constraint.** The model must have **>= 128k context**. On a 32k model, Experiment 2's
"uncapped" cell collapses into its 32k cell and the sweep loses its top end.

**Candidates, probed 2026-09-27** (one two-fact ReAct prompt each, not yet the 35-step run):

| Model | Context | Result | Note |
|---|---|---|---|
| gemini-2.5-flash / -flash-lite | 1M | 404 | closed to new keys |
| gemini-3.1-flash-lite | 1M | wrong | read a document instead of using facts already given |
| **gemini-3.5-flash-lite** | 1M | correct | no hidden thinking at `minimal`; **selected** |
| gemini-3.6-flash | 1M | correct | thinks by default; price doubles 1 Jan 2027 |

**Real runs, one task each, 27 September** (full chain in DECISIONS.md, 2026-09-27): at 8k
with 900-token documents, flash-lite lost its place and even Oracle failed; 3.6/3.8-flash are
capped at 20 requests a day on the free tier. At **2k with 220-token documents** -
unchanged recall pressure, a quarter of the text - Oracle read all nineteen documents in
order and answered correctly in 21 steps, and NoOp read the same, went back for the fact it
had lost, and ran out of steps.

**Answer (provisional, n=1):** gemini-3.5-flash-lite sustains the task at the 2k scale and
not at 8k. The n=30 bracket confirms or refutes it.

**Answer (29 September, n=30): confirmed at 25 steps, not tested at 35.** Oracle reached a
verdict in 30 of 30 runs, 24 of them in exactly 21 steps (the canonical trajectory). 26 runs
had no unparseable reply at all; the other 4 had 10 between them and recovered. No run
collapsed or looped.

**Sampled rerun (30 September).** At temperature 1.0 Oracle still reached a verdict in 58
of 60 runs, mostly in 21-23 steps, but sampling costs format discipline: 128 unparseable
replies across 38 of 60 runs, against 10 across 4 of 30 at temperature 0. Two runs hit the
three-in-a-row limit and ended without an answer. The model sustains the task when
sampled, with a visibly higher malformed-reply rate that the planner absorbs.

**Deviation from the declared method.** The method above says `max_steps: 35`. The bracket
ran at 25, set deliberately in the configs so a run cannot re-read the whole corpus and
dodge the recall requirement (21 canonical steps plus 4 slack). The model is therefore
shown to sustain 21-25-step tool use; a 35-step claim is not made.

**Provider.** Run on Vertex AI rather than AI Studio, so the education credits pay for it
(PR #6). Same model, same prompt shape; the model id reads
`google/gemini-3.5-flash-lite` because Vertex names publisher models that way.

### 4. What is the real intra-task correlation?

**Method.** From the measured per-task variance across seeds in both cells, estimate the
intra-task correlation, re-derive the MDE for the paired design, and set the equivalence
margin from it.

**Declared in advance.** A naive bound at n=80 gives approximately 20pp. We state what the
design actually supports rather than claiming a margin the precision does not earn, and the
margin is fixed here before any comparison is run.

**Answer (30 September): about zero. Seeds behave as independent replicates.**

| Condition | Seed 1 | Seed 2 | Tasks that flip between seeds | Intra-task correlation |
|---|---|---|---|---|
| noop | 0/30 | 0/30 | 0 | undefined (no variance) |
| episodic | 0/30 | 0/30 | 0 | undefined (no variance) |
| graph | 28/30 | 28/30 | 4 | -0.05 |
| hybrid | 26/30 | 30/30 | 4 | -0.05 |
| oracle | 27/30 | 29/30 | 4 | -0.05 |
| vector | 30/30 | 29/30 | 1 | 0.00 |

No task failed under both seeds in any condition. The few failures are sampling noise, not
hard tasks, so a second seed adds close to a full replicate (design effect about 1). The
estimate rests on 2-4 failures per condition and is imprecise; it is a ceiling-regime
number and will not carry over to harder tasks, where failures should cluster on tasks.

**MDE and equivalence margin.** At this success rate the paired graph-minus-vector
difference has a 95% CI half-width of about 7pp (n=30 tasks x 2 seeds), so the design
detects differences of roughly 10pp at 80% power here - far better than the naive 20pp,
but only because success is near ceiling, where variance is small. **We do not fix an
equivalence margin from this pilot.** The margin was to be fixed before any comparison, and
the pilot's primary comparison has now been seen; a margin chosen after it would not be
pre-declared. It is fixed for Experiment 1 on the harder task set, before that sweep runs.

**What made this answerable.** `run.seed` originally reached only the `run_id` and the `seed` column. It is not sent to the endpoint
and no component reads it (checked 29 September). At `temperature: 0.0` a second seed
therefore differed from the first only by provider nondeterminism. The team agreed to sample
at temperature 1.0 with the seed sent (DECISIONS.md, 2026-09-29); the temperature-0 run is
kept in `results/runs.db` as a deterministic baseline and agrees with sampled seed 1 on 174
of 180 cells.

---

## Forces-recall rejection rate

The rejection rate is itself a result. A family that mostly fails the gate is a family whose
design does not depend on memory.

| Family | Candidates | Rejected (static) | Rejected (NoOp solved) | Accepted |
|---|---|---|---|---|
| multihop_qa | 30 | 0 | 0 | 30 |

Static column: every task in the suite shares `recall_distance: 12`, and about 7 padded
documents fit in the 2k window, so `static_gate` (distance > NoOp window) admits all 30.
This was checked by hand from the suite config, not by running `filter_specs` - the suite
builder does not call it. The empirical gate (NoOp solves it) rejected none of the 30, so
the geometry forces recall on every task at this budget. The sampled rerun agrees: NoOp
solved 0 of 60 (30 tasks x 2 seeds).

---

## Pilot results (30 September)

Six conditions x 30 tasks x 2 seeds, temperature 1.0, 2k budget, Vertex AI
`google/gemini-3.5-flash-lite`. 360 runs, $6.85. Full table with token counts:
[pilot-table.md](pilot-table.md). Preliminary - no significance claimed.

| Condition | Success [95% CI] | Ret. R (step mean) | Ret. R (answer step) |
|---|---|---|---|
| noop | 0.00 [0.00, 0.00] | 0.00 | 0.00 |
| episodic | 0.00 [0.00, 0.00] | 0.04 | 0.06 |
| graph | 0.93 [0.87, 0.98] | 0.07 | 0.93 |
| hybrid | 0.93 [0.87, 0.98] | 0.96 | 1.00 |
| oracle | 0.93 [0.87, 0.98] | 1.00 | 1.00 |
| vector | 0.98 [0.95, 1.00] | 0.97 | 0.61 |

**Primary comparison, graph - vector (paired): -0.05 [-0.12, 0.02].** No detectable
difference. The temperature-0 run (seed 1) gives the same ranking and -0.07 [-0.17, 0.00].

**What it shows.**

- Memory is the difference between 0% and 93%: the instrument moves, and it moves because
  of the layer under test.
- Conditions separate. Episodic fails completely; graph, hybrid and vector are near
  ceiling. Stable across two seeds and across temperature 0 and 1.0.

**What it does not show, and why.**

- **Ceiling effect.** Graph, hybrid and vector all reach Oracle; vector exceeds it. A
  two-fact chain is solved by any memory that can return one old fact, so this task set
  cannot rank the three. Experiment 1 needs harder tasks - longer chains, denser
  distractors - to separate them. This is the main result for planning.
- **Oracle is not a strict ceiling.** Its misses are the model answering wrongly with every
  fact in hand, or ending on malformed replies. Headroom above 1.0 (vector 1.05) is that
  noise, not memory better than perfect.
- **Episodic at 0% is unexplained.** Identical to NoOp. It may be the expected result for
  a recency-ordered store when the needed fact is old, or a defect. It is checked before it
  is presented as a finding.
- **Neither recall column measures recovery of the evicted fact.** The step mean charges
  graph for every step before the linking document is read (0.07, DECISIONS.md
  2026-09-30). The answer-step figure counts facts still in the window: vector returns the
  evicted fact but not the one just read, and scores 0.61 while solving 58/60. Success is
  the headline; recall is reported with this caveat or not at all.
- **Scope.** One task family, one chain length, one model, one budget, 30 tasks.
