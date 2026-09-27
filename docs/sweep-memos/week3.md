# Week 3 memo - the checkpoint that decides everything

**Status: harness ready, gate not yet answered.** Written the day the instruments landed, so
the questions and the method are recorded before any data is seen. Answers get filled in
below, the same day the bracket runs, and not carried in anyone's head.

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

**Answer:** PENDING

### 2. Is Oracle high?

**Method.** Oracle success rate alone, by task family and by chain length.

**Why it is asked separately.** A low ceiling means the model cannot use information even
when handed it perfectly - the tasks are too hard or badly posed. That is a different
problem with a different fix, and it is invisible without the ceiling.

**Answer:** PENDING

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

### 4. What is the real intra-task correlation?

**Method.** From the measured per-task variance across seeds in both cells, estimate the
intra-task correlation, re-derive the MDE for the paired design, and set the equivalence
margin from it.

**Declared in advance.** A naive bound at n=80 gives approximately 20pp. We state what the
design actually supports rather than claiming a margin the precision does not earn, and the
margin is fixed here before any comparison is run.

**Answer:** PENDING

---

## Forces-recall rejection rate

The rejection rate is itself a result. A family that mostly fails the gate is a family whose
design does not depend on memory.

| Family | Candidates | Rejected (static) | Rejected (NoOp solved) | Accepted |
|---|---|---|---|---|
| multihop_qa | PENDING | PENDING | PENDING | PENDING |
