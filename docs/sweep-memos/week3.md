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

The bracket mechanism is verified offline in `tests/test_loop.py`: the same task, budget and
planner fail under `NoOpMemory` and pass under `OracleMemory`. That proves the instrument
moves. It says nothing about whether a **model** can use it, which is what the gate asks.

---

## What blocks the gate

1. **No looping planner.** `SingleStepPlanner` answers in one call and cannot read a
   document. The bracket needs `ReActPlanner`, scheduled for week 4 (Ansh). The week-3 gate
   therefore depends on a week-4 deliverable - a real scheduling defect in PLAN.md, flagged
   rather than worked around.
2. **No model endpoint.** No key is configured and no local vLLM is running. Question 3 is a
   model selection gate, so it cannot be answered by a scripted client: the answer *is* the
   model's behaviour.

Neither blocker touches the four questions below. They stand as written.

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

**Answer:** PENDING - candidate models and their context lengths to be recorded here.

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
