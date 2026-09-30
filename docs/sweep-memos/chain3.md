# Chain-3 sweep - 30 September 2026

Six conditions x 30 tasks x 2 seeds, three-fact chains, 2k budget, temperature 1.0 with the
seed sent, Vertex AI `google/gemini-3.5-flash-lite`. 360 runs, $7.22, no failed cells.
Full table: [chain3-table.md](chain3-table.md). Preliminary - no significance claimed.

## What we expected

Declared in DECISIONS.md (2026-09-30, chain-3 entry) before the first chain-3 call, from an
offline sweep with the scripted perfect reader (vector 4/30, graph 30/30, hybrid 30/30,
Oracle 30/30, NoOp and episodic 0/30):

1. graph - vector positive, 95% CI excluding 0;
2. NoOp and episodic at 0;
3. Oracle at or above 80%, or the comparison measures the model rather than memory;
4. graph and hybrid tied within a 10pp equivalence margin.

## What happened

| Condition | Success [95% CI] | Seed 1 | Seed 2 | Dropped-fact recall |
|---|---|---|---|---|
| noop | 0.00 [0.00, 0.00] | 0/30 | 0/30 | 0.00 |
| episodic | 0.00 [0.00, 0.00] | 0/30 | 0/30 | 0.00 |
| vector | 0.03 [0.00, 0.08] | 2/30 | 0/30 | 0.58 |
| graph | 0.65 [0.52, 0.77] | 23/30 | 16/30 | 0.92 |
| hybrid | 0.80 [0.70, 0.90] | 26/30 | 22/30 | 0.97 |
| oracle | 0.82 [0.72, 0.92] | 26/30 | 23/30 | 1.00 |

Paired over tasks, seeds averaged within a task, 2000 cluster-bootstrap resamples:

| Comparison | Difference [95% CI] | Against the declaration |
|---|---|---|
| **graph - vector** (primary) | **+0.62 [0.48, 0.75]** | 1: holds |
| hybrid - graph | +0.15 [-0.02, 0.30] | 4: **not shown** - neither different nor within 10pp |
| oracle - hybrid | +0.02 [-0.12, 0.15] | - |
| oracle - graph | +0.17 [0.00, 0.33] | - |

Declarations 2 and 3 hold: NoOp and episodic 0/60; Oracle 0.82, point estimate above the
80% line, though its CI reaches 0.72.

**The ceiling is broken.** At chain 2 graph, hybrid and vector were tied with Oracle
(graph - vector -0.05 [-0.12, 0.02]). At chain 3 the same three separate by 60 points, and
the separation is the one the offline sweep predicted.

## What surprised us

- **Graph and hybrid fail mostly with the facts in hand.** Of graph's 21 failures, 16 had
  dropped-fact recall 1.0 - memory returned everything the window had lost - and the model
  still did not answer. Hybrid 9 of 12, Oracle 10 of 11. At temperature 1.0 the model emits
  empty or action-less replies ("Thought: I have gathered all ..." with no action); each
  costs a step, and a run has four steps of slack (25 allowed, 21 canonical). The failures
  land on reading the last document or just after it. This lowers the ceiling for every
  condition alike and does not favour any memory, but it is why the real numbers sit below
  the offline 30/30.
- **Graph trails hybrid and Oracle although its dropped-fact recall is 0.92.** Most of the
  gap is the model failure above, and it is larger under graph (16 of 60 runs) than under
  Oracle (10) or hybrid (9). Why the model fails more often when graph fills the memory
  segment - graph also returns the listing chunk and loosely linked items alongside the
  facts - is not established. The hybrid-graph difference is not significant, so this is a
  question, not a finding.
- **Seed-to-seed spread is larger than at chain 2.** Graph 23 then 16, hybrid 26 then 22.
  At chain 2 no task failed under both seeds; harder tasks make single runs less stable,
  which is what question 4 of the week-3 memo predicted for a non-ceiling regime.
- **Vector's retrieval is exactly as the offline run said.** Dropped-fact recall 0.58: it
  returns the root fact by similarity to the goal but, among sixteen look-alike "access code
  for X is N" documents, not the middle link. The live demo shows this on task 000.

## What this means for the panel

- The harness separates memory designs when the task demands it, and the separation was
  predicted offline and declared before the paid run.
- The primary comparison has an answer at chain 3: graph beats vector by 62 points.
- Graph versus hybrid is open: the pilot is not powered to separate them or to call them
  equal at a 10pp margin.
- For Experiment 1: chain 3 is the suite. The malformed-reply step loss should be settled
  before it runs - more step slack or a lower temperature, applied to every condition -
  since it caps every condition's ceiling.
