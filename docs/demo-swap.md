# Live component swap - runbook

Panel item 2 (PLAN.md): change one config line, rerun, show a different trace. About 90
seconds, two real model runs, roughly $0.04.

## Before the panel

1. On the laptop you will present from, once:
   `gcloud auth application-default login` as the account that owns `btp-vivek`.
2. In the terminal you will present from:

   ```
   export HARNESS_LLM_BASE_URL=https://aiplatform.googleapis.com/v1/projects/btp-vivek/locations/global/endpoints/openapi
   cd project
   ```
3. Rehearse it twice (PLAN.md, week 7): `scripts/demo_swap.sh`. The output should match
   `docs/demo-recording.txt`.

## What to say while it runs

- **The diff.** One line, `type: vector` to `type: graph`. No harness source changes: the
  registry resolves the name from config. That is the substitutability claim, live.
- **The task.** Find the access code for `site_000_02`. Three facts chain to it:
  `doc_05` holds `site_000_00 = 5014`, `doc_06` says `site_000_01` is that followed by `10`,
  `doc_18` says `site_000_02` is that followed by `15`. Answer `50141015`. Only about seven
  documents fit in the 2k window, so by the time `doc_18` is read, `doc_05` and `doc_06`
  are gone: the agent can only answer if memory brings them back.
- **Vector.** On the answer step it returns `doc_05` and four look-alike distractors
  (`doc_07`, `doc_10`, `doc_13`, `doc_15` - every document states "the access code for
  X is N"), but not `doc_06`. The chain breaks; the agent runs out of steps. FAIL.
- **Graph.** It follows the entity links from `site_000_02` to `doc_06` and on to `doc_05`,
  returns exactly the three facts, and the agent answers `50141015` in 22 steps. SUCCESS.
- **The point.** Same task, same model, same budget, same seed. The only thing that changed
  is the memory layer, and the trace shows exactly what each memory handed back.

## If the network fails

`scripts/demo_swap.sh --replay` prints the recorded rehearsal, which is what two
consecutive live runs produced on 30 September, identically.
