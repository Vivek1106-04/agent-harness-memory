# CONTRIBUTING

Four people building in parallel against frozen interfaces. These rules exist so that
parallel work merges cleanly and the numbers at the end are trustworthy.

## Before you write code

1. Read [SPEC.md](SPEC.md). The interfaces are the contract.
2. Check [PLAN.md](PLAN.md) for what your layer owes this week.
3. If you are touching the tool runtime, config loading, or plugin discovery, read
   [SECURITY.md](SECURITY.md) first.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
cp .env.example .env      # then fill it in; .env is gitignored
uv run pytest             # should be green before you start
```

## Ownership

| Layer | Owner | Reviewer |
|---|---|---|
| Spine, planner | Ansh | Vivek |
| Context manager | Muni | Jugal |
| Environment, tool runtime, instruments, runner | Vivek | Ansh |
| Evaluator | Jugal | Muni |
| Vector + Episodic memory | Ansh + Vivek | each other |
| Graph + Hybrid memory | Muni + Jugal | each other |

The owner is accountable for **all seven conditions** on their layer: frozen payload,
abstract interface, ≥2 implementations, conformance suite, config-registry entry, trace
emission, documentation. A layer is not delivered until all seven hold.

**Memory implementations are peer-reviewed by the paired owner.** This is not a formality —
the effort confound is the most serious threat in the study, and one person working alone
on an implementation is exactly where it starts.

## Branches and commits

```
<type>/<layer>-<short-description>
feat/memory-vector-retrieve
fix/context-eviction-accounting
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`.

Commit messages:

```
<type>: <description>

<why, if it isn't obvious from the diff>
```

Never commit directly to `main`. Never force-push a shared branch.

## Pull requests

Every PR states:

- **What changed** and which layer it belongs to
- **Which contract suite** now passes, or which tests were added
- **Anything you noticed but did not fix** — note it, do not fix it in the same PR

Merge requires: CI green (ruff, mypy, pytest) and approval from the reviewer above.

Keep PRs small. A PR that touches three layers is three PRs.

## The rules that protect the measurement

These are not style preferences. Breaking one produces numbers that look fine and are wrong.

1. **Payload types are frozen.** Add a field with a default if you must. Never remove,
   rename or retype one. Opening `types.py` in a feature PR is a red flag.
2. **Never mutate a payload.** `dataclasses.replace`, always. Copy nested dicts and lists.
3. **Components are selected by config, never by import.** No factory with a match statement
   over known names. If a swap requires editing harness source, the change is wrong.
4. **No logging outside the trace bus.** No `print`, no `logging.info` inside a layer.
5. **Determinism.** Every source of randomness takes an explicit seed from config. No
   unseeded RNG, no set-iteration order leaking into output, no wall-clock inside anything
   the evaluator reads.
6. **`reset()` clears everything.** State leaking between runs produces plausible-looking
   wrong results, which is the worst kind.
7. **Every result carries its provenance** — config hash, git commit, payload version,
   model and embedding revisions, all seeds. A result you cannot reconstruct is not a
   result.

## Tests

Write the test first. It should fail before your change and pass after.

- Arrange–Act–Assert, sections visible.
- Names describe behaviour: `test_retrieve_on_empty_returns_empty_list`.
- Unit tests never touch the network or a real LLM. Use `MockToolRuntime` and the fake
  client in `tests/fakes.py`.
- Anything costing tokens is `@pytest.mark.expensive` and excluded from CI.
- A new implementation without a contract-suite subclass will not be merged.

## Adding a new component

```python
from harness.registry import register
from harness.interfaces import MemoryProvider

@register("memory", name="my_memory")
class MyMemory(MemoryProvider):
    capabilities = frozenset({"semantic"})
    ...
```

```python
# tests/contracts/test_my_memory.py
from tests.contracts.memory import MemoryProviderContract

class TestMyMemory(MemoryProviderContract):
    provider_factory = MyMemory
```

Then it is selectable as `memory: {type: my_memory}`. Nothing in the harness names your
class. If you had to edit harness source to make it work, that is a bug in the harness —
file it.

## Communication

- **Monday:** each owner posts what landed, what is blocked, what they need from whom.
- **Friday:** one person mails the supervisor a five-line summary. Non-negotiable — the
  last review went badly because she heard things late.
- Blocked for more than half a day? Say so immediately. Four people on a 6-week runway
  cannot absorb silent blockers.

## Logs you must keep

- `docs/sweep-memos/` — one page after every sweep: expected, happened, surprised us.
- `docs/decisions.md` — every place an implementation did not fit an interface, what
  changed, what it cost. All five interfaces, not memory alone.
- `docs/citation-ledger.md` — no citation enters any document until someone has opened the
  paper and pasted the sentence supporting the claim.
- `docs/tuning-hours.md` — declared budget and actual hours per memory implementation. This
  is the effort-confound mitigation and it gets reported in the thesis.
