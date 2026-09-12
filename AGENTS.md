# AGENTS.md

Instructions for AI coding agents working in this repository. Human contributors should
read [CONTRIBUTING.md](CONTRIBUTING.md) instead — this file is the machine-readable
version of the same rules, plus the constraints that matter most when code is generated
rather than typed.

## What this project is

A five-layer, component-substitutable LLM agent harness, plus a controlled experiment run
on it. The harness is a **measuring instrument**. Code that makes a measurement wrong is
worse than code that fails loudly.

Read [SPEC.md](SPEC.md) before writing anything. The interfaces there are frozen; they are
the contract, not a suggestion.

## Commands

```bash
uv sync --all-extras          # install
uv run pytest                 # all tests
uv run pytest tests/contracts # conformance suites only
uv run pytest -k memory       # one layer
uv run ruff check . --fix     # lint
uv run ruff format .          # format
uv run mypy src/              # type-check (strict on src/harness/)
```

CI runs `ruff check`, `ruff format --check`, `mypy src/`, and `pytest` with coverage.
All four must pass. Do not open a PR with any of them red.

## Hard rules

**1. Payload types are frozen.** Everything in `src/harness/types.py` is a
`@dataclass(frozen=True)` with a declared version. Within a major version you may **add**
a field with a default. You may never remove a field, rename one, or change its type.
Breaking this silently breaks every external component. If you believe a payload must
change, stop and open an issue — do not edit the file.

**2. Never mutate a payload.** Construct a new one with `dataclasses.replace`. Frozen
dataclasses enforce this at the top level; nested `dict` and `list` fields are your
responsibility. Copy before you write.

**3. A layer is not done until its conformance suite passes.** Every implementation
subclasses the contract test for its layer and inherits its tests. A new implementation
with no contract subclass is not admissible, and a PR adding one will be rejected.

**4. Components are selected by config, never by import.** No `if config.memory == "graph": from ... import GraphMemory`.
No factory functions with a match statement over known names. Use the registry
(`@register("memory", name="...")`). If a swap requires editing harness source, the
substitutability claim is false and the change is wrong.

**5. No logging outside the trace bus.** No `print`, no `logging.info` inside a layer.
Layers emit `TraceEvent` through the emitter they are given. A layer that logs its own way
is a layer that cannot be measured.

**6. Determinism.** Every source of randomness takes an explicit seed from config. No bare
`random.random()`, no unseeded `numpy.random`, no `set` iteration order leaking into
output, no wall-clock timestamps inside anything the evaluator reads.

**7. Never pickle.** Memory persistence, trace storage and results are JSON or SQLite.
`pickle.loads` on stored state is remote code execution. See [SECURITY.md](SECURITY.md).

**8. No secrets in code, config, traces, or tests.** API keys come from environment
variables only. The trace redactor strips anything matching a key pattern before write.

## Code style

- **ASCII only.** Code, comments, docstrings, strings and commit messages contain ASCII
  characters only. No emoji, em dashes, section signs, arrows or math glyphs. Write `-`,
  `->`, `>=`, `section 5`, `WARNING:` instead. Prose files (`*.md`) are unaffected.
- Python 3.12. Full type annotations on everything in `src/harness/`; `mypy` is strict there.
- `ruff` for lint and format, default line length 100. Do not hand-format.
- Functions under 50 lines, files under 800. Split rather than nest.
- Early returns over nested conditionals. Maximum 4 levels.
- Names: `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE` constants.
  Booleans read as `is_`, `has_`, `should_`, `can_`.
- No magic numbers. Budgets, caps, thresholds and limits are named constants or config.
- Docstrings on public interfaces only. Do not narrate obvious code in comments.

## Testing

Write the test first. Every behaviour change needs a test that fails before your change
and passes after.

- Arrange–Act–Assert, with the sections visible.
- Test names describe behaviour: `test_retrieve_on_empty_returns_empty_list`,
  not `test_retrieve_2`.
- Unit tests must not call a real LLM or touch the network. Use `MockToolRuntime` and the
  fake LLM client in `tests/fakes.py`.
- Anything that costs money or tokens is marked `@pytest.mark.expensive` and is excluded
  from CI by default.

## Scope discipline

This repository has a fixed deadline and a panel review on 26 October 2026. Therefore:

- Implement exactly what the task asks. No speculative abstraction, no configurability
  nobody requested, no error handling for impossible states.
- Do not refactor adjacent code you were not asked to touch. If you notice a problem,
  note it in the PR description; do not fix it in the same PR.
- Do not add a dependency without saying why in the PR. The dependency list in
  `pyproject.toml` is deliberately short.
- If a simpler approach exists than the one you were asked for, say so before building it.

## Things that are deliberately not here

Do not add them:

- Adaptive or learned component selection — future work, out of scope.
- `MemoryProvider`s that wrap other `MemoryProvider`s — composition is out of scope.
- Scheduling or recovery as harness layers — no interface, no implementation.
- An LLM-as-judge evaluator — the evaluator is deterministic by design; an LLM judge
  reintroduces the confound the whole study exists to remove.

## When you are unsure

Stop and ask. A wrong guess in a measuring instrument produces numbers that look fine and
are wrong, which is the single most expensive failure mode in this project.
