# SECURITY

This is a research harness that runs model-generated actions against a filesystem, loads
third-party code, and holds API credentials. Three of those are genuinely dangerous. Read
this before touching the tool runtime, the config loader, or plugin discovery.

Two kinds of risk here, and both matter:

- **Classical** — credential leakage, path traversal, code execution.
- **Measurement integrity** — a defect that silently produces wrong numbers. In a project
  whose only output is numbers, this is the more expensive failure.

---

## 1. Secrets

- API keys come from **environment variables only**. Never in YAML, never in code, never in
  a test fixture, never in a commit message.
- `.env` is gitignored. `.env.example` holds names with empty values.
- **The trace redactor runs before any trace is written.** Traces capture prompts and
  responses; a key pasted into a prompt during debugging would otherwise be written to disk
  and pushed. Redact anything matching a key pattern (`sk-`, `AIza`, long base64-ish runs)
  at emit time, not at read time.
- `results/`, `traces/`, `.sandbox/` and `*.db` are gitignored. Traces contain full prompts
  and are large; they are artefacts, not source.
- Add a pre-commit secret scan before the first real commit. A leaked key on a public repo
  is found by scanners in minutes.
- If a key is exposed: **rotate it first, then rewrite history.** In that order.

---

## 2. Spine and runtime

- **Global spend kill-switch and per-run token cap go in on day one**, not after the first
  incident. One looping agent will otherwise burn a very large context repeatedly.
- Hard `max_steps` enforced by the runtime, not trusted to the planner.
- Wall-clock timeout per run. A hung tool call must not stall a sweep overnight.
- The runner is resumable and idempotent per `run_id`. A retry must never write a duplicate
  results row — duplicated rows silently corrupt every downstream statistic.

---

## 3. Planner

- **Model output is data, never code.** Never `eval`, `exec`, or `pickle.loads` anything the
  model produced.
- Parse the action, then **validate it against a tool allowlist and an argument schema
  before execution.** An unknown tool name returns an error `Observation`; it never reaches
  the runtime.
- Malformed output is an expected condition, not an exception. `PlannerContract` tests this.
- ⚠️ **Prompt injection is live here.** Tool observations and retrieved memory items are
  attacker-controllable in any real deployment, and in our own tasks the `repo_edit` family
  puts generated file content into the prompt. Keep system instructions and goal in
  segments the context manager can never evict, and treat everything else as untrusted
  content.

---

## 4. Context manager

- The budget must bind. `test_never_exceeds_declared_budget` is a security-adjacent test:
  an unbounded prompt is an unbounded bill.
- System instructions and goal are never evictable — that is both a correctness property
  and the only structural defence against injected content displacing them.
- `dropped` records what was evicted. It must not record secret content verbatim; log
  segment identifiers and reasons, not payloads.

---

## 5. Tool runtime — the highest-risk layer

`LocalToolRuntime` executes actions chosen by a language model. Treat every action as
hostile input.

**Required, all of them:**

- **Sandbox root.** Every filesystem operation is confined to a per-run directory under
  `sandbox_root`. Created fresh per run, deleted after.
- **Path validation:** `Path(root, user_path).resolve()` then assert the result is a
  descendant of `root.resolve()`. This catches `../`, absolute paths, and symlink escapes.
  A path outside the sandbox returns an error `Observation` — it never raises and never
  executes. `ToolRuntimeContract.test_path_outside_sandbox_is_refused` covers this.
- **No shell.** `subprocess` with a list argv and `shell=False`, or no subprocess at all.
  Never interpolate model output into a shell string.
- **Explicit tool allowlist.** Tools are registered by name; anything else is an error
  observation.
- **No network access from tools.** Nothing in the task families requires it.
- **Timeout and output size cap** on every tool call. An unbounded tool result blows the
  context budget and the bill.
- Never delete outside the sandbox root. Never write outside it.

`MockToolRuntime` touches no filesystem at all and is what unit tests use.

---

## 6. Memory

- **Never pickle.** Persistence is JSON or SQLite. `pickle.loads` on stored state is remote
  code execution, and a memory store is exactly the kind of file that gets shared between
  machines.
- **`reset()` must actually clear everything** — indices, caches, embedded vectors, graph
  nodes, file handles. State leaking between runs is simultaneously a correctness bug and a
  cross-run contamination bug, and it produces plausible-looking wrong numbers.
- Retrieved items are re-injected into the prompt. They are untrusted content; see §3.
- Memory content is scoped per `run_id`. No shared global store across runs, ever.

---

## 7. Third-party components — the trust boundary

Registration without forking means **installing a third-party package executes its code**,
at import time, with your privileges. This is inherent to the design, not a flaw in it, but
it must be stated plainly:

- **The conformance suite is a specification, not a sandbox.** Passing `MemoryProviderContract`
  proves a component honours the interface. It proves nothing about whether the component is
  safe. Do not confuse admissibility with trust.
- Only install components from sources you trust. For the thesis, that means components
  written by this team plus the one worked external example.
- The extension guide must say this in the same words. A user who believes the contract
  suite is a security boundary will get hurt.
- Entry-point discovery loads every installed `harness.*` entry point. Run experiments in a
  clean virtualenv containing only what the study needs.

---

## 8. Evaluator

- Deterministic and offline. Same trace ⇒ same verdict, always.
- **No LLM-as-judge.** Beyond the confound it reintroduces, an LLM judge reading
  model-generated traces is directly injectable — a trace could contain text arguing for a
  passing verdict.

---

## 9. Repository

- `main` is protected: no force-push, no direct commits, PR required.
- Every PR reviewed by the peer owner (Ansh↔Vivek, Muni↔Jugal).
- CI must be green before merge: ruff, mypy, pytest.
- Give write access to the four team members only.
- Never commit: keys, `.env`, traces, results databases, model weights, sandbox contents.

---

## Checklist before any commit

- [ ] No hardcoded credentials anywhere in the diff
- [ ] No `pickle`, `eval`, `exec`, or `shell=True`
- [ ] Any new filesystem path goes through the sandbox validator
- [ ] New model-facing input is treated as untrusted
- [ ] No `print` or direct `logging` inside a layer — trace bus only
- [ ] New randomness takes an explicit seed
- [ ] Tests added, and they pass without network access
