# harness-keyword-memory

A memory provider for the harness that lives outside it. Owner: Gangavarapu Vivek.

It ranks stored items by the number of distinct query words they contain, newest first on
ties - no embeddings, no graph. It is here to prove the registry's claim: a component in a
separate package becomes selectable by name in config, and passes the harness's own
conformance suite, without a line of harness source changing.

```
uv pip install -e external/harness-keyword-memory     # from the harness repo root
uv run --no-sync pytest external/harness-keyword-memory/tests
```

Then in any config: `memory: {type: keyword}`. The whole integration is the entry point in
`pyproject.toml`:

```toml
[project.entry-points."harness.memory"]
keyword = "harness_keyword_memory:KeywordMemory"
```

The contract test imports `tests.contracts.memory` from the harness repo, so run it from the
repo root: the conformance suite is not yet part of the installed `harness` package
(DECISIONS.md, 2026-09-30).
