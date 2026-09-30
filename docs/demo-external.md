# External component - runbook

PLAN.md, week 6: one provider in a separate package, registered without forking. Owner:
Vivek. About 40 seconds live, or 5 seconds offline.

```
scripts/demo_external.sh           # offline: grep, entry point, install, discovery, contract suite
scripts/demo_external.sh --live    # plus one real task with memory: {type: keyword}
```

What each step shows:

1. `grep -ri keyword src/harness configs` finds nothing: the harness does not know the
   component exists.
2. The package's only integration is one entry point in its own `pyproject.toml`.
3. After `uv pip install -e`, `available('memory')` lists `keyword` beside the built-ins.
4. The package's test subclasses the harness's `MemoryProviderContract` and passes all of
   it: the same admission test every built-in memory passes.
5. (`--live`) A real Vertex run with `type: keyword`. It fails task 000 - word overlap
   returns the newest look-alike documents, not the linked facts - and that is fine: the
   claim is substitutability, not that this memory is good.

Recorded run, 30 September: [demo-external-recording.txt](demo-external-recording.txt).
