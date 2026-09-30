#!/usr/bin/env bash
# External component demo (PLAN.md, week 6): a memory provider written in a separate
# package, installed alongside the harness, selected by name in config - no harness fork.
#
#   scripts/demo_external.sh              # offline: install, discover, contract suite
#   scripts/demo_external.sh --live       # also runs one real task with it (~45 s, ~$0.02)
#
# --live needs HARNESS_LLM_BASE_URL exported, as for scripts/demo_swap.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
PACKAGE=external/harness-keyword-memory

step() { echo; echo "=== $1 ==="; }

step "1. Harness source never names the new component"
if grep -rqi "keyword" src/harness configs; then
    echo "unexpected: harness source mentions it"; exit 1
fi
echo "grep -ri keyword src/harness configs -> no match"

step "2. Its whole integration is one entry point in its own pyproject.toml"
sed -n '/entry-points/,/^keyword/p' "$PACKAGE/pyproject.toml"

step "3. Install it next to the harness"
uv pip install -q -e "$PACKAGE"
uv run --no-sync python -c \
    "from harness.registry import available; print('memory providers:', available('memory'))"

step "4. It passes the harness's own memory contract suite"
uv run --no-sync pytest "$PACKAGE/tests" -q -p no:cacheprovider 2>&1 | tail -1

if [[ "${1:-}" == "--live" ]]; then
    : "${HARNESS_LLM_BASE_URL:?export HARNESS_LLM_BASE_URL first}"
    step "5. One real task with memory: {type: keyword}"
    WORK=$(mktemp -d)
    trap 'rm -rf "$WORK"' EXIT
    sed 's/^  type: vector$/  type: keyword/' configs/demo_swap.yaml > "$WORK/demo_keyword.yaml"
    uv run --no-sync python -m harness run "$WORK/demo_keyword.yaml" \
        --out "$WORK/keyword.db" --traces "$WORK/traces" 2>&1 | grep -E "^condition=|FAILED" || true
    uv run --no-sync python scripts/demo_summary.py "$WORK"/traces/*.jsonl
fi
