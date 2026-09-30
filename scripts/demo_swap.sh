#!/usr/bin/env bash
# Live component swap (PLAN.md, panel item 2): one task, one config line, two memories.
#
#   export HARNESS_LLM_BASE_URL=https://aiplatform.googleapis.com/v1/projects/btp-vivek/locations/global/endpoints/openapi
#   scripts/demo_swap.sh              # live, two real runs, about 90 seconds
#   scripts/demo_swap.sh --replay     # no network: prints the recorded rehearsal
#
# Every run writes to a fresh temporary directory, so a rehearsal never finds its own
# earlier rows and skips the run as already done.
set -euo pipefail

cd "$(dirname "$0")/.."
RECORDING=docs/demo-recording.txt

if [[ "${1:-}" == "--replay" ]]; then
    cat "$RECORDING"
    exit 0
fi

: "${HARNESS_LLM_BASE_URL:?export HARNESS_LLM_BASE_URL first - see the header of this script}"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
sed 's/^  type: vector$/  type: graph/' configs/demo_swap.yaml > "$WORK/demo_graph.yaml"

run() {
    local label=$1 config=$2
    echo
    echo "=== $label ==="
    uv run python -m harness run "$config" \
        --out "$WORK/$label.db" --traces "$WORK/$label-traces" 2>&1 \
        | grep -E "^condition=|FAILED" || true
    uv run python scripts/demo_summary.py "$WORK/$label-traces"/*.jsonl
}

echo "The only difference between the two runs:"
diff configs/demo_swap.yaml "$WORK/demo_graph.yaml" || true

run vector configs/demo_swap.yaml
run graph "$WORK/demo_graph.yaml"
