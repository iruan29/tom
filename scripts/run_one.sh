#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: bash scripts/run_one.sh BENCHMARK SETTING TURN [extra inference.py arguments]" >&2
  exit 2
fi

BENCHMARK="$1"
SETTING="$2"
TURN="$3"
shift 3

case "$BENCHMARK" in
  culture|education|pref|swe) ;;
  *) echo "Unknown benchmark: $BENCHMARK" >&2; exit 2 ;;
esac

case "$SETTING" in
  joint|direct_solution|cot_solution|direct_belief_profile) ;;
  *) echo "Unknown setting: $SETTING" >&2; exit 2 ;;
esac

case "$TURN" in
  0|5|10) ;;
  *) echo "TURN must be one of: 0, 5, 10" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${MODEL_NAME:-gpt-5.6-sol}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUEST_WORKERS="${REQUEST_WORKERS:-8}"
SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/inference/$SAFE_MODEL_NAME}"

command=(
  "$PYTHON_BIN" "$ROOT_DIR/inference.py"
  --input "$ROOT_DIR/data/benchmarks/${BENCHMARK}-benchmark.json"
  --model "$MODEL_NAME"
  --steps "$TURN"
  --mode "$SETTING"
  --workers "$REQUEST_WORKERS"
  --output-dir "$OUTPUT_DIR"
  --trajectories-dir "$ROOT_DIR/data/trajectories"
  "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
