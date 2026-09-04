#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-${INPUT_DIR:-$ROOT_DIR/results/inference/gpt-5.6-sol}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
MAX_JOBS="${MAX_JOBS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/evaluation}"

if [[ -z "$JUDGE_MODEL" ]]; then
  echo "Set JUDGE_MODEL before running evaluation." >&2
  exit 2
fi
if ! [[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ && "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_WORKERS and MAX_JOBS must be positive integers." >&2
  exit 2
fi

shopt -s nullglob
inputs=("$INPUT_DIR"/inference_*.json)
if (( ${#inputs[@]} == 0 )); then
  echo "No inference files found in $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
running=0
failed=0
for input in "${inputs[@]}"; do
  output="$OUTPUT_DIR/evaluation_${input##*/}"
  command=(
    "$PYTHON_BIN" "$ROOT_DIR/evaluate.py"
    --input "$input"
    --output "$output"
    --judge-model "$JUDGE_MODEL"
    --workers "$EVAL_WORKERS"
  )
  echo "[EVAL] ${input##*/}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi &
  running=$((running + 1))
  if (( running >= MAX_JOBS )); then
    wait -n || failed=$((failed + 1))
    running=$((running - 1))
  fi
done

while (( running > 0 )); do
  wait -n || failed=$((failed + 1))
  running=$((running - 1))
done

echo "Evaluated ${#inputs[@]} files; failures: $failed"
(( failed == 0 ))
