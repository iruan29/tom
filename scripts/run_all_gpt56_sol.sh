#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT_DIR/scripts/gpt56_sol/manifest.tsv"
MAX_JOBS="${MAX_JOBS:-1}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/results/logs/${MODEL_NAME:-gpt-5.6-sol}}"

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
running=0
failed=0
total=0

while IFS=$'\t' read -r benchmark setting turn script; do
  [[ "$benchmark" == "benchmark" ]] && continue
  total=$((total + 1))
  log_file="$LOG_DIR/${benchmark}_${setting}_turn${turn}.log"
  echo "[START] $benchmark / $setting / ${turn}-turn"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=1 bash "$ROOT_DIR/$script"
  else
    bash "$ROOT_DIR/$script" >"$log_file" 2>&1
  fi &
  running=$((running + 1))

  if (( running >= MAX_JOBS )); then
    wait -n || failed=$((failed + 1))
    running=$((running - 1))
  fi
done < "$MANIFEST"

while (( running > 0 )); do
  wait -n || failed=$((failed + 1))
  running=$((running - 1))
done

echo "Completed $total jobs; failures: $failed"
(( failed == 0 ))
