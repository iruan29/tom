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
pids=()
failed=0
total=0

# Bash 3.2 (still shipped by default on macOS) does not support `wait -n`.
# Keep a FIFO of child PIDs and wait for the oldest job whenever the configured
# concurrency limit is reached. This may wait for an older, slower job while a
# newer one has already completed, but it remains portable and never exceeds
# MAX_JOBS.
wait_for_oldest_job() {
  local pid="${pids[0]}"

  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi

  if (( ${#pids[@]} == 1 )); then
    pids=()
  else
    pids=("${pids[@]:1}")
  fi
}

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
  pids+=("$!")

  if (( ${#pids[@]} >= MAX_JOBS )); then
    wait_for_oldest_job
  fi
done < "$MANIFEST"

while (( ${#pids[@]} > 0 )); do
  wait_for_oldest_job
done

echo "Completed $total jobs; failures: $failed"
(( failed == 0 ))
