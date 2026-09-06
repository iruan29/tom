#!/usr/bin/env bash
# All 48 settings, using GPT-5.6 Sol with explicit high reasoning effort.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_NAME="gpt-5.6-sol"
export INFERENCE_REASONING_EFFORT="high"
export INFERENCE_TIMEOUT_SECONDS="${INFERENCE_TIMEOUT_SECONDS:-600}"
export INFERENCE_SDK_RETRIES="${INFERENCE_SDK_RETRIES:-0}"
# Zero means no explicit output-token cap (server defaults apply).
export INFERENCE_MAX_OUTPUT_TOKENS="${INFERENCE_MAX_OUTPUT_TOKENS:-0}"
export REQUEST_WORKERS="${REQUEST_WORKERS:-10}"
export MAX_JOBS="${MAX_JOBS:-1}"
export RESUME="${RESUME:-1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/inference_results/gpt-5.6-sol-high}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/batch_logs/gpt-5.6-sol-high/$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${OPENAI_API_KEY:-}" ]]; then
  echo 'Set OPENAI_API_KEY before running.' >&2
  exit 2
fi
echo "Model: $MODEL_NAME; reasoning_effort: $INFERENCE_REASONING_EFFORT"
echo "API: $OPENAI_BASE_URL; workers: $REQUEST_WORKERS; jobs: $MAX_JOBS; resume: $RESUME"
echo "Output: $OUTPUT_DIR"
exec bash "$ROOT_DIR/scripts/run_all_gpt56_sol.sh"
