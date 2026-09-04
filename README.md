# SynchToM: four-setting benchmark runner

This repository packages the original SynchToM benchmark data together with the
updated inference and evaluation pipeline used for the four-setting experiment.
It includes an explicit script for every combination of:

- 4 benchmarks: `culture`, `education`, `pref`, and `swe`
- 4 settings: `joint`, `direct_solution`, `cot_solution`, and
  `direct_belief_profile`
- 3 trajectory contexts: 0, 5, and 10 turns

The complete matrix contains **48 experiments** and defaults to the model name
`gpt-5.6-sol`.

## Repository layout

```text
.
├── data/
│   ├── benchmarks/          # Four original benchmark JSON files (391 instances)
│   └── trajectories/        # Original per-instance trajectories (392 files)
├── scripts/
│   ├── gpt56_sol/           # 48 explicit experiment scripts + manifest
│   ├── run_one.sh           # Validated single-experiment launcher
│   ├── run_all_gpt56_sol.sh # Batch launcher for the full 4 × 4 × 3 matrix
│   └── evaluate_all.sh      # Batch evaluator
├── tests/
├── inference.py
└── evaluate.py
```

The benchmark JSON files are byte-for-byte unchanged from the original release.
Their legacy `trajectories/...` references are resolved by `inference.py` against
`data/trajectories`, so the data itself does not need to be rewritten.

## Setup

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Export the endpoint configuration before running. The scripts do not read `.env`
automatically.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

## Run GPT-5.6 Sol

Run one experiment:

```bash
bash scripts/gpt56_sol/culture_joint_turn0.sh
bash scripts/gpt56_sol/swe_cot_solution_turn10.sh
```

Run the full 48-experiment matrix:

```bash
REQUEST_WORKERS=8 MAX_JOBS=1 bash scripts/run_all_gpt56_sol.sh
```

`REQUEST_WORKERS` controls concurrent requests inside one experiment. `MAX_JOBS`
controls how many experiment scripts run simultaneously, so their product is the
maximum possible request concurrency. Results are written under
`results/inference/gpt-5.6-sol/` and logs under
`results/logs/gpt-5.6-sol/`; both are ignored by Git.

Resume is enabled by default. Re-running the same command reuses the matching
result file, skips valid completed instances, and requests only missing or failed
instances. Each successful result is written atomically, so an interrupted run
can safely continue:

```bash
REQUEST_WORKERS=32 MAX_JOBS=8 bash scripts/run_all_gpt56_sol.sh
```

Set `RESUME=0` to deliberately start a new timestamped result file instead:

```bash
RESUME=0 REQUEST_WORKERS=8 MAX_JOBS=1 bash scripts/run_all_gpt56_sol.sh
```

Useful overrides:

```bash
MODEL_NAME=gpt-5.6-sol REQUEST_WORKERS=4 \
  bash scripts/gpt56_sol/education_direct_solution_turn5.sh

DRY_RUN=1 bash scripts/run_all_gpt56_sol.sh
```

The canonical list of all combinations is
[`scripts/gpt56_sol/manifest.tsv`](scripts/gpt56_sol/manifest.tsv).

## Four settings

| Setting | Inference output | Evaluation dimensions |
|---|---|---|
| `joint` | belief, profile, solution | belief, profile, solution |
| `direct_solution` | solution only | solution |
| `cot_solution` | solution only, with a step-by-step instruction | solution |
| `direct_belief_profile` | belief and profile only | belief and profile |

Each result records `inference_mode` and `inference_dimensions`. The evaluator
uses those fields to select only the matching rubrics, and it recalculates all
criterion totals and percentages instead of trusting model-generated totals.

## Evaluate results

Evaluate one file:

```bash
python evaluate.py \
  --input results/inference/gpt-5.6-sol/INFERENCE_FILE.json \
  --judge-model YOUR_JUDGE_MODEL \
  --workers 8
```

Evaluate every inference file in a result directory:

```bash
export JUDGE_MODEL=YOUR_JUDGE_MODEL
bash scripts/evaluate_all.sh results/inference/gpt-5.6-sol
```

The judge uses `OPENAI_API_KEY` and `OPENAI_BASE_URL` unless they are overridden
with `evaluate.py --api-key` and `--base-url`.

## Verification

```bash
python scripts/validate_repository.py
python -m unittest discover -s tests -v
```

## License

See [LICENSE](LICENSE).
