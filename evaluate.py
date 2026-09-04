"""Evaluate SynchToM inference results with an OpenAI-compatible judge."""

import argparse
import fcntl
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


DIMENSIONS = (
    "latent_belief_explanation",
    "user_profile_modeling",
    "correct_resolution",
)
DIMENSION_ORDER = DIMENSIONS
MODE_DIMENSIONS = {
    "joint": DIMENSIONS,
    "direct_solution": ("correct_resolution",),
    "cot_solution": ("correct_resolution",),
    "direct_belief_profile": (
        "latent_belief_explanation",
        "user_profile_modeling",
    ),
}
api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3.8-27b-judge")
TIMEOUT_SECONDS = 180
MAX_RETRIES = 3
NUM_INSTANCES_TO_EVALUATE = None
eval_dir = Path(os.environ.get("EVALUATION_DIR", "evaluation_results"))


def extract_json(text):
    if not isinstance(text, str):
        raise ValueError("Judge response content is not text")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    without_think = re.sub(
        r"<(?:think|thinking)>.*?</(?:think|thinking)>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    decoder = json.JSONDecoder()
    for candidate in (without_think, text):
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError(f"No valid JSON object in response: {text[:300]!r}")


extract_json_from_text = extract_json


def dimensions_for(instance):
    inference = instance["inference"]
    rubrics = instance["rubrics"]
    declared = instance.get("inference_dimensions")
    if declared is not None:
        dimensions = [item for item in DIMENSIONS if item in declared]
    else:
        mode = instance.get("inference_mode")
        dimensions = (
            list(MODE_DIMENSIONS[mode])
            if mode in MODE_DIMENSIONS
            else [item for item in DIMENSIONS if item in inference]
        )
    if not dimensions:
        raise ValueError("No evaluable dimensions")
    for dimension in dimensions:
        if not isinstance(inference.get(dimension), str) or not inference[dimension].strip():
            raise ValueError(f"Missing inference field: {dimension}")
        if not isinstance(rubrics.get(dimension), list) or not rubrics[dimension]:
            raise ValueError(f"Missing rubric: {dimension}")
    return dimensions


get_evaluation_dimensions = dimensions_for


def build_judge_prompt(inference, rubrics, dimensions):
    candidate = {item: inference[item] for item in dimensions}
    active_rubrics = {item: rubrics[item] for item in dimensions}
    schema = {
        item: {
            "criterion_scores": {
                f"criterion_{index}": 0
                for index in range(1, len(active_rubrics[item]) + 1)
            },
            "feedback": "Concise explanation of which criteria passed or failed",
        }
        for item in dimensions
    }
    return f"""# Rubric-based evaluation

Evaluate the candidate response only on the rubric dimensions provided below.

For every criterion, assign a binary score:

- 1: the candidate explicitly and correctly covers the criterion's core meaning and does not contradict it.
- 0: the candidate omits it, is materially incomplete or vague, is incorrect, or contradicts it.

Judge semantic meaning rather than requiring an exact keyword match. Do not assume claims that are not present in the candidate. Assess each criterion independently, keep the feedback concise, and return only the requested fields.

## Candidate response

{json.dumps(candidate, indent=2, ensure_ascii=False)}

## Rubrics

{json.dumps(active_rubrics, indent=2, ensure_ascii=False)}

## Response format

Return only one valid JSON object matching this structure. Replace each example score with the appropriate 0 or 1.

{json.dumps(schema, indent=2, ensure_ascii=False)}
"""


def judge_prompt(instance, dimensions):
    return build_judge_prompt(instance["inference"], instance["rubrics"], dimensions)


def binary(value, name):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    raise ValueError(f"{name} is not binary: {value!r}")


def normalize(raw, instance, dimensions, raw_text):
    result = {}
    total = 0
    maximum = 0
    for dimension in dimensions:
        section = raw.get(dimension)
        scores = section.get("criterion_scores") if isinstance(section, dict) else None
        if not isinstance(scores, dict):
            raise ValueError(f"Missing scores for {dimension}")
        normalized_scores = {}
        for index in range(1, len(instance["rubrics"][dimension]) + 1):
            key = f"criterion_{index}"
            if key not in scores:
                raise ValueError(f"Missing {dimension}.{key}")
            normalized_scores[key] = binary(scores[key], f"{dimension}.{key}")
        subtotal = sum(normalized_scores.values())
        submax = len(normalized_scores)
        result[dimension] = {
            "criterion_scores": normalized_scores,
            "total_score": subtotal,
            "max_score": submax,
            "feedback": str(section.get("feedback", "")).strip(),
        }
        total += subtotal
        maximum += submax
    result["overall_summary"] = {
        "total_score": total,
        "max_score": maximum,
        "percentage": total / maximum * 100 if maximum else 0.0,
        "evaluated_dimensions": list(dimensions),
        "overall_assessment": (
            f"Passed {total} of {maximum} rubric criteria across "
            f"{len(dimensions)} evaluated dimension(s)."
        ),
    }
    result["raw_response"] = raw_text
    return result


def normalize_judge_result(raw, rubrics, dimensions, raw_response):
    return normalize(raw, {"rubrics": rubrics}, dimensions, raw_response)


def evaluate_one(instance, client, model, timeout):
    dimensions = dimensions_for(instance)
    prompt = judge_prompt(instance, dimensions)
    last_error = None
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=timeout,
            )
            text = response.choices[0].message.content
            judged = normalize(extract_json(text), instance, dimensions, text)
            return {
                "instance_id": instance["instance_id"],
                "domain": instance.get("domain", "Unknown"),
                "inference_mode": instance.get("inference_mode", "legacy_joint"),
                "evaluated_dimensions": dimensions,
                "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ground_truth": instance.get("ground_truth", {}),
                "inference": instance["inference"],
                "rubrics": {item: instance["rubrics"][item] for item in dimensions},
                "evaluation": judged,
            }, None
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 ** attempt)
    return None, str(last_error)


def evaluate_with_rubric(
    instance_id, ground_truth, inference_result, rubrics, client, dimensions=None
):
    del ground_truth
    if dimensions is None:
        dimensions = [item for item in DIMENSIONS if item in inference_result and item in rubrics]
    instance = {
        "instance_id": instance_id,
        "inference": inference_result,
        "rubrics": rubrics,
        "inference_dimensions": dimensions,
    }
    result, error = evaluate_one(instance, client, JUDGE_MODEL, TIMEOUT_SECONDS)
    if result is None:
        raise RuntimeError(error)
    return result["evaluation"]


def evaluate_instance(instance, client):
    """Backward-compatible entry point for evaluating one inference instance."""
    return evaluate_one(instance, client, JUDGE_MODEL, TIMEOUT_SECONDS)


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


write_json_atomic = atomic_write


def build_summary(results, input_file, output_file, judge_model, failed_count):
    dimension_scores = {item: [] for item in DIMENSIONS}
    overall_scores = []
    modes = set()
    for result in results:
        modes.add(result.get("inference_mode", "legacy_joint"))
        evaluation = result["evaluation"]
        for dimension in DIMENSIONS:
            if dimension in evaluation:
                score = evaluation[dimension]
                dimension_scores[dimension].append(
                    score["total_score"] / score["max_score"] if score["max_score"] else 0.0
                )
        overall_scores.append(evaluation["overall_summary"]["percentage"])
    scores = {}
    for dimension, values in dimension_scores.items():
        if values:
            average = sum(values) / len(values)
            scores[dimension] = {
                "evaluated_instances": len(values),
                "average_score": round(average, 4),
                "percentage": round(average * 100, 2),
            }
    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    scores["overall"] = {
        "evaluated_instances": len(overall_scores),
        "average_score": round(overall / 100, 4),
        "percentage": round(overall, 2),
    }
    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "judge_model": judge_model,
        "inference_modes": sorted(modes),
        "total_instances": len(results) + failed_count,
        "successful_evaluations": len(results),
        "failed_evaluations": failed_count,
        "scores": scores,
    }


def append_summary(summary_file, summary):
    summary_file = Path(summary_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = summary_file.with_name(f".{summary_file.name}.lock")
    with open(lock_file, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                entries = json.loads(summary_file.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                entries = []
            if not isinstance(entries, list):
                entries = []
            entries.append(summary)
            atomic_write(summary_file, entries)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-model", "-j", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1 second")
    input_path = Path(args.input)
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    model = args.judge_model or args.model or JUDGE_MODEL
    client = OpenAI(api_key=args.api_key or api_key, base_url=args.base_url or base_url)
    input_name = input_path.stem.removeprefix("inference_")
    output = args.output or str(
        eval_dir / f"evaluation_{input_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    results = []
    failures = []
    lock = threading.Lock()
    atomic_write(output, results)

    def process(instance):
        result, error = evaluate_one(instance, client, model, args.timeout)
        with lock:
            if result is not None:
                results.append(result)
                atomic_write(output, results)
            else:
                failures.append({"instance_id": instance["instance_id"], "error": error})
            done = len(results) + len(failures)
            print(
                f"{Path(output).name}: {done}/{len(dataset)} "
                f"success={len(results)} failed={len(failures)}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, instance) for instance in dataset]
        for future in as_completed(futures):
            future.result()
    if results:
        summary = build_summary(results, input_path, output, model, len(failures))
        summary_file = Path(os.environ.get("EVALUATION_SUMMARY_FILE", "evaluation_summary.json"))
        append_summary(summary_file, summary)
    print(json.dumps({"output": output, "success": len(results), "failures": failures}))


if __name__ == "__main__":
    main()
