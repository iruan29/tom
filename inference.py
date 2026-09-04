"""
Inference Script for Theory-of-Mind (ToM) Dataset

This script runs four inference variants over the same observations, instructions,
and trajectories: joint ToM inference, direct solution, CoT solution, and direct
belief/profile inference.

Usage:
    python inference.py --input <dataset.json> --model <model_name> --steps <num_steps> --mode <mode> [options]

Environment Variables:
    OPENAI_API_KEY: API key for the inference model (default: 'EMPTY')
    OPENAI_BASE_URL: Base URL for the OpenAI API (default: 'http://localhost:8080/v1')
    TRAJECTORIES_DIR: Base directory for trajectory files (default: './data/trajectories')
    IMAGES_DIR: Directory containing images (default: './data/images')

Example:
    python inference.py --input dataset.json --model gpt-4 --steps 5 --mode joint
    python inference.py --input dataset.json --model gpt-4 --steps 5 --mode direct_solution
    python inference.py -i dataset.json -m qwen3-32b -s 0  # No trajectory
"""

try:
    from openai import OpenAI
except ModuleNotFoundError:  # Allows prompt/unit tests without the API dependency.
    OpenAI = None
import json
import datetime
import os
import logging
import time
import re
import argparse
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables. Never put live credentials in source.
api_key = os.environ.get('OPENAI_API_KEY', 'EMPTY')
base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:8080/v1')

# Number of instances to process (set to None to process all)
NUM_INSTANCES_TO_PROCESS = None

# Timeout and retry configuration
TIMEOUT_SECONDS = 180
MAX_RETRIES = 2

# Create inference directory (relative path)
inference_dir = Path("inference_results")
inference_dir.mkdir(exist_ok=True)

# Trajectories and images directories (configurable via env vars)
TRAJECTORIES_DIR = Path(os.environ.get('TRAJECTORIES_DIR', './data/trajectories'))
IMAGES_DIR = Path(os.environ.get('IMAGES_DIR', './data/images'))


DIMENSION_ORDER = (
    "latent_belief_explanation",
    "user_profile_modeling",
    "correct_resolution",
)

INFERENCE_MODES = {
    "joint": {
        "description": "infer belief, profile, and solution together",
        "output_fields": DIMENSION_ORDER,
    },
    "direct_solution": {
        "description": "produce only the solution, without an explicit reasoning trace",
        "output_fields": ("correct_resolution",),
    },
    "cot_solution": {
        "description": "think step by step carefully, then produce only the solution",
        "output_fields": ("correct_resolution",),
    },
    "direct_belief_profile": {
        "description": "infer only the user's latent belief and profile",
        "output_fields": (
            "latent_belief_explanation",
            "user_profile_modeling",
        ),
    },
}

COT_INSTRUCTION = "Think step by step carefully before giving your answer."


def encode_image_to_base64(image_id: str) -> str:
    """Encode image file to base64 string.

    Args:
        image_id: Image identifier (e.g., "11-1")

    Returns:
        Base64 encoded string of the image
    """
    # Try direct path first
    image_path = IMAGES_DIR / f"{image_id}.png"
    if image_path.exists():
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Try subdirectories
    for subdir in ["image1", "malay", "new_data", "0502-double_ambiguity"]:
        image_path = IMAGES_DIR / subdir / f"{image_id}.png"
        if image_path.exists():
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

    raise FileNotFoundError(f"Image not found for image_id: {image_id}")


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, handling cases where it's wrapped in markdown or other text.

    Enhanced version with multiple fallback strategies for robust JSON extraction.
    """
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    think_pattern = r'<(?:think|thinking)>.*?</(?:think|thinking)>'
    text_no_think = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    try:
        result = json.loads(text_no_think.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Decode the first complete JSON object. JSONDecoder correctly handles
    # nested objects and braces inside quoted strings.
    decoder = json.JSONDecoder()
    for candidate in (text_no_think, text):
        for match in re.finditer(r'\{', candidate):
            try:
                result, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                return result

    raise ValueError(f"Could not extract valid JSON from response after trying all strategies. Response preview: {text[:500]}...")


def load_trajectory(trajectory_path, step_num=None):
    """Load trajectory from file and filter out 'thought' field to prevent information leakage.

    Args:
        trajectory_path: Path to the trajectory file (relative to TRAJECTORIES_DIR or absolute)
        step_num: Number of steps to keep (None = keep all steps)

    Returns:
        list: Filtered trajectory, or None if file doesn't exist
    """
    trajectory_path = Path(trajectory_path)
    candidates = [trajectory_path, TRAJECTORIES_DIR / trajectory_path]
    # The published benchmark JSON is kept byte-for-byte unchanged and contains
    # legacy paths beginning with "trajectories/". Resolve that prefix against
    # the organized data directory without rewriting the source data.
    if trajectory_path.parts[:1] == ("trajectories",):
        candidates.insert(0, TRAJECTORIES_DIR.joinpath(*trajectory_path.parts[1:]))
    full_path = next((path for path in candidates if path.exists()), None)

    if full_path is None:
        logger.warning(f"   ⚠️  Trajectory file not found: {trajectory_path}")
        return None

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Filter out 'thought' field from each turn
        trajectory = data.get('trajectory', data)
        filtered_trajectory = []
        for turn in trajectory:
            filtered_turn = {
                "turn": turn["turn"],
                "action": turn["action"],
                "observation": turn.get("observation", "")
            }
            filtered_trajectory.append(filtered_turn)

        # Keep only first step_num steps if specified
        if step_num is not None and step_num > 0:
            filtered_trajectory = filtered_trajectory[:step_num]
            logger.info(f"   📏 Keeping first {step_num} steps (total available: {len(data.get('trajectory', data))} steps)")
        if step_num == 0:
            filtered_trajectory = []
        return filtered_trajectory
    except Exception as e:
        logger.error(f"   ✗ Error loading trajectory {trajectory_path}: {e}")
        return None


def build_inference_prompt(observation, explicit_instruction, trajectory, inference_mode):
    """Build a mode-specific prompt while keeping the evidence presentation fixed."""
    if inference_mode not in INFERENCE_MODES:
        raise ValueError(
            f"Unknown inference mode: {inference_mode}. "
            f"Choose from: {', '.join(INFERENCE_MODES)}"
        )

    trajectory_text = (
        json.dumps(trajectory, indent=2, ensure_ascii=False)
        if trajectory
        else "No behavioral trajectory is provided for this setting."
    )
    context = f"""You need to infer key aspects of a user-agent interaction based on the user's observation, explicit instruction, and behavioral trajectory.

## Given information

### User's observation
{observation}

### User's explicit instruction
{explicit_instruction}

### User's behavioral trajectory
{trajectory_text}
"""

    belief_task = """What does the user believe is the root cause of the problem? Explain the user's mental model, how it accounts for the observable instruction or behavior, and why the user holds this belief."""
    profile_task = """What relevant preference, bias, cultural background, assumption, experience level, or worldview led the user to form this belief? Explain why this profile produced the misconception."""
    solution_task = """What is the actual root cause of the problem, and what is the correct solution? Explain what the user should do to resolve the underlying issue."""

    direct_solution_task = f"""## Task

{solution_task}"""

    tasks = {
        "joint": f"""## Task

Infer the following three aspects together:

1. `latent_belief_explanation`: {belief_task}
2. `user_profile_modeling`: {profile_task}
3. `correct_resolution`: {solution_task}

Keep the three fields consistent with one another and make each answer specific.""",
        "direct_solution": direct_solution_task,
        "cot_solution": direct_solution_task + f"\n\n{COT_INSTRUCTION}",
        "direct_belief_profile": f"""## Task

Infer these two aspects directly:

1. `latent_belief_explanation`: {belief_task}
2. `user_profile_modeling`: {profile_task}

Keep the two fields consistent. Do not propose the correct solution in this setting.""",
    }
    examples = {
        "joint": {
            "latent_belief_explanation": "Specific explanation of the user's inferred belief",
            "user_profile_modeling": "Specific model of the relevant user characteristics",
            "correct_resolution": "Actual root cause and concrete resolution",
        },
        "direct_solution": {
            "correct_resolution": "Actual root cause and concrete resolution",
        },
        "cot_solution": {
            "correct_resolution": "Actual root cause and concrete resolution",
        },
        "direct_belief_profile": {
            "latent_belief_explanation": "Specific explanation of the user's inferred belief",
            "user_profile_modeling": "Specific model of the relevant user characteristics",
        },
    }

    return (
        context
        + "\n"
        + tasks[inference_mode]
        + "\n\n## Response format\n\nReturn only one valid JSON object with exactly these fields:\n"
        + json.dumps(examples[inference_mode], indent=2, ensure_ascii=False)
    )


def validate_inference_result(result, inference_mode):
    """Validate the model response against the selected experiment contract."""
    if not isinstance(result, dict):
        raise ValueError("Inference output must be a JSON object")
    expected_fields = set(INFERENCE_MODES[inference_mode]["output_fields"])
    actual_fields = set(result)
    missing = expected_fields - actual_fields
    extra = actual_fields - expected_fields
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields: {sorted(missing)}")
        if extra:
            details.append(f"unexpected fields: {sorted(extra)}")
        raise ValueError("Invalid inference output (" + "; ".join(details) + ")")
    for field in expected_fields:
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"Inference field '{field}' must be a non-empty string")
    return result


def run_inference(
    observation,
    explicit_instruction,
    trajectory,
    client,
    inference_model,
    inference_mode="joint",
    image_id=None,
):
    """Run one of the four inference variants."""
    inference_prompt = build_inference_prompt(
        observation, explicit_instruction, trajectory, inference_mode
    )

    logger.info(f"   🤖 Calling inference model ({inference_model})...")
    # Prepare messages with optional image
    messages = [{"role": "user", "content": inference_prompt}]

    # Retry loop with timeout and JSON extraction error handling
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"   🔄 Attempt {attempt}/{MAX_RETRIES}")

            inference_response = client.chat.completions.create(
                model=inference_model,
                messages=messages,
                temperature=0.0,
                timeout=TIMEOUT_SECONDS
            )

            response_text = inference_response.choices[0].message.content
            logger.debug(f"   Raw inference response: {response_text[:1000]}")

            # Extract JSON from response (handles markdown code blocks and incomplete JSON)
            try:
                result = validate_inference_result(
                    extract_json_from_text(response_text), inference_mode
                )
                logger.info("   ✓ Inference completed")
                return result
            except ValueError as json_error:
                # JSON extraction failed
                logger.warning(f"   ⚠️  JSON extraction failed on attempt {attempt}/{MAX_RETRIES}: {str(json_error)[:100]}")
                if attempt < MAX_RETRIES:
                    # Save the problematic response for debugging
                    logger.debug(f"   Problematic response: {response_text[:300]}")
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.info(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue  # Retry with new API call
                else:
                    logger.error(f"   ✗ Max retries reached - all JSON extraction attempts failed")
                    raise

        except ValueError:
            # Re-raise JSON extraction error (already logged above)
            if attempt >= MAX_RETRIES:
                raise
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                logger.warning(f"   ⏱️  Timeout on attempt {attempt}/{MAX_RETRIES}: {error_msg}")
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.info(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"   ✗ Max retries reached for inference")
                    raise
            else:
                logger.error(f"   ✗ Non-timeout error: {error_msg}")
                raise


def infer_three_dimensions(observation, explicit_instruction, trajectory, client, inference_model, image_id=None):
    """Backward-compatible entry point for the original joint inference mode."""
    return run_inference(
        observation=observation,
        explicit_instruction=explicit_instruction,
        trajectory=trajectory,
        client=client,
        inference_model=inference_model,
        inference_mode="joint",
        image_id=image_id,
    )


def process_instance(instance, client, inference_model, step_num, inference_mode="joint"):
    """Process a single instance with inference."""
    instance_id = instance['id']
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {instance_id}")
    logger.info(f"{'='*70}")

    try:
        # Load trajectory
        logger.info(f"📂 Loading trajectory from {instance['trajectory']}...")
        trajectory = load_trajectory(instance['trajectory'], step_num=step_num)

        # Skip if trajectory file not found
        if trajectory is None:
            logger.warning(f"⏭️  Skipping {instance_id} - trajectory file not found")
            return None, "Trajectory file not found"

        logger.info(f"   ✓ Loaded {len(trajectory)} turns")

        # Run Inference
        logger.info("📝 Running inference...")

        # Check if instance has image
        image_id = instance.get('image', None)
        if image_id:
            logger.info(f"🖼️  Instance has image: {image_id}")

        inference_result = run_inference(
            observation=instance['observation'],
            explicit_instruction=instance['explicit_instruction'],
            trajectory=trajectory,
            client=client,
            inference_model=inference_model,
            inference_mode=inference_mode,
            image_id=image_id
        )

        # Combine original data with inference
        result = {
            "instance_id": instance_id,
            "domain": instance['domain'],
            "observation": instance['observation'],
            "explicit_instruction": instance['explicit_instruction'],
            "trajectory_path": instance['trajectory'],
            "ground_truth": {
                'user_profile': instance['user_profile'],
                'user_latent_belief': instance['user_latent_belief'],
                'true_latent_state': instance['true_latent_state'],
                'root_cause_of_misconception': instance['root_cause_of_misconception']
            },
            "rubrics": instance.get('rubrics', {}),
            "inference": inference_result,
            "inference_mode": inference_mode,
            "inference_dimensions": [
                field for field in DIMENSION_ORDER if field in inference_result
            ],
            "inferred_at": datetime.datetime.now().isoformat()
        }

        logger.info(f"✓ {instance_id} inference completed")

        return result, None

    except Exception as e:
        logger.error(f"✗ Error processing {instance_id}: {str(e)}")
        return None, str(e)


def main():
    """Main inference loop."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run inference on latent belief dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python inference.py -i data/benchmarks/pref-benchmark.json -m gpt-5.6-sol -s 5
  python inference.py -i data/benchmarks/swe-benchmark.json -m gpt-5.6-sol -s 10 --mode direct_solution
  python inference.py -i data/benchmarks/culture-benchmark.json -m gpt-5.6-sol -s 0 --mode cot_solution
        """
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input dataset JSON file'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Name of the inference model exposed by the configured endpoint'
    )
    parser.add_argument(
        '--steps', '-s',
        type=int,
        required=True,
        help='Number of trajectory steps to keep (0 = no trajectory, positive number = first N steps)'
    )
    parser.add_argument(
        '--mode',
        choices=tuple(INFERENCE_MODES),
        default='joint',
        help='Inference variant to run (default: joint)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of concurrent inference requests (default: 4)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=str(inference_dir),
        help='Directory for inference result files (default: inference_results)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (overrides OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='OpenAI API base URL (overrides OPENAI_BASE_URL env var)'
    )
    parser.add_argument(
        '--trajectories-dir',
        type=str,
        default=None,
        help='Base directory for trajectory files (overrides TRAJECTORIES_DIR env var)'
    )

    args = parser.parse_args()
    input_file = args.input
    inference_model = args.model
    step_num = args.steps
    inference_mode = args.mode
    num_workers = args.workers
    if num_workers < 1:
        parser.error('--workers must be at least 1')
    if step_num < 0:
        parser.error('--steps must be 0 or a positive integer')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Override configurations from command line if provided
    global api_key, base_url, TRAJECTORIES_DIR
    if args.api_key:
        api_key = args.api_key
    if args.base_url:
        base_url = args.base_url
    if args.trajectories_dir:
        TRAJECTORIES_DIR = Path(args.trajectories_dir)

    # Create output filename with model name, steps, and timestamp
    # Sanitize model name for filename (replace special characters)
    safe_model_name = inference_model.replace('/', '_').replace(':', '_').replace(' ', '_')
    input_data_name = input_file.split('/')[-1].split('.')[0]

    output_file = output_dir / f"inference_{input_data_name}_{safe_model_name}_{inference_mode}_step{step_num}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Print pipeline information
    logger.info("="*70)
    logger.info("Latent Belief Inference Pipeline")
    logger.info("="*70)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Inference model: {inference_model}")
    logger.info(f"Inference mode: {inference_mode} ({INFERENCE_MODES[inference_mode]['description']})")
    logger.info(f"Trajectory steps: {step_num if step_num > 0 else 'No trajectory (observation & instruction only)'}")
    logger.info(f"Concurrent workers: {num_workers}")
    logger.info("="*70)

    # Load dataset
    logger.info(f"\n📖 Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    logger.info(f"   ✓ Loaded {len(dataset)} instances")

    # Select subset if NUM_INSTANCES_TO_PROCESS is set
    if NUM_INSTANCES_TO_PROCESS is not None:
        dataset = dataset[:NUM_INSTANCES_TO_PROCESS]
        logger.info(f"   ℹ️  Processing only first {len(dataset)} instances")

    # Initialize OpenAI client
    if OpenAI is None:
        raise ModuleNotFoundError(
            "The 'openai' package is required to run inference. "
            "Install dependencies with: pip install -r requirements.txt"
        )
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Initialize results file with empty list
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f)
    logger.info(f"   ✓ Initialized output file: {output_file}")

    # Process instances concurrently. File updates are guarded so that every
    # completed result is persisted immediately without concurrent JSON writes.
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_instances = []
    skipped_instances = []
    completed_count = 0
    file_lock = threading.Lock()

    def process_and_save(instance):
        nonlocal success_count, failed_count, skipped_count, completed_count
        result, error = process_instance(
            instance, client, inference_model, step_num, inference_mode
        )
        with file_lock:
            completed_count += 1
            if result:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                results.append(result)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                success_count += 1
                logger.info(f"💾 Saved result for {instance['id']}")
            elif error == "Trajectory file not found":
                skipped_count += 1
                skipped_instances.append(instance['id'])
            else:
                failed_count += 1
                failed_instances.append({
                    "instance_id": instance['id'],
                    "error": error
                })
            logger.info(f"Progress: {completed_count}/{len(dataset)} completed")

    logger.info(f"\n🚀 Starting concurrent inference with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_instance = {
            executor.submit(process_and_save, instance): instance
            for instance in dataset
        }
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                future.result()
            except Exception as error:
                logger.error(f"✗ Instance {instance['id']} raised: {error}")
                with file_lock:
                    failed_count += 1
                    failed_instances.append({
                        "instance_id": instance['id'],
                        "error": str(error)
                    })

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("INFERENCE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"✓ Successful: {success_count}/{len(dataset)}")
    logger.info(f"⏭️  Skipped (no trajectory): {skipped_count}/{len(dataset)}")
    logger.info(f"✗ Failed: {failed_count}/{len(dataset)}")

    if skipped_instances:
        logger.info(f"\nSkipped instances (trajectory not found):")
        for instance_id in skipped_instances:
            logger.info(f"  - {instance_id}")

    if failed_instances:
        logger.warning(f"\nFailed instances:")
        for fail in failed_instances:
            logger.warning(f"  - {fail['instance_id']}: {fail['error']}")

    logger.info(f"\n📊 Inference results saved to: {output_file}")
    logger.info(f"   Use this file as input for evaluation.py to score the inferences")


if __name__ == "__main__":
    main()
