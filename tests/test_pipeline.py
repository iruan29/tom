import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluate
import inference


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        message = SimpleNamespace(content=response)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class ConcurrentCompletions:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def create(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        message = SimpleNamespace(
            content=json.dumps({"correct_resolution": "solution"})
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ConcurrentJudgeCompletions(ConcurrentCompletions):
    def create(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        message = SimpleNamespace(content=json.dumps({
            "correct_resolution": {
                "criterion_scores": {"criterion_1": 1},
                "feedback": "criterion passed",
            }
        }))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class InferenceContractTests(unittest.TestCase):
    @staticmethod
    def make_dataset(temp_path, count):
        trajectory_path = temp_path / "trajectory.json"
        trajectory_path.write_text('{"trajectory": []}', encoding="utf-8")
        dataset = []
        for index in range(count):
            dataset.append({
                "id": f"test_{index}",
                "domain": "test",
                "observation": "observation",
                "explicit_instruction": "instruction",
                "trajectory": str(trajectory_path),
                "user_profile": "profile",
                "user_latent_belief": "belief",
                "true_latent_state": "state",
                "root_cause_of_misconception": "cause",
                "rubrics": {
                    "correct_resolution": [{"criterion": "criterion"}]
                },
            })
        dataset_path = temp_path / "dataset.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
        return dataset_path, dataset

    def test_organized_data_resolves_legacy_trajectory_reference(self):
        with patch.object(inference, "TRAJECTORIES_DIR", Path("data/trajectories")):
            trajectory = inference.load_trajectory(
                "trajectories/culture-benchmark/culture_0010.json", step_num=5
            )
        self.assertIsNotNone(trajectory)
        self.assertLessEqual(len(trajectory), 5)

    def test_each_mode_has_the_expected_output_contract(self):
        outputs = {
            "joint": {
                "latent_belief_explanation": "belief",
                "user_profile_modeling": "profile",
                "correct_resolution": "solution",
            },
            "direct_solution": {"correct_resolution": "solution"},
            "cot_solution": {"correct_resolution": "solution"},
            "direct_belief_profile": {
                "latent_belief_explanation": "belief",
                "user_profile_modeling": "profile",
            },
        }
        for mode, output in outputs.items():
            with self.subTest(mode=mode):
                client = FakeClient([json.dumps(output)])
                result = inference.run_inference(
                    observation="observation",
                    explicit_instruction="instruction",
                    trajectory=[],
                    client=client,
                    inference_model="fake-model",
                    inference_mode=mode,
                )
                self.assertEqual(result, output)
                prompt = client.chat.completions.calls[0]["messages"][0]["content"]
                self.assertIn("No behavioral trajectory is provided", prompt)

    def test_cot_prompt_only_adds_think_instruction_to_direct_prompt(self):
        direct_prompt = inference.build_inference_prompt(
            "observation", "instruction", [], "direct_solution"
        )
        cot_prompt = inference.build_inference_prompt(
            "observation", "instruction", [], "cot_solution"
        )
        expected_cot_prompt = direct_prompt.replace(
            "\n\n## Response format",
            f"\n\n{inference.COT_INSTRUCTION}\n\n## Response format",
            1,
        )
        self.assertEqual(cot_prompt, expected_cot_prompt)

    def test_cot_contract_rejects_separate_reasoning_field(self):
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            inference.validate_inference_result(
                {
                    "reasoning": "reasoning trace",
                    "correct_resolution": "solution",
                },
                "cot_solution",
            )

    def test_mode_contract_rejects_extra_dimensions(self):
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            inference.validate_inference_result(
                {
                    "correct_resolution": "solution",
                    "user_profile_modeling": "profile",
                },
                "direct_solution",
            )

    def test_main_runs_multiple_requests_concurrently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path, dataset = self.make_dataset(temp_path, 4)
            completions = ConcurrentCompletions()
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )
            argv = [
                "inference.py",
                "--input", str(dataset_path),
                "--model", "fake-model",
                "--steps", "0",
                "--mode", "direct_solution",
                "--workers", "4",
                "--output-dir", str(temp_path / "results"),
            ]
            with patch.object(inference, "OpenAI", return_value=fake_client), patch(
                "sys.argv", argv
            ):
                inference.main()

            self.assertGreater(completions.max_active, 1)
            output_files = list((temp_path / "results").glob("*.json"))
            self.assertEqual(len(output_files), 1)
            self.assertEqual(len(json.loads(output_files[0].read_text())), 4)

    def test_resume_requests_only_missing_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path, dataset = self.make_dataset(temp_path, 4)
            output_dir = temp_path / "results"
            output_dir.mkdir()
            prefix = inference.experiment_file_prefix(
                dataset_path, "fake-model", "direct_solution", 0
            )
            output_path = output_dir / f"{prefix}.json"
            existing = {
                "instance_id": dataset[0]["id"],
                "inference_mode": "direct_solution",
                "inference": {"correct_resolution": "existing solution"},
            }
            output_path.write_text(json.dumps([existing]), encoding="utf-8")

            client = FakeClient([
                json.dumps({"correct_resolution": f"solution {index}"})
                for index in range(1, 4)
            ])
            argv = [
                "inference.py",
                "--input", str(dataset_path),
                "--model", "fake-model",
                "--steps", "0",
                "--mode", "direct_solution",
                "--workers", "1",
                "--output-dir", str(output_dir),
                "--resume",
            ]
            with patch.object(inference, "OpenAI", return_value=client), patch(
                "sys.argv", argv
            ):
                inference.main()

            self.assertEqual(len(client.chat.completions.calls), 3)
            results = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [result["instance_id"] for result in results],
                [instance["id"] for instance in dataset],
            )

            with patch.object(inference, "OpenAI") as openai_constructor, patch(
                "sys.argv", argv
            ):
                inference.main()
            openai_constructor.assert_not_called()

    def test_failed_run_exits_nonzero_and_resume_fills_the_gap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path, dataset = self.make_dataset(temp_path, 3)
            output_dir = temp_path / "results"
            argv = [
                "inference.py",
                "--input", str(dataset_path),
                "--model", "fake-model",
                "--steps", "0",
                "--mode", "direct_solution",
                "--workers", "1",
                "--output-dir", str(output_dir),
                "--resume",
            ]
            first_client = FakeClient([
                json.dumps({"correct_resolution": "solution 0"}),
                RuntimeError("simulated request failure"),
                json.dumps({"correct_resolution": "solution 2"}),
            ])
            with patch.object(inference, "OpenAI", return_value=first_client), patch(
                "sys.argv", argv
            ), self.assertRaisesRegex(SystemExit, "1"):
                inference.main()

            prefix = inference.experiment_file_prefix(
                dataset_path, "fake-model", "direct_solution", 0
            )
            output_path = output_dir / f"{prefix}.json"
            partial = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [result["instance_id"] for result in partial],
                [dataset[0]["id"], dataset[2]["id"]],
            )

            retry_client = FakeClient([
                json.dumps({"correct_resolution": "retried solution"})
            ])
            with patch.object(inference, "OpenAI", return_value=retry_client), patch(
                "sys.argv", argv
            ):
                inference.main()

            self.assertEqual(len(retry_client.chat.completions.calls), 1)
            completed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [result["instance_id"] for result in completed],
                [instance["id"] for instance in dataset],
            )


class EvaluationContractTests(unittest.TestCase):
    rubrics = {
        "latent_belief_explanation": [{"criterion": "belief criterion"}],
        "user_profile_modeling": [{"criterion": "profile criterion"}],
        "correct_resolution": [
            {"criterion": "resolution criterion 1"},
            {"criterion": "resolution criterion 2"},
        ],
    }

    def test_solution_mode_evaluates_only_resolution(self):
        instance = {
            "inference_mode": "cot_solution",
            "inference_dimensions": ["correct_resolution"],
            "inference": {
                "reasoning": "rationale",
                "correct_resolution": "solution",
            },
            "rubrics": self.rubrics,
        }
        self.assertEqual(
            evaluate.get_evaluation_dimensions(instance), ["correct_resolution"]
        )
        prompt = evaluate.build_judge_prompt(
            instance["inference"], self.rubrics, ["correct_resolution"]
        )
        self.assertNotIn("reasoning", prompt)
        self.assertNotIn("Chain of Thought Required", prompt)
        self.assertNotIn("latent_belief_explanation", prompt)

    def test_legacy_joint_output_is_auto_detected(self):
        instance = {
            "inference": {
                "latent_belief_explanation": "belief",
                "user_profile_modeling": "profile",
                "correct_resolution": "solution",
            },
            "rubrics": self.rubrics,
        }
        self.assertEqual(
            evaluate.get_evaluation_dimensions(instance),
            list(evaluate.DIMENSION_ORDER),
        )

    def test_judge_totals_are_recalculated(self):
        raw = {
            "correct_resolution": {
                "criterion_scores": {
                    "criterion_1": "1",
                    "criterion_2": 0,
                },
                "total_score": 99,
                "max_score": 99,
                "feedback": "one criterion passed",
            },
            "overall_summary": {"total_score": 99},
        }
        normalized = evaluate.normalize_judge_result(
            raw,
            self.rubrics,
            ["correct_resolution"],
            json.dumps(raw),
        )
        self.assertEqual(normalized["correct_resolution"]["total_score"], 1)
        self.assertEqual(normalized["correct_resolution"]["max_score"], 2)
        self.assertEqual(normalized["overall_summary"]["percentage"], 50.0)

    def test_summary_append_preserves_existing_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            evaluate.append_summary(summary_path, {"run": 1})
            evaluate.append_summary(summary_path, {"run": 2})
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")),
                [{"run": 1}, {"run": 2}],
            )

    def test_main_persists_concurrent_results_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset = []
            for index in range(4):
                dataset.append({
                    "instance_id": f"test_{index}",
                    "domain": "test",
                    "inference_mode": "direct_solution",
                    "inference_dimensions": ["correct_resolution"],
                    "inference": {"correct_resolution": "solution"},
                    "rubrics": {
                        "correct_resolution": [{"criterion": "criterion"}]
                    },
                })
            dataset_path = temp_path / "inference_test.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            completions = ConcurrentJudgeCompletions()
            fake_client = SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )
            argv = [
                "evaluate.py",
                "--input", str(dataset_path),
                "--workers", "4",
                "--judge-model", "fake-model",
            ]
            with patch.object(evaluate, "OpenAI", return_value=fake_client), patch.object(
                evaluate, "eval_dir", temp_path / "results"
            ), patch.object(
                evaluate, "append_summary"
            ), patch("sys.argv", argv):
                evaluate.main()

            self.assertGreater(completions.max_active, 1)
            output_files = list((temp_path / "results").glob("*.json"))
            self.assertEqual(len(output_files), 1)
            results = json.loads(output_files[0].read_text(encoding="utf-8"))
            self.assertEqual(len(results), 4)
            self.assertEqual(len({result["instance_id"] for result in results}), 4)


if __name__ == "__main__":
    unittest.main()
