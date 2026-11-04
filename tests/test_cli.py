import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        return subprocess.run(
            [sys.executable, "-m", "hepose.cli", *args],
            env=env, text=True, capture_output=True, check=False,
        )

    def test_help_and_subcommand_help_without_model_packages(self):
        for args in [(), ("doctor",), ("run",), ("live",), ("benchmark",), ("compare",)]:
            with self.subTest(args=args):
                result = self.run_cli(*args, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("torch", result.stderr)

    def test_invalid_model_and_missing_input_are_usage_errors(self):
        result = self.run_cli("run", "--model", "unknown")
        self.assertEqual(result.returncode, 2)

    def test_client_doctor_reports_separate_checks_without_cuda(self):
        result = self.run_cli("doctor", "--role", "client", "--json")
        self.assertIn(result.returncode, (0, 1))
        data = json.loads(result.stdout)
        self.assertEqual(data["role"], "client")
        self.assertNotIn("cuda", data["checks"])
        self.assertEqual(data["camera_opened"], False)

    def test_worker_doctor_reports_unimplemented_models_honestly(self):
        result = self.run_cli("doctor", "--model", "all", "--device", "cuda:0", "--json")
        self.assertEqual(result.returncode, 1)
        data = json.loads(result.stdout)
        self.assertEqual(set(data["models"]), {"yolo26", "detrpose", "tar-vitpose"})
        self.assertTrue(all(item["status"] == "not_implemented" for item in data["models"].values()))

    def test_unimplemented_run_is_not_reported_as_inference_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.mp4"
            source.write_bytes(b"not-a-real-video")
            output = Path(tmp) / "results"
            result = self.run_cli(
                "run", "--model", "yolo26", "--input", str(source),
                "--output", str(output), "--device", "cuda:0",
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("未実装", result.stderr)
            self.assertFalse((output / "annotated.mp4").exists())


if __name__ == "__main__":
    unittest.main()
