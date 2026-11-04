import csv
import json
import tempfile
import unittest
from pathlib import Path

from hepose.results import RunWriter
from hepose.schema import FrameResult, Person
from hepose.timing import measure


class ResultTests(unittest.TestCase):
    def test_no_inference_frames_cannot_report_a_successful_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with RunWriter(output, "r", {}):
                pass
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["frame_counts"]["success"], 0)

    def test_jsonl_csv_and_metadata_keep_zero_and_multiple_people(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with RunWriter(output, "r", {"git_commit": "example", "purpose": "test"}) as writer:
                writer.write(FrameResult("r", "yolo26", 0, 0, 32, 32))
                writer.write(FrameResult(
                    "r", "yolo26", 1, 40, 32, 32,
                    people=(Person((0, 0, 10, 10)), Person((12, 12, 30, 30))),
                ), {"inference_ms": 2.5})
            rows = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]
            self.assertEqual([len(row["people"]) for row in rows], [0, 2])
            with (output / "timings.csv").open(newline="") as stream:
                timings = list(csv.DictReader(stream))
            self.assertEqual(timings[0]["inference_ms"], "")
            self.assertEqual(timings[1]["inference_ms"], "2.5")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["frame_counts"]["success"], 2)
            self.assertEqual(summary["status"], "success")
            self.assertNotIn("annotated.mp4", summary["artifacts"])
            self.assertEqual(json.loads((output / "environment.json").read_text())["purpose"], "test")

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            output.mkdir()
            marker = output / "summary.json"
            marker.write_text("existing result")
            with self.assertRaises(FileExistsError):
                with RunWriter(output, "r", {}):
                    pass
            self.assertEqual(marker.read_text(), "existing result")

    def test_failure_and_interrupt_are_recorded_and_files_close(self):
        for exception, status in [(RuntimeError("inference failed"), "failed"),
                                  (KeyboardInterrupt(), "interrupted")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "run"
                with self.assertRaises(type(exception)):
                    with RunWriter(output, "r", {}) as writer:
                        writer.write(FrameResult("r", "yolo26", 0, 0, 32, 32))
                        raise exception
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["status"], status)
                self.assertEqual(summary["last_frame_index"], 0)
                self.assertTrue(writer.closed)

    def test_wrong_run_id_cannot_be_mixed_into_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                with RunWriter(Path(tmp) / "run", "a", {}) as writer:
                    writer.write(FrameResult("b", "yolo26", 0, 0, 32, 32))

    def test_synchronized_timer_covers_exception_cleanup(self):
        calls = []
        clock = iter([1_000_000, 4_000_000])
        durations = {}
        with self.assertRaises(RuntimeError):
            with measure(durations, "model_ms", lambda: calls.append("sync"), lambda: next(clock)):
                raise RuntimeError("model failed")
        self.assertEqual(calls, ["sync", "sync"])
        self.assertEqual(durations["model_ms"], 3)


if __name__ == "__main__":
    unittest.main()
