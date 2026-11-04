import hashlib
import json

import pytest

from hepose.adapters import Prediction
from hepose.pipeline import run_video
from hepose.schema import Keypoint, Person, TemporalContext, coco17
from hepose.video import VideoReader


class TestOnlyAdapter:
    __test__ = False
    model_id = "test-stub"
    temporal = TemporalContext()

    def __init__(self, temporal=None, fail_at=None, interrupt_at=None):
        self.temporal = temporal or self.temporal
        self.fail_at = fail_at
        self.interrupt_at = interrupt_at
        self.calls = []
        self.closed = False

    def environment(self):
        return {"purpose": "test_only", "weights": [], "backend": "test_only"}

    def predict(self, frames, target_frame_index):
        self.calls.append((target_frame_index, [frame.frame_index for frame in frames]))
        if target_frame_index == self.fail_at:
            raise RuntimeError("test inference failure")
        if target_frame_index == self.interrupt_at:
            raise KeyboardInterrupt
        points = coco17((Keypoint("left_shoulder", 30, 30), Keypoint("left_elbow", 40, 40)))
        people = () if target_frame_index == 0 else (Person((20, 20, 80, 60), points),)
        return Prediction(people, None, None)

    def close(self):
        self.closed = True


def test_pipeline_writes_all_six_outputs(sample_video, tmp_path):
    output = tmp_path / "result"
    adapter = TestOnlyAdapter()
    summary = run_video(sample_video, output, lambda: adapter, model_id=adapter.model_id)
    assert summary["status"] == "success"
    assert summary["details"]["first_adapter_call_ms"] >= 0
    assert {path.name for path in output.iterdir()} == {
        "annotated.mp4",
        "poses.jsonl",
        "timings.csv",
        "summary.json",
        "environment.json",
        "run.log",
    }
    poses = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]
    assert [row["frame_index"] for row in poses] == list(range(8))
    assert poses[0]["people"] == [] and poses[0]["status"] == "success"
    assert [row["output_index"] for row in poses] == list(range(8))
    environment = json.loads((output / "environment.json").read_text())
    assert environment["input"]["sha256"] == hashlib.sha256(sample_video.read_bytes()).hexdigest()
    assert environment["adapter"]["purpose"] == "test_only"
    assert environment["requested_device"] == "cuda:0"
    assert environment["device"] is None, "A test adapter must not claim it used the requested GPU"
    with VideoReader(output / "annotated.mp4") as reader:
        assert reader.fps == 25
        assert len(list(reader)) == 8
    assert adapter.closed


def test_temporal_boundaries_preserve_input_timeline(sample_video, tmp_path):
    adapter = TestOnlyAdapter(TemporalContext(5, 1, 2))
    output = tmp_path / "temporal"
    summary = run_video(sample_video, output, lambda: adapter, model_id=adapter.model_id)
    rows = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == [
        "not_estimated",
        "not_estimated",
        "success",
        "success",
        "success",
        "success",
        "not_estimated",
        "not_estimated",
    ]
    assert rows[2]["reference_frame_indices"] == [0, 1, 2, 3, 4]
    assert rows[5]["reference_frame_indices"] == [3, 4, 5, 6, 7]
    assert rows[2]["source_time_ms"] == 80
    assert rows[2]["output_index"] == 0
    assert rows[0]["output_index"] is None
    assert summary["frame_counts"] == {"success": 4, "failed": 0, "not_estimated": 4}
    with VideoReader(output / "annotated.mp4") as reader:
        assert len(list(reader)) == 8


@pytest.mark.parametrize("kind", ["failure", "interrupt"])
def test_errors_preserve_logs_and_release_resources(sample_video, tmp_path, kind):
    adapter = TestOnlyAdapter(fail_at=3) if kind == "failure" else TestOnlyAdapter(interrupt_at=3)
    output = tmp_path / kind
    exception = RuntimeError if kind == "failure" else KeyboardInterrupt
    with pytest.raises(exception):
        run_video(sample_video, output, lambda: adapter, model_id=adapter.model_id)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == ("failed" if kind == "failure" else "interrupted")
    assert adapter.closed
    assert (output / "run.log").stat().st_size > 0
    rows = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]
    if kind == "failure":
        assert rows[-1]["frame_index"] == 3 and rows[-1]["status"] == "failed"


def test_factory_failure_has_no_fake_video(sample_video, tmp_path):
    def missing_weights():
        raise FileNotFoundError("weights not prepared")

    output = tmp_path / "missing-weights"
    with pytest.raises(FileNotFoundError):
        run_video(sample_video, output, missing_weights, model_id="yolo26")
    assert not (output / "annotated.mp4").exists()
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"


def test_existing_results_are_protected_before_loading_adapter(sample_video, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "marker").write_text("keep")
    adapter = TestOnlyAdapter()
    with pytest.raises(FileExistsError):
        run_video(sample_video, output, lambda: adapter, model_id=adapter.model_id)
    assert not adapter.closed and not adapter.calls
    assert (output / "marker").read_text() == "keep"


def test_input_read_failure_still_writes_failure_log(sample_video, tmp_path, monkeypatch):
    def unreadable(*args):
        raise OSError("input read failed")

    monkeypatch.setattr("hepose.environment.hashlib.file_digest", unreadable)
    output = tmp_path / "read-error"
    with pytest.raises((OSError, RuntimeError)):
        run_video(sample_video, output, TestOnlyAdapter, model_id="test-stub")
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
    assert (output / "run.log").is_file()


def test_video_without_any_complete_window_is_not_success(sample_video, tmp_path):
    adapter = TestOnlyAdapter(TemporalContext(9, 1, 4))
    output = tmp_path / "short"
    with pytest.raises(RuntimeError, match="中心フレーム"):
        run_video(sample_video, output, lambda: adapter, model_id=adapter.model_id)
    rows = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert all(row["status"] == "not_estimated" for row in rows)
    assert not adapter.calls and adapter.closed


@pytest.mark.parametrize(
    "temporal,expected",
    [
        (TemporalContext(3, 2, 2), [0, 2, 4]),
        (TemporalContext(3, 1, 0), [0, 1, 2]),
    ],
)
def test_strided_and_causal_windows_use_source_indices(sample_video, tmp_path, temporal, expected):
    adapter = TestOnlyAdapter(temporal)
    run_video(sample_video, tmp_path / "window", lambda: adapter, model_id=adapter.model_id)
    assert adapter.calls[0][1] == expected


def test_video_handles_close_on_interrupt(sample_video, tmp_path, monkeypatch):
    import hepose.pipeline as pipeline

    readers = []

    class TrackedReader(VideoReader):
        def __init__(self, path):
            super().__init__(path)
            readers.append(self)

    monkeypatch.setattr(pipeline, "VideoReader", TrackedReader)
    adapter = TestOnlyAdapter(interrupt_at=3)
    with pytest.raises(KeyboardInterrupt):
        run_video(sample_video, tmp_path / "interrupt", lambda: adapter, model_id=adapter.model_id)
    assert readers and all(reader.closed for reader in readers)
    assert all(not reader.capture.isOpened() for reader in readers)
