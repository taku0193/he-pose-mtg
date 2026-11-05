import csv

from hepose.adapters import Prediction
from hepose.benchmark import make_plan
from hepose.pipeline import run_video
from hepose.schema import TemporalContext


def test_shared_frames_are_selected_after_common_warmup():
    plan = make_plan(375, ("yolo26", "detrpose", "tar-vitpose"))
    assert list(plan.warmup_frames) == list(range(2, 22))
    assert list(plan.measured_frames) == list(range(22, 122))


def test_short_measurement_does_not_trim_video_or_turn_missing_times_into_zero(
    sample_video, tmp_path
):
    class TemporalAdapter:
        model_id = "tar-vitpose"
        temporal = TemporalContext(5, 1, 2)

        def predict(self, frames, target_frame_index):
            assert [f.frame_index for f in frames] == list(
                range(target_frame_index - 2, target_frame_index + 3)
            )
            return Prediction((), None, None)

        def environment(self):
            return {}

        def close(self):
            pass

    plan = make_plan(8, ("yolo26", "tar-vitpose"), warmup=2)
    out = tmp_path / "short"
    summary = run_video(
        sample_video, out, TemporalAdapter, model_id="tar-vitpose", measurement=plan
    )
    measured = summary["details"]["benchmark"]
    assert summary["frame_counts"] == {"success": 4, "failed": 0, "not_estimated": 4}
    assert measured["status"] == "insufficient_samples"
    assert measured["frame_indices"] == [4, 5]
    assert measured["frames_without_people"] == 2
    assert measured["timing"]["inference_ms"]["p50_ms"] is None
    with (out / "timings.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["frame_index"]) for row in rows] == list(range(8))
