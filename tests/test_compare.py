import json

import pytest

from hepose.adapters import Prediction
from hepose.compare import compare_video
from hepose.pipeline import run_video
from hepose.schema import TemporalContext
from hepose.video import VideoReader


def create_run(source, out, model, temporal):
    class Adapter:
        model_id = model

        def predict(self, frames, target_frame_index):
            return Prediction((), None, None)

        def environment(self):
            return {}

        def close(self):
            pass

    adapter = Adapter()
    adapter.temporal = temporal
    run_video(source, out, lambda: adapter, model_id=model)


@pytest.fixture
def runs(sample_video, tmp_path):
    root = tmp_path / "runs"
    create_run(sample_video, root / "tar", "tar-vitpose", TemporalContext(5, 1, 2))
    create_run(sample_video, root / "yolo", "yolo26", TemporalContext())
    return root


def test_comparison_keeps_full_timeline_and_tars_unestimated_boundaries(
    runs, tmp_path, monkeypatch
):
    from hepose import compare

    statuses = []
    draw = compare.draw_poses

    def track(image, result):
        statuses.append((result.model_id, result.frame_index, result.status))
        return draw(image, result)

    monkeypatch.setattr(compare, "draw_poses", track)
    output = tmp_path / "comparison.mp4"
    compare_video(runs, output)
    with VideoReader(output) as reader:
        assert reader.fps == 25
        assert len(list(reader)) == 8
    assert [(index, status) for model, index, status in statuses if model == "tar-vitpose"] == [
        (i, "not_estimated" if i in (0, 1, 6, 7) else "success") for i in range(8)
    ]


def test_duplicate_model_refused_before_creating_video(runs, sample_video, tmp_path):
    create_run(sample_video, runs / "second-yolo", "yolo26", TemporalContext())
    output = tmp_path / "comparison.mp4"
    with pytest.raises(ValueError, match="yolo26"):
        compare_video(runs, output)
    assert not output.exists()


def test_corrupted_frame_identity_cannot_produce_comparison(runs, tmp_path):
    poses = runs / "tar" / "poses.jsonl"
    lines = poses.read_text().splitlines()
    row = json.loads(lines[3])
    lines[3] = json.dumps(dict(row, run_id="another-run"))
    poses.write_text("\n".join(lines) + "\n")
    output = tmp_path / "comparison.mp4"
    with pytest.raises(ValueError, match="run_id"):
        compare_video(runs, output)
    assert not output.exists()


def test_wrong_original_video_is_refused(runs, sample_video, tmp_path):
    wrong = tmp_path / "wrong.mp4"
    wrong.write_bytes(sample_video.read_bytes() + b"different")
    output = tmp_path / "comparison.mp4"
    with pytest.raises(ValueError, match="SHA256"):
        compare_video(runs, output, source=wrong)
    assert not output.exists()
