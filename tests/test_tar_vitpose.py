"""Model-free contracts for TAR joint semantics, crops and five-frame windows."""

import numpy as np
import pytest

from hepose.adapters import InputFrame
from hepose.tar_vitpose import decode_person, validate_window


def test_tar_preserves_head_semantics_and_restores_crop_coordinates():
    heatmaps = np.zeros((1, 17, 4, 3), dtype=np.float32)
    heatmaps[:, :, 2, 1] = 1
    person = decode_person(heatmaps, (10, 20, 70, 100), (20, 30, 60, 90))
    common = {point.name: point for point in person.keypoints}
    native = {point.name: point for point in person.native_keypoints}
    assert (common["nose"].x, common["nose"].y) == (30, 60)
    assert (native["head_bottom"].x, native["head_bottom"].y) == (30, 60)
    assert native["head_top"].x == 30
    assert common["left_eye"].x is None and common["right_eye"].x is None
    assert common["left_ear"].x is None and common["right_ear"].x is None
    assert person.crop_xyxy == (10, 20, 70, 100)
    assert person.bbox == (20, 30, 60, 90)
    assert all(p.confidence is None and p.visibility is None for p in person.native_keypoints)


@pytest.mark.parametrize("shape", [(17, 4, 3), (1, 15, 4, 3), (2, 17, 4, 3), (1, 17, 0, 3)])
def test_tar_rejects_wrong_heatmap_shape(shape):
    with pytest.raises(ValueError):
        decode_person(np.zeros(shape), (10, 20, 70, 100), (20, 30, 60, 90))


def test_tar_rejects_nonfinite_heatmaps_and_invalid_crops():
    heatmaps = np.zeros((1, 17, 4, 3))
    heatmaps[0, 0, 1, 1] = float("nan")
    with pytest.raises(ValueError):
        decode_person(heatmaps, (10, 20, 70, 100), (20, 30, 60, 90))
    with pytest.raises(ValueError):
        decode_person(np.zeros_like(heatmaps), (10, 20, 10, 100), (20, 30, 60, 90))


def test_tar_window_is_centered_and_never_repeats_boundary_frames():
    frames = [InputFrame(i, i * 40, 96, 64, np.zeros((64, 96, 3))) for i in range(5)]
    assert validate_window(frames, 2).frame_index == 2
    for bad in (frames[:4], [frames[0], *frames[:4]], frames[::-1]):
        with pytest.raises(ValueError):
            validate_window(bad, 2)
    with pytest.raises(ValueError):
        validate_window(frames, 3)


def test_tar_configuration_resolves_all_three_weight_paths(tmp_path):
    from hepose.config import load_config

    path = tmp_path / "local.toml"
    path.write_text('''schema_version = 1
[models.tar-vitpose]
weights = "models/tar.pt"
pretrained = "models/pretrain.pth"
detector_weights = "models/yolo11x.pt"
repository = "third_party/TARViTPose"
''')
    settings = load_config(path).model("tar-vitpose")
    assert settings["pretrained"] == str(tmp_path / "models/pretrain.pth")
    assert settings["detector_weights"] == str(tmp_path / "models/yolo11x.pt")
