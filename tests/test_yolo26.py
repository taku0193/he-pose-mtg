"""Model-free tests: no Ultralytics import, checkpoints, CUDA allocation or downloads."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from hepose.yolo26 import parse_result, validate_runtime_device, verify_checkpoint


def result_fixture(count=1, channels=3):
    points = np.ones((count, 17, channels), dtype=float)
    points[..., 0] = 30
    points[..., 1] = 40
    if channels == 3:
        points[..., 2] = 0.9
    boxes = np.array([[10, 20, 80, 100]] * count, dtype=float).reshape(count, 4)
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=boxes),
        keypoints=SimpleNamespace(data=points),
        orig_shape=(128, 96),
        speed={"inference": 2.5},
    )


def test_result_coordinates_are_original_pixels_and_visibility_is_unknown():
    prediction = parse_result(result_fixture(), 96, 128)
    assert prediction.people[0].bbox == (10, 20, 80, 100)
    assert prediction.people[0].keypoints[0].x == 30
    assert prediction.people[0].keypoints[0].y == 40
    assert prediction.people[0].keypoints[0].visibility is None
    assert prediction.people[0].track_id is None


def test_zero_and_multiple_people_are_valid_results():
    assert parse_result(result_fixture(0), 96, 128).people == ()
    assert len(parse_result(result_fixture(2), 96, 128).people) == 2


def test_low_confidence_coordinates_are_missing_without_claiming_invisibility():
    result = result_fixture()
    result.keypoints.data[0, 1] = (0, 0, 0.1)
    point = parse_result(result, 96, 128).people[0].keypoints[1]
    assert point.x is None and point.y is None
    assert point.confidence == 0.1 and point.visibility is None


def test_absent_confidence_and_model_time_stay_null():
    result = result_fixture(channels=2)
    result.speed = {}
    prediction = parse_result(result, 96, 128)
    assert prediction.inference_ms is None
    assert prediction.people[0].keypoints[0].confidence is None


@pytest.mark.parametrize("failure", ["joints", "count", "shape", "nan"])
def test_malformed_model_output_is_not_zero_person_success(failure):
    result = result_fixture()
    if failure == "joints":
        result.keypoints.data = np.zeros((1, 16, 3))
    elif failure == "count":
        result.keypoints.data = np.zeros((2, 17, 3))
    elif failure == "shape":
        result.orig_shape = (256, 192)
    else:
        result.keypoints.data[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        parse_result(result, 96, 128)


def test_weights_must_exist_and_match_the_expected_fingerprint(tmp_path):
    path = tmp_path / "weights.pt"
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(path)
    path.write_bytes(b"test-only bytes")
    with pytest.raises(ValueError, match="SHA256"):
        verify_checkpoint(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    info = verify_checkpoint(path, expected_sha256=expected)
    assert info["sha256"] == expected and info["size"] == len(b"test-only bytes")


def test_unavailable_cuda_never_falls_back_to_cpu():
    runtime = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_runtime_device("cuda:0", runtime)


def test_requested_gpu_uuid_must_match():
    properties = SimpleNamespace(name="RTX 3090", uuid="GPU-A", total_memory=24576 * 1024**2)
    runtime = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: properties,
        )
    )
    with pytest.raises(RuntimeError, match="UUID"):
        validate_runtime_device("cuda:0", runtime, "GPU-B")
    assert validate_runtime_device("cuda:0", runtime, "GPU-A")["gpu_uuid"] == "GPU-A"


def test_explicit_cpu_is_recorded_without_gpu_claims():
    info = validate_runtime_device("cpu", SimpleNamespace())
    assert info["device"] == "cpu" and info["gpu_uuid"] is None
