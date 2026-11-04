"""Pure conversion contracts: no model imports, weights, CUDA or network."""

import numpy as np
import pytest

from hepose.detrpose import parse_result


def result_fixture(count=1):
    points = np.zeros((count, 17, 2))
    points[..., 0] = 30
    points[..., 1] = 40
    return {
        "scores": np.full(count, 0.9),
        "labels": np.ones(count, dtype=int),
        "keypoints": points,
        "original_size": (96, 128),
    }


def test_detrpose_coordinates_are_pixels_and_unavailable_fields_are_null():
    prediction = parse_result(result_fixture(), 96, 128)
    person = prediction.people[0]
    assert person.bbox is None and person.track_id is None
    point = person.keypoints[0]
    assert point.name == "nose" and (point.x, point.y) == (30, 40)
    assert point.confidence is None and point.visibility is None
    assert prediction.inference_ms is None and prediction.measurement_method is None


def test_detrpose_thresholds_people_without_creating_joint_confidence():
    result = result_fixture(3)
    result["scores"] = np.array([0.5, 0.5001, 0.9])
    prediction = parse_result(result, 96, 128, inference_ms=2.5, method="test clock")
    assert len(prediction.people) == 2
    assert prediction.inference_ms == 2.5 and prediction.measurement_method == "test clock"
    assert all(p.confidence is None for person in prediction.people for p in person.keypoints)


def test_detrpose_empty_and_multiple_people():
    assert parse_result(result_fixture(0), 96, 128).people == ()
    assert len(parse_result(result_fixture(2), 96, 128).people) == 2


@pytest.mark.parametrize("failure", ["joints", "count", "dimensions", "shape", "nan", "score"])
def test_invalid_detrpose_results_fail_instead_of_becoming_zero_people(failure):
    result = result_fixture()
    if failure == "joints":
        result["keypoints"] = np.zeros((1, 16, 2))
    elif failure == "count":
        result["scores"] = np.ones(2)
    elif failure == "dimensions":
        result["keypoints"] = np.ones((1, 17, 3))
    elif failure == "shape":
        result["original_size"] = (128, 96)
    elif failure == "nan":
        result["keypoints"][0, 0, 0] = float("nan")
    else:
        result["scores"][0] = float("nan")
    with pytest.raises(ValueError):
        parse_result(result, 96, 128)
