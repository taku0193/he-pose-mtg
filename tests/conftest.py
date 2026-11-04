"""Synthetic codec fixtures only; these are not pose-estimation results."""

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 64))
    assert writer.isOpened(), "The test environment must support MP4 encoding"
    try:
        for index in range(8):
            image = np.zeros((64, 96, 3), dtype=np.uint8)
            image[30:40, 10 + index : 20 + index] = 200
            writer.write(image)
    finally:
        writer.release()
    return path
