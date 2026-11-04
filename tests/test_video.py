import cv2
import numpy as np
import pytest

from hepose.render import draw_poses
from hepose.schema import FrameResult, Keypoint, Person, coco17
from hepose.video import VideoError, VideoReader, VideoWriter


def test_video_preserves_indices_time_size_and_frame_count(sample_video, tmp_path):
    with VideoReader(sample_video) as reader:
        assert (reader.width, reader.height, reader.fps) == (96, 64, 25)
        frames = list(reader)
    assert reader.closed
    assert [frame.frame_index for frame in frames] == list(range(8))
    assert [frame.source_time_ms for frame in frames] == [index * 40 for index in range(8)]
    output = tmp_path / "roundtrip.mp4"
    with VideoWriter(output, 25, (96, 64)) as writer:
        for frame in frames:
            writer.write(frame.image)
    assert writer.closed
    with VideoReader(output) as check:
        assert len(list(check)) == 8


def test_missing_and_corrupt_video_are_errors(tmp_path):
    with pytest.raises(VideoError):
        with VideoReader(tmp_path / "missing.mp4"):
            pass
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"invalid video")
    with pytest.raises(VideoError):
        with VideoReader(corrupt):
            pass


def test_invalid_fps_releases_capture(sample_video, monkeypatch):
    class InvalidCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return 0 if prop == cv2.CAP_PROP_FPS else 96

        def release(self):
            self.released = True

    capture = InvalidCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(VideoError, match="FPS"):
        with VideoReader(sample_video):
            pass
    assert capture.released


def test_premature_decode_failure_is_not_normal_eof(sample_video, monkeypatch):
    original = cv2.VideoCapture

    class TruncatedCapture:
        def __init__(self, path):
            self.inner = original(path)
            self.reads = 0

        def __getattr__(self, key):
            return getattr(self.inner, key)

        def read(self):
            self.reads += 1
            return self.inner.read() if self.reads <= 3 else (False, None)

    monkeypatch.setattr(cv2, "VideoCapture", TruncatedCapture)
    with pytest.raises(VideoError, match="早期終了"):
        with VideoReader(sample_video) as reader:
            list(reader)
    assert reader.closed


def test_writer_rejects_wrong_size_and_closes(tmp_path):
    with pytest.raises(VideoError, match="サイズ"):
        with VideoWriter(tmp_path / "wrong.mp4", 25, (96, 64)) as writer:
            writer.write(np.zeros((32, 96, 3), dtype=np.uint8))
    assert writer.closed


def test_writer_refuses_existing_files_and_odd_dimensions(tmp_path):
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        with VideoWriter(existing, 25, (96, 64)):
            pass
    assert existing.read_bytes() == b"preserve"
    with pytest.raises(VideoError, match="偶数"):
        with VideoWriter(tmp_path / "odd.mp4", 25, (95, 63)):
            pass


def test_render_uses_original_pixels_without_mutating_input():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    points = coco17((Keypoint("left_shoulder", 30, 30, 0.9), Keypoint("left_elbow", 40, 40, 0.9)))
    result = FrameResult("r", "test-stub", 0, 0, 96, 64, people=(Person((20, 20, 80, 60), points),))
    rendered = draw_poses(image, result)
    assert not image.any()
    assert rendered[35, 35].any(), "Skeleton must connect the original pixel coordinates"
    assert not rendered[0, 0].any(), "Missing joints must not be drawn at (0, 0)"


def test_render_boundary_is_visibly_distinct():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    success = FrameResult("r", "test-stub", 0, 0, 96, 64)
    boundary = FrameResult(
        "r", "test-stub", 0, 0, 96, 64, status="not_estimated", reason="temporal_boundary"
    )
    assert not np.array_equal(draw_poses(image, success), draw_poses(image, boundary))
