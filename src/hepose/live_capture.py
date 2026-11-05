"""The capture thread owns the camera and publishes only the newest waiting frame."""

import math
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic, perf_counter_ns

from hepose.live_protocol import check_size
from hepose.video import VideoError, backend


@dataclass(frozen=True, slots=True)
class Captured:
    index: int
    image: object
    at_ns: int


class LatestFrame:
    def __init__(self):
        self.lock = Lock()
        self.pending = None
        self.acquired = self.dropped = 0

    def put(self, frame):
        with self.lock:
            self.acquired += 1
            if self.pending is not None:
                self.dropped += 1
            self.pending = frame

    def take(self):
        with self.lock:
            frame, self.pending = self.pending, None
            return frame

    def counts(self):
        with self.lock:
            return self.acquired, self.dropped, int(self.pending is not None)


class Capture:
    def __init__(self, options):
        self.options = options
        self.slot = LatestFrame()
        self.stop = Event()
        self.done = Event()
        self.thread = Thread(target=self._read, name="hepose-capture", daemon=True)
        self.error = None
        self.sizes = set()
        self.first_ns = self.last_ns = None
        self.backend_name = self.reported_fps = None

    def _read(self):
        capture = None
        try:
            cv2 = backend()
            options = self.options
            source = str(options.video) if options.video else options.camera
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise VideoError(f"映像入力を開けません: {source}（Macのカメラ許可も確認）")
            self.backend_name = capture.getBackendName()
            if not options.video:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, options.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, options.height)
                capture.set(cv2.CAP_PROP_FPS, options.fps)
            fps = capture.get(cv2.CAP_PROP_FPS)
            self.reported_fps = fps if math.isfinite(fps) and fps > 0 else None
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) if options.video else 0
            expected = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else None
            start = monotonic()
            index = 0
            while not self.stop.is_set() and (options.limit is None or index < options.limit):
                if options.video and self.stop.wait(
                    max(0, start + index / options.fps - monotonic())
                ):
                    break
                ok, image = capture.read()
                at_ns = perf_counter_ns()
                if self.stop.is_set():
                    break
                if not ok:
                    if options.video and index and (expected is None or index >= expected):
                        break
                    raise VideoError(f"映像フレームを取得できません（取得済み {index} 枚）。")
                if (
                    image is None
                    or image.ndim != 3
                    or image.shape[2] != 3
                    or image.dtype != "uint8"
                ):
                    raise VideoError("映像フレームの画素形式が不正です。")
                height, width = image.shape[:2]
                check_size(width, height)
                self.sizes.add((width, height))
                if self.first_ns is None:
                    self.first_ns = at_ns
                self.last_ns = at_ns
                self.slot.put(Captured(index, image, at_ns))
                index += 1
        except Exception as exc:
            self.error = exc
        finally:
            try:
                if capture is not None:
                    capture.release()
            except Exception as exc:
                self.error = self.error or exc
            finally:
                self.done.set()

    def metadata(self):
        acquired, _, _ = self.slot.counts()
        span = (self.last_ns - self.first_ns) / 1e9 if self.last_ns is not None else None
        return {
            "backend": self.backend_name,
            "reported_fps": self.reported_fps,
            "observed_sizes": sorted(self.sizes.copy()),
            "observed_capture_fps": (acquired - 1) / span if span and acquired > 1 else None,
            "camera_internal_drops": None,
        }
