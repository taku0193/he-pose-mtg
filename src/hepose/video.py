"""CFR video I/O with explicit validation and deterministic resource release."""

import math
from pathlib import Path
from time import perf_counter_ns

from hepose.adapters import InputFrame


class VideoError(RuntimeError):
    pass


def backend():
    try:
        import cv2
    except ImportError as exc:
        raise VideoError(
            "OpenCVがありません。準備時にvideoまたはclient依存を導入してください。"
        ) from exc
    return cv2


class VideoReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.closed = True
        self.decoded = 0

    def __enter__(self):
        if not self.path.is_file():
            raise VideoError(f"入力動画がありません: {self.path}")
        cv2 = backend()
        self.capture = cv2.VideoCapture(str(self.path))
        try:
            if not self.capture.isOpened():
                raise VideoError(f"動画を開けません: {self.path}")
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise VideoError("入力動画のFPSが不明または不正です。")
            dimensions = [
                self.capture.get(prop)
                for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
            ]
            if any(not math.isfinite(value) or value < 1 for value in dimensions):
                raise VideoError("入力動画の画像サイズが不正です。")
            self.width, self.height = map(int, dimensions)
            count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.declared_frame_count = int(count) if math.isfinite(count) and count > 0 else None
        except BaseException:
            self.capture.release()
            raise
        self.closed = False
        return self

    def __iter__(self):
        if self.closed:
            raise VideoError("動画リーダーは閉じられています。")
        while True:
            started = perf_counter_ns()
            ok, image = self.capture.read()
            elapsed = (perf_counter_ns() - started) / 1_000_000
            if not ok:
                if self.decoded == 0:
                    raise VideoError("入力動画に読み取れるフレームがありません。")
                if self.declared_frame_count and self.decoded < self.declared_frame_count:
                    raise VideoError(
                        f"動画の復号が早期終了しました: {self.decoded}/{self.declared_frame_count}"
                    )
                return
            if image is None or image.shape != (self.height, self.width, 3):
                raise VideoError("復号した画像サイズが入力動画の宣言値と一致しません。")
            frame = InputFrame(
                self.decoded,
                self.decoded * 1000 / self.fps,
                self.width,
                self.height,
                image,
                elapsed,
                started,
            )
            self.decoded += 1
            yield frame

    def __exit__(self, exc_type, exc_value, traceback):
        self.capture.release()
        self.closed = True


class VideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.fps = fps
        self.size = size
        self.written = 0
        self.closed = True

    def __enter__(self):
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        width, height = self.size
        if min(width, height) < 2 or width % 2 or height % 2:
            raise VideoError("MP4出力の幅・高さは正の偶数が必要です。")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise VideoError("出力FPSが不正です。")
        cv2 = backend()
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise VideoError(f"動画出力を開けません: {self.path}")
        self.closed = False
        return self

    def write(self, image):
        width, height = self.size
        if self.closed or image.shape != (height, width, 3):
            raise VideoError("出力画像のサイズが一致しないか、出力が閉じられています。")
        if str(image.dtype) != "uint8":
            raise VideoError("出力画像はuint8のBGR画像でなければなりません。")
        self.writer.write(image)
        self.written += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.release()
        self.closed = True
        if exc_type is None:
            # VideoWriter.write has no success return value. Decode the result before accepting it.
            if not self.written or not self.path.is_file() or not self.path.stat().st_size:
                raise VideoError("動画出力が空です。")
            with VideoReader(self.path) as check:
                if (check.width, check.height) != self.size:
                    raise VideoError("出力動画のサイズが入力と一致しません。")
                if not math.isclose(check.fps, self.fps, rel_tol=0.001):
                    raise VideoError("出力動画のFPSが入力と一致しません。")
                if sum(1 for _ in check) != self.written:
                    raise VideoError("書き込んだ動画のフレーム数が一致しません。")
