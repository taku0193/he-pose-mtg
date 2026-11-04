"""Model-independent coordinates, frame identity and temporal context."""

import json
import math
from dataclasses import asdict, dataclass, replace

COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
MODEL_IDS = ("yolo26", "detrpose", "tar-vitpose")
SCHEMA_VERSION = "1.0"


def finite(value: float, name: str, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"invalid {name}: {value}")


def integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, slots=True)
class Keypoint:
    name: str
    x: float | None = None
    y: float | None = None
    confidence: float | None = None
    confidence_meaning: str | None = None
    visibility: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("keypoint name is required")
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must both be present or both be null")
        for name in ("x", "y", "confidence"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
        if self.visibility is not None:
            integer(self.visibility, "visibility")
            if self.visibility > 2:
                raise ValueError("visibility must be 0, 1, 2 or null")


def coco17(points: tuple[Keypoint, ...]) -> tuple[Keypoint, ...]:
    """Match names, preserving absent coordinates as null; never relabel heads as eyes."""
    lookup = {point.name: point for point in points}
    if len(lookup) != len(points):
        raise ValueError("duplicate keypoint names")
    return tuple(lookup.get(name, Keypoint(name)) for name in COCO17)


def restore_crop(
    points: tuple[Keypoint, ...],
    crop_xyxy: tuple[float, float, float, float],
    model_size: tuple[int, int],
) -> tuple[Keypoint, ...]:
    """Restore coordinates from model pixels to original pixels; sizes use width, height."""
    x1, y1, x2, y2 = crop_xyxy
    for value in crop_xyxy:
        finite(value, "crop coordinate")
    width, height = model_size
    integer(width, "model width", 1)
    integer(height, "model height", 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("crop must have positive area")
    return tuple(
        replace(point, x=x1 + point.x * (x2 - x1) / width,
                y=y1 + point.y * (y2 - y1) / height)
        if point.x is not None else point
        for point in points
    )


@dataclass(frozen=True, slots=True)
class Person:
    bbox: tuple[float, float, float, float]
    keypoints: tuple[Keypoint, ...] = ()
    track_id: str | None = None
    joint_definition: str = "coco17"
    native_keypoints: tuple[Keypoint, ...] = ()
    native_joint_definition: str | None = None
    crop_xyxy: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError("bbox must be xyxy")
        for value in self.bbox:
            finite(value, "bbox coordinate")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("bbox must have positive area")
        for points in (self.keypoints, self.native_keypoints):
            if len({point.name for point in points}) != len(points):
                raise ValueError("duplicate person keypoints")


@dataclass(frozen=True, slots=True)
class TemporalContext:
    window_size: int = 1
    step: int = 1
    future_frames: int = 0

    def __post_init__(self) -> None:
        integer(self.window_size, "window_size", 1)
        integer(self.step, "step", 1)
        integer(self.future_frames, "future_frames")
        if self.future_frames > (self.window_size - 1) * self.step:
            raise ValueError("future_frames exceeds window span")
        if self.future_frames % self.step:
            raise ValueError("future_frames must be a multiple of step")

    def references(self, target: int, frame_count: int) -> tuple[int, ...] | None:
        integer(target, "target")
        integer(frame_count, "frame_count")
        if target >= frame_count:
            raise ValueError("target is outside input")
        first = target - ((self.window_size - 1) * self.step - self.future_frames)
        last = target + self.future_frames
        if first < 0 or last >= frame_count:
            return None
        return tuple(range(first, last + 1, self.step))


@dataclass(frozen=True, slots=True)
class FrameResult:
    run_id: str
    model_id: str
    frame_index: int
    source_time_ms: float
    width: int
    height: int
    people: tuple[Person, ...] = ()
    status: str = "success"
    reason: str | None = None
    output_index: int | None = None
    source_time_basis: str = "frame_index/source_fps"
    reference_frame_indices: tuple[int, ...] = ()
    temporal: TemporalContext = TemporalContext()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id or not self.model_id:
            raise ValueError("run_id and model_id are required")
        integer(self.frame_index, "frame_index")
        integer(self.width, "width", 1)
        integer(self.height, "height", 1)
        finite(self.source_time_ms, "source_time_ms", 0)
        if self.output_index is not None:
            integer(self.output_index, "output_index")
        if self.status not in {"success", "failed", "not_estimated"}:
            raise ValueError("unknown frame status")
        if self.status != "success" and (not self.reason or self.people):
            raise ValueError("unsuccessful frames require a reason and no predictions")
        for index in self.reference_frame_indices:
            integer(index, "reference_frame_index")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)
