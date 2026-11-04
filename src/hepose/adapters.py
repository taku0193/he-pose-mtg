"""Adapter interface: model dependencies belong to model-specific environments."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from hepose.schema import Person, TemporalContext, finite


@dataclass(frozen=True, slots=True)
class InputFrame:
    frame_index: int
    source_time_ms: float
    width: int
    height: int
    image: object
    decode_ms: float | None = None
    read_started_ns: int | None = None


@dataclass(frozen=True, slots=True)
class Prediction:
    people: tuple[Person, ...]
    inference_ms: float | None
    measurement_method: str | None

    def __post_init__(self):
        if self.inference_ms is not None:
            finite(self.inference_ms, "inference_ms", 0)
            if not self.measurement_method:
                raise ValueError("measured inference time requires a measurement method")


class PoseAdapter(Protocol):
    model_id: str
    temporal: TemporalContext

    def environment(self) -> dict:
        """Return source revision, backend, precision, input size, weights and licenses."""
        ...

    def predict(self, frames: Sequence[InputFrame], target_frame_index: int) -> Prediction:
        """Receive the required temporal window and return original-image coordinates."""
        ...

    def close(self) -> None:
        """Release resources owned by this adapter."""
        ...


def load_adapter(model_id: str, device: str, model_config: dict | None = None) -> PoseAdapter:
    """Model branches will add lazy, local-weight-only implementations here."""
    if model_id == "yolo26":
        from hepose.yolo26 import Yolo26Adapter

        return Yolo26Adapter(device, model_config or {})
    raise NotImplementedError(f"{model_id}: モデルアダプタは未実装です。")
