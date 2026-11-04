"""Adapter interface: model dependencies belong to model-specific environments."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from hepose.schema import Person, TemporalContext


@dataclass(frozen=True, slots=True)
class InputFrame:
    frame_index: int
    source_time_ms: float
    width: int
    height: int
    image: object


@dataclass(frozen=True, slots=True)
class Prediction:
    people: tuple[Person, ...]
    inference_ms: float | None
    measurement_method: str | None


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
