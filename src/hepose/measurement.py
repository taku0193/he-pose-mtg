"""A shared source-frame schedule and timings from the actual offline pipeline."""

from dataclasses import dataclass
from time import perf_counter_ns

from hepose.results import TIMING_FIELDS
from hepose.schema import integer
from hepose.timing import summarize, throughput


@dataclass(frozen=True, slots=True)
class MeasurementPlan:
    source_count: int
    first: int
    stop: int
    warmup: int
    samples: int
    sha256: str | None = None

    def __post_init__(self):
        for name in ("source_count", "first", "stop", "warmup", "samples"):
            integer(getattr(self, name), name, 100 if name == "samples" else 0)
        if not self.first <= self.stop <= self.source_count:
            raise ValueError("invalid benchmark source range")

    @property
    def warmup_frames(self):
        return range(self.first, min(self.first + self.warmup, self.stop))

    @property
    def measured_frames(self):
        first = self.warmup_frames.stop
        return range(first, min(first + self.samples, self.stop))


class Measurement:
    def __init__(self, plan):
        self.plan = plan
        self.warmed = 0
        self.first_ns = self.last_ns = None
        self.attempted = 0
        self.frames = []
        self.without_people = 0
        self.values = {name: [] for name in TIMING_FIELDS}
        self.methods = set()

    def begin(self, frame_index, read_started_ns):
        if frame_index in self.plan.measured_frames:
            if self.first_ns is None:
                self.first_ns = read_started_ns
            self.attempted += 1

    def finish(self, result, timings):
        index = result.frame_index
        if index in self.plan.warmup_frames and result.status == "success":
            self.warmed += 1
        if index not in self.plan.measured_frames:
            return
        self.last_ns = perf_counter_ns()
        if result.status != "success":
            return
        self.frames.append(index)
        self.without_people += not result.people
        for name in self.values:
            self.values[name].append(timings.get(name))
        if method := timings.get("measurement_method"):
            self.methods.add(method)

    def summary(self):
        elapsed = (
            (self.last_ns - self.first_ns) / 1e6
            if self.first_ns is not None and self.last_ns is not None
            else None
        )
        return {
            "status": "success"
            if len(self.frames) >= self.plan.samples
            else "insufficient_samples",
            "repetitions": 1,
            "warmup_requested": self.plan.warmup,
            "warmup_completed": self.warmed,
            "samples_requested": self.plan.samples,
            "samples_attempted": self.attempted,
            "frame_indices": self.frames,
            "frames_without_people": self.without_people,
            "measured_ms": elapsed,
            "processing_fps": throughput(len(self.frames), elapsed)
            if elapsed is not None
            else None,
            "timing": {name: summarize(values) for name, values in self.values.items()},
            "inference_measurement_methods": sorted(self.methods),
            "scope": (
                "From the new source read for the first measured target through the last "
                "measured pose and timing write. Includes decode, adapter, render and per-frame "
                "writes. Excludes startup, warmup, output close and verification. For temporal "
                "models, prior context frames are already buffered."
            ),
        }
