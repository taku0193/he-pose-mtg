"""Milliseconds and statistics without substituting zero for missing data."""

from collections.abc import Iterable
from contextlib import contextmanager
from time import perf_counter_ns

from hepose.schema import finite, integer


def summarize(values: Iterable[float | None]) -> dict[str, float | int | None]:
    samples = []
    for value in values:
        if value is not None:
            finite(value, "duration_ms", 0)
            samples.append(value)
    samples.sort()

    def percentile(fraction: float) -> float | None:
        if not samples:
            return None
        position = fraction * (len(samples) - 1)
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)

    return {
        "count": len(samples),
        "mean_ms": sum(samples) / len(samples) if samples else None,
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
    }


def throughput(success_count: int, elapsed_ms: float) -> float | None:
    integer(success_count, "success_count")
    finite(elapsed_ms, "elapsed_ms", 0)
    return success_count * 1000 / elapsed_ms if elapsed_ms > 0 else None


@contextmanager
def measure(target: dict, stage: str, synchronize=None, clock=perf_counter_ns):
    """A supplied synchronization hook fences both ends, including on errors."""
    if synchronize:
        synchronize()
    start = clock()
    try:
        yield
    finally:
        if synchronize:
            synchronize()
        target[stage] = (clock() - start) / 1_000_000
