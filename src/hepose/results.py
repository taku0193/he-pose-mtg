"""Write frame results without overwriting previous runs or inventing measurements."""

import csv
import json
import os
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from hepose.schema import SCHEMA_VERSION, FrameResult, finite

TIMING_FIELDS = (
    "decode_ms",
    "preprocess_ms",
    "detector_ms",
    "inference_ms",
    "postprocess_ms",
    "render_ms",
    "write_ms",
    "total_ms",
    "adapter_ms",
)


def write_json(path: Path, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


class RunWriter:
    def __init__(self, output: Path, run_id: str, environment: dict):
        self.output = Path(output)
        self.run_id = run_id
        self.environment = environment
        self.counts = Counter({name: 0 for name in ("success", "failed", "not_estimated")})
        self.last_frame_index = None
        self.closed = True
        self._stack = ExitStack()
        self.details = {}
        self._recorded_error = None

    def __enter__(self):
        # Serialize first so invalid metadata does not create half-initialized runs.
        json.dumps(self.environment, allow_nan=False)
        if not self.run_id:
            raise ValueError("run_id is required")
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            write_json(self.output / "environment.json", self.environment)
            self._poses = self._stack.enter_context(
                (self.output / "poses.jsonl").open("x", encoding="utf-8")
            )
            self._timings = self._stack.enter_context(
                (self.output / "timings.csv").open("x", encoding="utf-8", newline="")
            )
            self._log = self._stack.enter_context(
                (self.output / "run.log").open("x", encoding="utf-8")
            )
            self._csv = csv.DictWriter(
                self._timings,
                fieldnames=(
                    "run_id",
                    "frame_index",
                    "source_time_ms",
                    "status",
                    "reason",
                    *TIMING_FIELDS,
                    "measurement_method",
                ),
            )
            self._csv.writeheader()
            self.started_at = datetime.now(UTC).isoformat()
            self.closed = False
            self._log.write(f"{self.started_at} started run={self.run_id}\n")
        except BaseException:
            self._stack.close()
            raise
        return self

    def write(self, frame: FrameResult, timings: dict | None = None) -> None:
        if self.closed:
            raise ValueError("run writer is closed")
        if frame.run_id != self.run_id:
            raise ValueError("result belongs to another run")
        if self.last_frame_index is not None and frame.frame_index <= self.last_frame_index:
            raise ValueError("source frame indices must increase")
        timings = dict(timings or {})
        unknown = timings.keys() - {*TIMING_FIELDS, "measurement_method"}
        if unknown:
            raise ValueError(f"unknown timing fields: {sorted(unknown)}")
        for name in TIMING_FIELDS:
            if timings.get(name) is not None:
                finite(timings[name], name, 0)
        line = frame.to_json()
        row = {
            "run_id": self.run_id,
            "frame_index": frame.frame_index,
            "source_time_ms": frame.source_time_ms,
            "status": frame.status,
            "reason": frame.reason,
            **timings,
        }
        self._poses.write(line + "\n")
        self._csv.writerow(row)
        self._poses.flush()
        self._timings.flush()
        self.counts[frame.status] += 1
        self.last_frame_index = frame.frame_index

    def record_failure(self, error: BaseException) -> None:
        if self.closed:
            raise ValueError("run writer is closed")
        self._recorded_error = error

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and self._recorded_error is not None:
            exc_type, exc_value = type(self._recorded_error), self._recorded_error
        status = (
            "interrupted"
            if exc_type and issubclass(exc_type, KeyboardInterrupt)
            else (
                "failed"
                if exc_type or self.counts["failed"] or not self.counts["success"]
                else "success"
            )
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "frame_counts": dict(self.counts),
            "last_frame_index": self.last_frame_index,
            "details": self.details,
            "error": str(exc_value)
            if exc_type
            else ("no successful inference frames" if not self.counts["success"] else None),
            "artifacts": sorted(
                {path.name for path in self.output.iterdir() if path.is_file()} | {"summary.json"}
            ),
        }
        try:
            self._log.write(f"{summary['finished_at']} {status} error={summary['error']}\n")
        finally:
            self._stack.close()
            self.closed = True
        write_json(self.output / "summary.json", summary)
        return False
