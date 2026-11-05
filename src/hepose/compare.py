"""Re-render saved predictions on verified source frames and combine model panels."""

import hashlib
import json
import math
import os
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from hepose.render import draw_poses
from hepose.schema import SCHEMA_VERSION, FrameResult, Keypoint, Person, TemporalContext
from hepose.video import VideoReader, VideoWriter, backend

LABELS = {
    "yolo26": "YOLO26n-Pose",
    "detrpose": "DETRPose-N",
    "tar-vitpose": "TAR-ViTPose (offline)",
}


def _metadata(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON metadata: {path}")
    return value


def _frame(line):
    try:
        data = json.loads(line)
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported frame schema")
        people = []
        for person in data.pop("people"):
            if not isinstance(person, dict):
                raise ValueError("person must be an object")
            for name in ("keypoints", "native_keypoints"):
                person[name] = tuple(Keypoint(**point) for point in person.get(name, []))
            people.append(Person(**person))
        data["people"] = tuple(people)
        data["reference_frame_indices"] = tuple(data["reference_frame_indices"])
        data["temporal"] = TemporalContext(**data["temporal"])
        return FrameResult(**data)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid pose frame: {exc}") from exc


class SavedRun:
    def __init__(self, poses):
        self.poses = poses
        environment = _metadata(poses.with_name("environment.json"))
        self.summary = _metadata(poses.with_name("summary.json"))
        self.model_id = environment.get("model_id")
        self.run_id = self.summary.get("run_id")
        source = environment.get("input") or {}
        if not isinstance(source, dict):
            raise ValueError(f"invalid input metadata: {poses}")
        self.sha256 = source.get("sha256")
        self.source_path = source.get("path")
        if self.model_id not in LABELS or not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError(f"unknown model or invalid run_id: {poses}")
        if not isinstance(self.sha256, str) or not re.fullmatch("[a-f0-9]{64}", self.sha256):
            raise ValueError(f"missing or invalid input SHA256: {poses}")
        status = self.summary.get("status")
        if self.summary.get("schema_version") != SCHEMA_VERSION or status not in {
            "success",
            "failed",
            "interrupted",
        }:
            raise ValueError(f"unsupported run summary: {poses}")
        self.complete = status == "success"
        self.last_index = None
        self.counts = Counter(success=0, failed=0, not_estimated=0)

    def open(self, stack):
        stream = stack.enter_context(self.poses.open(encoding="utf-8"))
        self.frames = iter(stream)
        self.next = self._advance()

    def _advance(self):
        line = next(self.frames, None)
        if line is None:
            return None
        result = _frame(line)
        if result.run_id != self.run_id or result.model_id != self.model_id:
            raise ValueError(f"result run_id or model differs from summary: {self.poses}")
        if self.last_index is not None and result.frame_index <= self.last_index:
            raise ValueError(f"source frame indices must be unique and increasing: {self.poses}")
        if self.complete and result.status == "failed":
            raise ValueError(f"successful run contains a failed frame: {self.poses}")
        self.last_index = result.frame_index
        self.counts[result.status] += 1
        return result

    def match(self, source):
        common = (
            self.run_id,
            self.model_id,
            source.frame_index,
            source.source_time_ms,
            source.width,
            source.height,
        )
        if self.next is None or self.next.frame_index > source.frame_index:
            if self.complete:
                raise ValueError(f"missing source frame {source.frame_index}: {self.poses}")
            return FrameResult(*common, status="not_estimated", reason="no_saved_result")
        result = self.next
        if (
            result.frame_index != source.frame_index
            or (result.width, result.height) != (source.width, source.height)
            or not math.isclose(result.source_time_ms, source.source_time_ms, abs_tol=0.01)
        ):
            raise ValueError(f"source frame dimensions or timestamp do not match: {self.poses}")
        self.next = self._advance()
        return result

    def verify_end(self, source_count, fps):
        if (
            self.next is not None
            or self.summary.get("last_frame_index") != self.last_index
            or self.summary.get("frame_counts") != dict(self.counts)
        ):
            raise ValueError(f"frame data does not agree with source/run summary: {self.poses}")
        if self.complete:
            details = self.summary.get("details", {})
            if details.get("source_frame_count") != source_count or not math.isclose(
                details.get("source_fps", 0), fps, rel_tol=1e-6
            ):
                raise ValueError(f"source length or FPS does not match run summary: {self.poses}")


def _source_video(runs, source):
    if len({run.sha256 for run in runs}) != 1:
        raise ValueError("比較対象の入力SHA256が一致しません。")
    candidates = (
        [Path(source)]
        if source
        else [Path(run.source_path) for run in runs if isinstance(run.source_path, str)]
    )
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            with candidate.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == runs[0].sha256:
                    return candidate
    raise ValueError("SHA256が一致する元動画がありません。--input で元動画を指定してください。")


def _render_panel(cv2, frame, result, height):
    image = draw_poses(frame.image, result)
    panel = cv2.copyMakeBorder(image, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 27, 25))
    scale = min(0.65, frame.width / 400)
    color = (245, 245, 245)
    cv2.putText(
        panel,
        LABELS[result.model_id],
        (6, height // 2 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )
    label = {
        "success": "OK" if result.people else "NO PERSON",
        "failed": "FAILED",
        "not_estimated": "NOT ESTIMATED",
    }[result.status]
    if result.reason == "no_saved_result":
        label = "NO SAVED RESULT"
    cv2.putText(
        panel,
        f"#{frame.frame_index}  {frame.source_time_ms / 1000:.3f}s  {label}",
        (6, height - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.82,
        color,
        1,
        cv2.LINE_AA,
    )
    return panel


def compare_video(input_dir, output, *, source=None):
    import numpy as np

    input_dir, output = Path(input_dir), Path(output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("比較動画の出力先には .mp4 を指定してください。")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    by_model, run_ids = {}, set()
    for poses in sorted(input_dir.rglob("poses.jsonl")):
        run = SavedRun(poses)
        if run.model_id in by_model:
            raise ValueError(f"同じモデルの結果が複数あります: {run.model_id}")
        if run.run_id in run_ids:
            raise ValueError(f"重複したrun_id: {run.run_id}")
        by_model[run.model_id] = run
        run_ids.add(run.run_id)
    if len(by_model) < 2:
        raise ValueError("比較には2種類以上のモデルの実行結果が必要です。")
    runs = [by_model[model] for model in LABELS if model in by_model]
    video = _source_video(runs, source)
    cv2 = backend()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp, ExitStack() as inputs:
        staged = Path(temp) / "comparison.mp4"
        reader = inputs.enter_context(VideoReader(video))
        for run in runs:
            run.open(inputs)
        banner_height = 48
        size = (len(runs) * reader.width, reader.height + banner_height)
        with VideoWriter(staged, reader.fps, size) as writer:
            for frame in reader:
                panels = [
                    _render_panel(cv2, frame, run.match(frame), banner_height) for run in runs
                ]
                writer.write(np.concatenate(panels, axis=1))
            for run in runs:
                run.verify_end(reader.decoded, reader.fps)
        # Publish only after validation, with an atomic no-overwrite operation.
        os.link(staged, output)
    return reader.decoded
