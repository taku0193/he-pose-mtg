"""Bounded-window offline inference with injectable, model-specific adapters."""

import json
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

from hepose.environment import collect_environment
from hepose.render import draw_poses
from hepose.results import RunWriter, write_json
from hepose.schema import FrameResult
from hepose.timing import summarize, throughput
from hepose.video import VideoError, VideoReader, VideoWriter


def run_video(
    input_path: Path,
    output: Path,
    adapter_factory,
    *,
    model_id: str,
    device: str = "cuda:0",
    configuration: dict | None = None,
) -> dict:
    input_path, output = Path(input_path), Path(output)
    run_id = uuid4().hex
    started = perf_counter_ns()
    environment = {
        "model_id": model_id,
        "requested_device": device,
        "device": None,
        "input": {"path": str(input_path.resolve()), "sha256": None},
    }
    if configuration:
        environment["configuration"] = configuration
    model_times = []
    with RunWriter(output, run_id, environment) as results:
        environment.update(collect_environment(input_path, model_id, device))
        write_json(output / "environment.json", environment)
        if not input_path.is_file():
            raise VideoError(f"入力動画がありません: {input_path}")
        with ExitStack() as resources:
            load_started = perf_counter_ns()
            adapter = adapter_factory()
            resources.callback(adapter.close)
            results.details["model_load_ms"] = (perf_counter_ns() - load_started) / 1_000_000
            if adapter.model_id != model_id:
                raise ValueError("requested model and adapter identity do not match")
            environment["adapter"] = adapter.environment()
            environment["device"] = environment["adapter"].get("device")
            write_json(output / "environment.json", environment)
            reader = resources.enter_context(VideoReader(input_path))
            writer = resources.enter_context(
                VideoWriter(output / "annotated.mp4", reader.fps, (reader.width, reader.height))
            )
            temporal = adapter.temporal
            buffer = deque(maxlen=(temporal.window_size - 1) * temporal.step + 1)
            output_index = 0
            emitted = 0
            processing_started = perf_counter_ns()

            def emit(target_index, available_count):
                nonlocal output_index, emitted
                lookup = {frame.frame_index: frame for frame in buffer}
                target = lookup[target_index]
                references = temporal.references(target_index, available_count)
                timings = {"decode_ms": target.decode_ms}
                common = {
                    "run_id": run_id,
                    "model_id": model_id,
                    "frame_index": target_index,
                    "source_time_ms": target.source_time_ms,
                    "width": target.width,
                    "height": target.height,
                    "temporal": temporal,
                    "reference_frame_indices": references or (),
                }
                if references is None:
                    result = FrameResult(
                        **common, status="not_estimated", reason="temporal_boundary"
                    )
                else:
                    try:
                        adapter_started = perf_counter_ns()
                        prediction = adapter.predict(
                            [lookup[index] for index in references], target_index
                        )
                        timings["adapter_ms"] = (perf_counter_ns() - adapter_started) / 1_000_000
                        timings["inference_ms"] = prediction.inference_ms
                        timings["measurement_method"] = prediction.measurement_method
                        result = FrameResult(
                            **common, people=prediction.people, output_index=output_index
                        )
                        output_index += 1
                        model_times.append(prediction.inference_ms)
                    except Exception as exc:
                        results.write(
                            FrameResult(**common, status="failed", reason=str(exc)), timings
                        )
                        raise
                render_started = perf_counter_ns()
                rendered = draw_poses(target.image, result)
                timings["render_ms"] = (perf_counter_ns() - render_started) / 1_000_000
                write_started = perf_counter_ns()
                writer.write(rendered)
                timings["write_ms"] = (perf_counter_ns() - write_started) / 1_000_000
                if target.read_started_ns is not None:
                    timings["total_ms"] = (perf_counter_ns() - target.read_started_ns) / 1_000_000
                results.write(result, timings)
                emitted += 1

            for frame in reader:
                buffer.append(frame)
                if frame.frame_index >= temporal.future_frames:
                    emit(frame.frame_index - temporal.future_frames, reader.decoded)
            for index in range(emitted, reader.decoded):
                emit(index, reader.decoded)
            if not output_index:
                raise VideoError("推定可能な中心フレームがありません。")
        # Resources close (including output decode verification) before run success is recorded.
        elapsed_ms = (perf_counter_ns() - processing_started) / 1_000_000
        results.details.update(
            {
                "source_frame_count": reader.decoded,
                "declared_frame_count": reader.declared_frame_count,
                "source_fps": reader.fps,
                "processing_ms": elapsed_ms,
                "processing_fps": throughput(output_index, elapsed_ms),
                "inference": summarize(model_times),
                "total_before_summary_ms": (perf_counter_ns() - started) / 1_000_000,
                "timing_scope": {
                    "processing": "decode, adapter, render, writes, output verification; no load",
                    "total_ms": "target read start to before result writes; includes window wait",
                    "inference_ms": "adapter-reported synchronized model time; null if unknown",
                },
            }
        )
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))
