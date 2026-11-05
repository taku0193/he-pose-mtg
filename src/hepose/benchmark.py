"""Run isolated model environments in order, using the same GPU and source-frame schedule."""

import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter_ns

from hepose.adapters import MODEL_TEMPORAL
from hepose.environment import collect_environment
from hepose.measurement import MeasurementPlan
from hepose.results import RunWriter, write_json
from hepose.video import VideoReader

CSV_FIELDS = (
    "model_id",
    "status",
    "run_id",
    "device",
    "gpu_uuid",
    "backend",
    "precision",
    "input_size",
    "warmup_frames",
    "measured_frames",
    "first_measured_frame",
    "last_measured_frame",
    "measured_without_people",
    "measured_ms",
    "processing_fps",
    "inference_samples",
    "inference_mean_ms",
    "inference_p50_ms",
    "inference_p95_ms",
    "adapter_mean_ms",
    "adapter_p50_ms",
    "adapter_p95_ms",
    "gpu_peak_memory_bytes",
    "model_load_ms",
    "first_adapter_call_ms",
    "worker_wall_ms",
    "source_frames",
    "success_frames",
    "failed_frames",
    "not_estimated_frames",
    "accuracy",
)


def make_plan(source_count, models, *, warmup=20, samples=100, sha256=None):
    contexts = [MODEL_TEMPORAL[model] for model in models]
    first = min(
        source_count,
        max((item.window_size - 1) * item.step - item.future_frames for item in contexts),
    )
    stop = max(first, source_count - max(item.future_frames for item in contexts))
    return MeasurementPlan(source_count, first, stop, warmup, samples, sha256)


def _gpu_assignments(models, configuration, device):
    assignments = {}
    uuids = set()
    for model in models:
        settings = configuration.model(model)
        chosen = device or settings.get("device", "cuda:0")
        uuid = settings.get("gpu_uuid", "").lower().removeprefix("gpu-")
        if not uuid or not chosen.startswith("cuda:"):
            raise ValueError(f"{model}: benchmarkには割当済みCUDAとgpu_uuidが必要です。")
        uuids.add(uuid)
        assignments[model] = (chosen, uuid)
    if len(uuids) != 1:
        raise ValueError("全モデルに同じ割当GPUのgpu_uuidを設定してください。")
    return assignments


def _read_result(output, model, worker_ms):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    adapter = environment.get("adapter", {})
    details = summary["details"]
    measure = details.get("benchmark", {})
    timing = measure.get("timing", {})
    inference = timing.get("inference_ms", {})
    adapter_time = timing.get("adapter_ms", {})
    indices = measure.get("frame_indices", [])
    counts = summary["frame_counts"]
    row = {
        "model_id": model,
        "status": measure.get("status", "failed")
        if summary["status"] == "success"
        else summary["status"],
        "run_id": summary["run_id"],
        "device": environment.get("device"),
        "gpu_uuid": adapter.get("gpu", {}).get("gpu_uuid"),
        "backend": adapter.get("backend"),
        "precision": adapter.get("precision"),
        "input_size": "x".join(map(str, adapter["input_size"]))
        if adapter.get("input_size")
        else None,
        "warmup_frames": measure.get("warmup_completed"),
        "measured_frames": len(indices) if measure else None,
        "first_measured_frame": indices[0] if indices else None,
        "last_measured_frame": indices[-1] if indices else None,
        "measured_without_people": measure.get("frames_without_people"),
        "measured_ms": measure.get("measured_ms"),
        "processing_fps": measure.get("processing_fps"),
        "inference_samples": inference.get("count"),
        "inference_mean_ms": inference.get("mean_ms"),
        "inference_p50_ms": inference.get("p50_ms"),
        "inference_p95_ms": inference.get("p95_ms"),
        "adapter_mean_ms": adapter_time.get("mean_ms"),
        "adapter_p50_ms": adapter_time.get("p50_ms"),
        "adapter_p95_ms": adapter_time.get("p95_ms"),
        "gpu_peak_memory_bytes": adapter.get("gpu_peak_memory_bytes"),
        "model_load_ms": details.get("model_load_ms"),
        "first_adapter_call_ms": details.get("first_adapter_call_ms"),
        "worker_wall_ms": worker_ms,
        "source_frames": details.get("source_frame_count"),
        "success_frames": counts["success"],
        "failed_frames": counts["failed"],
        "not_estimated_frames": counts["not_estimated"],
        "accuracy": "N/A（GTなし）",
    }
    return row, environment, summary


def _report(output, rows, summary, environment):
    with (output / "benchmark.csv").open("x", encoding="utf-8", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        table.writeheader()
        table.writerows(rows)

    def format_value(value):
        return (
            f"{value:.2f}" if isinstance(value, float) else str(value) if value is not None else "—"
        )

    source = environment.get("input") or {}
    plan = environment.get("plan") or {}
    text = [
        "# 同一動画・同一GPUでの姿勢推定測定",
        "",
        f"- 実行状態: {summary['status']}",
        f"- 入力SHA256: {source.get('sha256') or '—'}",
        f"- 実行順序: {', '.join(environment['models'])}",
        f"- 条件: batch=1、共通ウォームアップ{plan.get('warmup', '—')}件、"
        f"本計測{plan.get('samples', '—')}件、反復なし",
        "- 精度: N/A（GTなし）",
    ]
    if summary["error"]:
        error = " ".join(summary["error"].splitlines())
        text.append(f"- 終了理由: {error}")
    columns = (
        "model_id",
        "status",
        "measured_frames",
        "processing_fps",
        "inference_mean_ms",
        "inference_p50_ms",
        "inference_p95_ms",
        "inference_samples",
    )
    for title, is_tar in (("単フレームモデル", False), ("TAR-ViTPose（オフライン）", True)):
        selected = [row for row in rows if (row["model_id"] == "tar-vitpose") == is_tar]
        if not selected:
            continue
        text.extend(
            (
                "",
                f"## {title}",
                "",
                "| モデル | 状態 | 計測件数 | 処理FPS | 推論mean (ms) | p50 (ms) | p95 (ms) | "
                "推論時間の有効件数 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for row in selected:
            text.append("| " + " | ".join(format_value(row.get(key)) for key in columns) + " |")
    text.extend(
        (
            "",
            "処理FPSは、本計測で正常に処理した件数を、復号・アダプタ・描画・"
            "フレーム単位の結果書込を含む実経過時間で割った値です。"
            "モデルロード、ウォームアップ、動画のクローズ・検証はこの区間の外です。",
            "",
            "TARは前後2枚を使い、YOLO11x検出器を含みます。推論時間はTAR本体の人物別forwardの合計で、"
            "人物がいないフレームでは欠測です。処理FPSには検出器とクロップ処理も含みます。",
            "",
            "GPUピークは各モデルのプロセスで初期化から全編処理までにPyTorchが確保した最大値で、"
            "GPU全体の使用量ではありません。条件とモデル別の出力はenvironment.jsonと"
            "各モデルのサブディレクトリにあります。ライブ表示FPSや精度の比較には使えません。",
            "",
        )
    )
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")


def run_benchmark(
    input_path, output, models, configuration, *, device=None, warmup=20, samples=100
):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter_ns()
    environment = {
        "schema_version": "1.0",
        "models": list(models),
        "batch_size": 1,
        "execution": "one subprocess at a time; one continuous video pass per model",
        "repetitions": 1,
        "state_reset": "fresh process for each model; no replay within a model",
        "started_at": datetime.now(UTC).isoformat(),
        "runs": {},
    }
    rows = []
    status, error = "success", None
    with (output / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        try:
            assignments = _gpu_assignments(models, configuration, device)
            environment.update(collect_environment(input_path, None, device))
            with VideoReader(input_path) as reader:
                source_video = {"fps": reader.fps, "width": reader.width, "height": reader.height}
                count = sum(1 for _ in reader)
            plan = make_plan(
                count, models, warmup=warmup, samples=samples, sha256=environment["input"]["sha256"]
            )
            environment.update(
                source_video=dict(source_video, frame_count=count),
                plan=asdict(plan),
                warmup_frame_indices=list(plan.warmup_frames),
                measured_frame_indices=list(plan.measured_frames),
            )
            plan_path = output / "plan.json"
            write_json(plan_path, asdict(plan))
            write_json(output / "environment.json", environment)
            if len(plan.measured_frames) < samples:
                status = "insufficient_samples"
            for model in models:
                chosen, uuid = assignments[model]
                child = output / model
                cmd = [
                    configuration.model(model).get("python", sys.executable),
                    "-m",
                    "hepose.cli",
                    "run",
                    "--local",
                    "--model",
                    model,
                    "--input",
                    str(input_path.resolve()),
                    "--output",
                    str(child.resolve()),
                    "--device",
                    chosen,
                    "--benchmark-plan",
                    str(plan_path.resolve()),
                ]
                if configuration.path:
                    cmd.extend(("--config", str(configuration.path)))
                log.write(f"Starting {model} on {chosen}\n")
                worker_started = perf_counter_ns()
                try:
                    code = subprocess.run(cmd, stdout=log, stderr=log, check=False).returncode
                    if code in (130, -2):
                        raise KeyboardInterrupt
                    if code:
                        raise RuntimeError(f"{model}: worker exit {code}; see run.log")
                except (OSError, RuntimeError, KeyboardInterrupt) as exc:
                    if not child.exists():
                        with RunWriter(
                            child,
                            f"controller-{model}",
                            {
                                "model_id": model,
                                "input": environment["input"],
                                "device": None,
                            },
                        ) as results:
                            results.record_failure(exc)
                    if (child / "summary.json").is_file():
                        row, _, _ = _read_result(
                            child, model, (perf_counter_ns() - worker_started) / 1e6
                        )
                        rows.append(row)
                    raise
                row, worker_env, summary = _read_result(
                    child, model, (perf_counter_ns() - worker_started) / 1e6
                )
                rows.append(row)
                measure = summary["details"]["benchmark"]
                if (
                    worker_env["model_id"] != model
                    or worker_env["input"]["sha256"] != plan.sha256
                    or (row["gpu_uuid"] or "").lower().removeprefix("gpu-") != uuid
                    or row["device"] != chosen
                    or measure["frame_indices"] != list(plan.measured_frames)
                    or measure["warmup_completed"] != len(plan.warmup_frames)
                    or row["source_frames"] != count
                    or summary["status"] != "success"
                ):
                    row["status"] = "failed"
                    raise RuntimeError(
                        f"{model}: 測定結果の入力・GPU・フレーム範囲が一致しません。"
                    )
                environment["runs"][model] = {
                    "run_id": row["run_id"],
                    "output": model,
                    "device": chosen,
                    "gpu": worker_env["adapter"]["gpu"],
                    "adapter": worker_env["adapter"],
                }
                log.write(f"Finished {model}: {row['status']}\n")
        except (Exception, KeyboardInterrupt) as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = str(exc) or type(exc).__name__
            log.write(f"{status}: {error}\n")
        if status == "insufficient_samples":
            error = f"共通の本計測フレームが不足: {len(plan.measured_frames)}/{samples}"
        summary = {
            "schema_version": "1.0",
            "status": status,
            "error": error,
            "models": rows,
            "wall_ms": (perf_counter_ns() - started) / 1e6,
        }
        log.write(f"Benchmark ended: {status}\n")
    write_json(output / "environment.json", environment)
    write_json(output / "summary.json", summary)
    _report(output, rows, summary, environment)
    return summary
