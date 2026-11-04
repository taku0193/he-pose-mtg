"""Opt-in actual-GPU TAR window, common video and official CLI validation; not a benchmark."""

import argparse
import json
import os
import runpy
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from hepose.cli import main as cli_main
from hepose.config import load_config
from hepose.render import draw_poses
from hepose.runtime import validate_runtime_device
from hepose.schema import FrameResult
from hepose.tar_vitpose import USED_JOINTS, TarVitposeAdapter, local_assets
from hepose.video import VideoReader

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def offline():
    original = socket.socket.connect
    attempts = []

    def connect(connection, address):
        if connection.family in {socket.AF_INET, socket.AF_INET6}:
            attempts.append(str(address))
            raise RuntimeError("Internet access is blocked during TAR validation")
        return original(connection, address)

    socket.socket.connect = connect
    try:
        yield attempts
        if attempts:
            raise AssertionError(f"unexpected network attempts: {attempts}")
    finally:
        socket.socket.connect = original


def window_check(adapter, frames, output):
    prediction = adapter.predict(frames, frames[2].frame_index)
    if not prediction.people:
        raise AssertionError("recorded human validation window produced no people")
    person = prediction.people[0]
    x1, y1, x2, y2 = (int(v) for v in person.crop_xyxy)
    data = (
        adapter.runtime.stack(
            [
                adapter._upstream.preprocess_frame(frame.image[y1:y2, x1:x2], (288, 384))
                for frame in frames
            ]
        )
        .unsqueeze(0)
        .to(adapter.device)
    )
    with adapter.runtime.no_grad():
        heatmaps = adapter._model(data)
    original = adapter._upstream.extract_kps(heatmaps, y2 - y1, x2 - x1)
    expected = original + np.array([x1, y1])
    actual = np.array(
        [[person.native_keypoints[i].x, person.native_keypoints[i].y] for i in USED_JOINTS]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-3)
    result = FrameResult(
        "window-smoke",
        "tar-vitpose",
        frames[2].frame_index,
        frames[2].source_time_ms,
        frames[2].width,
        frames[2].height,
        people=prediction.people,
        temporal=adapter.temporal,
        reference_frame_indices=tuple(f.frame_index for f in frames),
    )
    (output / "window-result.json").write_text(result.to_json() + "\n")
    if not cv2.imwrite(str(output / "window-annotated.jpg"), draw_poses(frames[2].image, result)):
        raise RuntimeError("failed to save actual window annotation")
    return {
        "people": len(prediction.people),
        "target_frame": frames[2].frame_index,
        "reference_frames": list(result.reference_frame_indices),
        "official_crop_decoder_max_abs_error_px": float(np.abs(actual - expected).max()),
    }


def run_official(args, settings):
    args.output.mkdir(parents=True, exist_ok=False)
    records, repository, source = local_assets(settings)
    state = Path(settings["weights"]).parent / "runtime"
    for name, directory in {
        "MPLCONFIGDIR": "matplotlib",
        "YOLO_CONFIG_DIR": "yolo",
        "TORCH_HOME": "torch",
        "CUDA_CACHE_PATH": "cuda-cache",
    }.items():
        os.environ[name] = str(state / directory)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    with offline() as attempts:
        import torch

        device_info = validate_runtime_device("cuda:0", torch, settings.get("gpu_uuid"))
        if torch.cuda.device_count() != 1:
            raise ValueError("official CLI smoke must expose only the assigned GPU")
        sys.path.insert(0, str(repository))
        # The untouched official CLI finds yolo11x.pt in this prepared directory.
        os.chdir(Path(settings["detector_weights"]).parent)
        sys.argv = [
            str(repository / "tools/inference.py"),
            "-c",
            settings["config"],
            "-w",
            settings["weights"],
            "-i",
            str(args.video),
            "-o",
            str(args.output / "annotated.mp4"),
            "--coco_json",
            str(args.output / "official-predictions.json"),
            "-g",
            "0",
        ]
        runpy.run_path(str(repository / "tools/inference.py"), run_name="__main__")
    with VideoReader(args.video) as reader:
        source_frames = sum(1 for _ in reader)
    with VideoReader(args.output / "annotated.mp4") as reader:
        result_frames = sum(1 for _ in reader)
    if result_frames != source_frames - 4:
        raise AssertionError("unexpected official five-frame window output count")
    report = {
        "status": "success",
        "source": source,
        "artifacts": records,
        "device": device_info,
        "input_frames": source_frames,
        "output_frames": result_frames,
        "source_index_mapping": "output_index + 2; derived from pinned official loop",
        "internet_connection_attempts": attempts,
        "upstream_source_patches": [],
        "configuration": settings["config"],
        "raw_json_warning": "official crop-relative joints and fixed visibility retained",
    }
    (args.output / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def compare_official(output, rows):
    raw = json.loads((output / "upstream/official-predictions.json").read_text())
    groups = {}
    for annotation in raw["annotations"]:
        groups.setdefault(annotation["image_id"], []).append(annotation)
    maximum, people, points, differing_points = 0.0, 0, 0, 0
    for output_index in range(len(raw["images"])):
        predictions = rows[output_index + 2]["people"]
        annotations = groups.get(output_index, [])
        if len(predictions) != len(annotations):
            raise AssertionError("common and official person counts differ")
        for person, annotation in zip(predictions, annotations, strict=True):
            people += 1
            x, y, width, height = annotation["bbox"]
            if person["crop_xyxy"] != [x, y, x + width, y + height]:
                raise AssertionError(f"official crop differs at source frame {output_index + 2}")
            for index in USED_JOINTS:
                point = person["native_keypoints"][index]
                error = max(
                    abs(point["x"] - annotation["keypoints"][index * 3] - x),
                    abs(point["y"] - annotation["keypoints"][index * 3 + 1] - y),
                )
                maximum = max(maximum, error)
                differing_points += error > 0.001
                points += 1
    return {
        "center_frames": len(raw["images"]),
        "people": people,
        "points": points,
        "crop_match": True,
        "max_abs_coordinate_error_px": maximum,
        "points_above_0_001_px": differing_points,
        "exact_coordinate_match": differing_points == 0,
        "scope": "independent GPU executions; coordinate variation is observed, not a GT metric",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "hepose.local.toml")
    parser.add_argument(
        "--video", type=Path, default=ROOT / "assets/sample/jumping-jacks-validation.mp4"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-start", type=int, default=50)
    parser.add_argument("--upstream-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config, args.video, args.output = (
        args.config.resolve(),
        args.video.resolve(),
        args.output.resolve(),
    )
    settings = load_config(args.config).model("tar-vitpose")
    if args.upstream_only:
        run_official(args, settings)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    device = settings.get("device", "cuda:0")
    if not device.startswith("cuda:"):
        raise ValueError("this smoke requires an explicitly assigned CUDA device")
    with VideoReader(args.video) as reader:
        frames = []
        for frame in reader:
            if args.window_start <= frame.frame_index < args.window_start + 5:
                frames.append(frame)
    source_frames = reader.decoded
    if len(frames) != 5:
        raise ValueError("validation window is outside the source video")
    with offline() as attempts:
        adapter = TarVitposeAdapter(device, settings)
        try:
            window = window_check(adapter, frames, args.output)
            (args.output / "window-environment.json").write_text(
                json.dumps(adapter.environment(), indent=2) + "\n"
            )
            gpu_uuid = adapter.device_info["gpu_uuid"]
        finally:
            adapter.close()
        common = args.output / "common"
        code = cli_main(
            [
                "run",
                "--local",
                "--config",
                str(args.config),
                "--model",
                "tar-vitpose",
                "--input",
                str(args.video),
                "--output",
                str(common),
                "--device",
                device,
            ]
        )
        if code:
            raise RuntimeError(f"common TAR CLI failed: {code}")
    summary = json.loads((common / "summary.json").read_text())
    rows = [json.loads(line) for line in (common / "poses.jsonl").read_text().splitlines()]
    with VideoReader(common / "annotated.mp4") as reader:
        decoded = sum(1 for _ in reader)
    if len(rows) != source_frames or decoded != source_frames:
        raise AssertionError("common output does not preserve the source timeline")
    boundaries = [0, 1, source_frames - 2, source_frames - 1]
    for index, row in enumerate(rows):
        if row["frame_index"] != index:
            raise AssertionError("source frame identity does not match")
        if index in boundaries:
            assert row["status"] == "not_estimated"
        else:
            assert row["status"] == "success"
            assert row["reference_frame_indices"] == list(range(index - 2, index + 3))
    assert summary["frame_counts"] == {
        "success": source_frames - 4,
        "failed": 0,
        "not_estimated": 4,
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--video",
        str(args.video),
        "--output",
        str(args.output / "upstream"),
        "--upstream-only",
    ]
    subprocess.run(
        command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid), check=True
    )
    full_parity = compare_official(args.output, rows)
    report = {
        "status": "success",
        "input_kind": "recorded motion; not comparative benchmark",
        "device": device,
        "gpu_uuid": gpu_uuid,
        "window": window,
        "input_frames": source_frames,
        "common_video_frames": decoded,
        "frame_counts": summary["frame_counts"],
        "boundary_indices": boundaries,
        "frames_with_people": sum(bool(row["people"]) for row in rows),
        "internet_connection_attempts": attempts,
        "official_cli": "upstream/smoke-report.json",
        "official_output_comparison": full_parity,
    }
    (args.output / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
