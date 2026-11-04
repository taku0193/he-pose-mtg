"""Opt-in DETRPose GPU smoke: official API parity, image and recorded video."""

import argparse
import json
import socket
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from hepose.adapters import InputFrame
from hepose.cli import main as cli_main
from hepose.config import load_config
from hepose.detrpose import DetrposeAdapter
from hepose.render import draw_poses
from hepose.schema import FrameResult
from hepose.video import VideoReader, VideoWriter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "hepose.local.toml")
    parser.add_argument("--image", type=Path, default=ROOT / "work/yolo26-smoke/bus.jpg")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--video", type=Path, help="既存の実動画を最後まで処理する")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    settings = load_config(args.config).model("detrpose")
    device = settings.get("device", "cuda:0")
    if not device.startswith("cuda:"):
        parser.error("this smoke requires an explicitly assigned CUDA device")
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("official sample image is missing or unreadable")
    height, width = image.shape[:2]
    output = args.output or ROOT / "outputs" / f"detrpose-smoke-{uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    attempted_network = []
    original_connect = socket.socket.connect

    def offline_connect(connection, address):
        if connection.family in {socket.AF_INET, socket.AF_INET6}:
            attempted_network.append(str(address))
            raise RuntimeError("Internet access is disabled during inference smoke")
        return original_connect(connection, address)

    socket.socket.connect = offline_connect
    adapter = None
    try:
        adapter = DetrposeAdapter(device, settings)
        prediction = adapter.predict([InputFrame(0, 0, width, height, image)], 0)
        if not prediction.people:
            raise AssertionError("official human sample produced no people")
        official = adapter._model(source=image, thrh=0.5, img_size=640, save=False, show=False)
        converted = np.array(
            [[[p.x, p.y] for p in person.keypoints] for person in prediction.people]
        )
        np.testing.assert_allclose(converted, official["keypoints"], rtol=1e-5, atol=1e-3)
        (output / "official-image-result.json").write_text(
            json.dumps(
                {key: official[key].tolist() for key in ("scores", "labels", "keypoints")}, indent=2
            )
            + "\n"
        )
        if not cv2.imwrite(
            str(output / "official-image-annotated.jpg"), official["annotated_image"]
        ):
            raise RuntimeError("failed to save official annotated image")
        result = FrameResult(
            "image-smoke", "detrpose", 0, 0, width, height, people=prediction.people
        )
        if not cv2.imwrite(str(output / "image-annotated.jpg"), draw_poses(image, result)):
            raise RuntimeError("failed to write annotated image")
        (output / "image-result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
        (output / "image-environment.json").write_text(
            json.dumps(adapter.environment(), indent=2) + "\n"
        )
        adapter.close()
        adapter = None
        if args.video:
            video = args.video.resolve()
            with VideoReader(video) as source:
                expected_frames = sum(1 for _ in source)
            input_kind = "recorded motion validation video; not a benchmark"
        else:
            video = output / "repeated-image-input.mp4"
            with VideoWriter(video, 10, (width, height)) as writer:
                for _ in range(args.frames):
                    writer.write(image)
            expected_frames = args.frames
            input_kind = "official still image repeated; not recorded motion or benchmark"
        run_output = output / "video"
        exit_code = cli_main(
            [
                "run",
                "--local",
                "--config",
                str(args.config),
                "--model",
                "detrpose",
                "--input",
                str(video),
                "--output",
                str(run_output),
                "--device",
                device,
            ]
        )
        if exit_code:
            raise RuntimeError(f"video CLI failed with code {exit_code}")
        summary = json.loads((run_output / "summary.json").read_text())
        rows = [json.loads(line) for line in (run_output / "poses.jsonl").read_text().splitlines()]
        with VideoReader(run_output / "annotated.mp4") as reader:
            decoded = sum(1 for _ in reader)
        if decoded != expected_frames or len(rows) != expected_frames:
            raise AssertionError("input/output frame counts do not match")
        if any(row["frame_index"] != index for index, row in enumerate(rows)):
            raise AssertionError("source frame identities do not match")
        if attempted_network:
            raise AssertionError(f"unexpected network attempts: {attempted_network}")
        report = {
            "status": "success",
            "device": device,
            "image_people": len(prediction.people),
            "official_api_parity": "keypoints match within rtol=1e-5, atol=1e-3 pixels",
            "video_frames": decoded,
            "summary_status": summary["status"],
            "internet_connection_attempts": attempted_network,
            "input_kind": input_kind,
            "video_frames_with_people": sum(bool(row["people"]) for row in rows),
            "output": str(output),
        }
        (output / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        if adapter is not None:
            adapter.close()
        socket.socket.connect = original_connect


if __name__ == "__main__":
    main()
