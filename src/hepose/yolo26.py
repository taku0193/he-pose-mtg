"""Pinned, local-checkpoint-only YOLO26n-Pose adapter; model imports remain lazy."""

import gc
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path

from hepose.adapters import Prediction
from hepose.runtime import validate_runtime_device as validate_runtime_device
from hepose.schema import COCO17, Keypoint, Person, TemporalContext, finite, integer

ULTRALYTICS_VERSION = "8.4.202"
TORCH_VERSION = "2.14.1"
TORCHVISION_VERSION = "0.29.1"
UPSTREAM_COMMIT = "ae725d5cb20c2d849b864cd34fc15f5c55f877a5"
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt"
WEIGHTS_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
WEIGHTS_SIZE = 7878574
KEYPOINT_THRESHOLD = 0.5


def verify_checkpoint(path: Path, *, expected_sha256: str = WEIGHTS_SHA256) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO26のローカル重みがありません: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"YOLO26重みのSHA256が一致しません: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size": path.stat().st_size,
        "source": WEIGHTS_URL,
    }


def runtime_versions() -> dict:
    result = {}
    for name in ("ultralytics", "torch", "torchvision"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def check_versions(versions: dict) -> None:
    expected = {
        "ultralytics": ULTRALYTICS_VERSION,
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
    }
    for name, version in expected.items():
        actual = versions.get(name)
        if actual is None or actual.split("+")[0] != version:
            raise RuntimeError(
                f"モデル環境の依存版が一致しません: {name}={actual}, expected={version}"
            )


def model_settings(settings: dict) -> tuple[Path, int]:
    if not settings.get("weights"):
        raise FileNotFoundError("YOLO26のローカル重みを設定してください（weights未指定）。")
    size = settings.get("input_size", 640)
    integer(size, "input_size", 32)
    if size % 32:
        raise ValueError("input_size must be a multiple of 32")
    return Path(settings["weights"]), size


def inspect_yolo26(settings: dict, device: str) -> dict:
    report = {"status": "weights_missing", "dependencies": runtime_versions()}
    try:
        weights, size = model_settings(settings)
        report["weights"] = verify_checkpoint(weights)
        report["input_size"] = size
        if any(version is None for version in report["dependencies"].values()):
            report["status"] = "dependency_missing"
            return report
        check_versions(report["dependencies"])
        runtime = importlib.import_module("torch")
        report["device_info"] = validate_runtime_device(device, runtime, settings.get("gpu_uuid"))
        report["status"] = "ready"
    except FileNotFoundError as exc:
        report["detail"] = str(exc)
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        report.update(status="unavailable", detail=str(exc))
    return report


def _as_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def parse_result(result, width: int, height: int, *, measurement_method=None) -> Prediction:
    if tuple(result.orig_shape) != (height, width):
        raise ValueError("YOLO26結果の元画像サイズが入力と一致しません。")
    boxes = _as_list(result.boxes.xyxy) if result.boxes is not None else []
    data = _as_list(result.keypoints.data) if result.keypoints is not None else []
    if len(boxes) != len(data):
        raise ValueError("人物とキーポイントの件数が一致しません。")
    people = []
    for bbox, joints in zip(boxes, data, strict=True):
        if len(joints) != len(COCO17):
            raise ValueError("YOLO26n-Poseの関節数がCOCO17と一致しません。")
        points = []
        for name, raw in zip(COCO17, joints, strict=True):
            if len(raw) not in {2, 3}:
                raise ValueError("キーポイントの次元が不正です。")
            x, y = float(raw[0]), float(raw[1])
            finite(x, "keypoint x")
            finite(y, "keypoint y")
            confidence = float(raw[2]) if len(raw) == 3 else None
            if confidence is not None:
                finite(confidence, "confidence")
                if confidence < KEYPOINT_THRESHOLD:
                    x = y = None
            points.append(
                Keypoint(
                    name,
                    x,
                    y,
                    confidence,
                    "Ultralytics keypoint confidence" if confidence is not None else None,
                    None,
                )
            )
        people.append(Person(tuple(float(value) for value in bbox), tuple(points)))
    model_ms = (getattr(result, "speed", None) or {}).get("inference")
    model_ms = float(model_ms) if model_ms is not None else None
    method = (measurement_method or "Ultralytics Profile") if model_ms is not None else None
    return Prediction(tuple(people), model_ms, method)


class Yolo26Adapter:
    model_id = "yolo26"
    temporal = TemporalContext()

    def __init__(self, device: str, settings: dict):
        weights, self.input_size = model_settings(settings)
        self.checkpoint = verify_checkpoint(weights)
        self.versions = runtime_versions()
        check_versions(self.versions)
        state = weights.parent / "runtime"
        state.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_CONFIG_DIR"] = str(state)
        os.environ["MPLCONFIGDIR"] = str(state / "matplotlib")
        os.environ["CUDA_CACHE_PATH"] = str(state / "cuda-cache")
        self.runtime = importlib.import_module("torch")
        self.device_info = validate_runtime_device(device, self.runtime, settings.get("gpu_uuid"))
        self.device = device
        self.closed = False
        if device.startswith("cuda:"):
            self.runtime.cuda.reset_peak_memory_stats(self.runtime.device(device))
        self._model = importlib.import_module("ultralytics").YOLO(str(weights), task="pose")
        try:
            if self._model.task != "pose":
                raise ValueError("ロードしたモデルは姿勢推定モデルではありません。")
            self._model.to(self.runtime.device(device))
            if device.startswith("cuda:"):
                self.runtime.cuda.synchronize(self.runtime.device(device))
        except BaseException:
            self.close()
            raise

    def predict(self, frames, target_frame_index: int) -> Prediction:
        if self.closed:
            raise RuntimeError("YOLO26アダプタは閉じられています。")
        if len(frames) != 1 or frames[0].frame_index != target_frame_index:
            raise ValueError("YOLO26には推定対象の1フレームを渡してください。")
        frame = frames[0]
        if frame.image.shape != (frame.height, frame.width, 3):
            raise ValueError("入力画像サイズが一致しません。")
        results = self._model.predict(
            source=frame.image,
            device=self.runtime.device(self.device),
            imgsz=self.input_size,
            rect=False,
            quantize=32,
            batch=1,
            verbose=False,
            save=False,
            show=False,
            stream=False,
            nms=False,
        )
        if len(results) != 1:
            raise ValueError("YOLO26の応答件数が入力と一致しません。")
        method = (
            "Ultralytics Profile: CUDA synchronization around model forward"
            if self.device.startswith("cuda:")
            else "Ultralytics Profile: host perf_counter"
        )
        return parse_result(results[0], frame.width, frame.height, measurement_method=method)

    def environment(self) -> dict:
        peak = (
            int(self.runtime.cuda.max_memory_allocated(self.runtime.device(self.device)))
            if self.device.startswith("cuda:")
            else None
        )
        return {
            "model_id": self.model_id,
            "architecture": "YOLO26n-Pose",
            "backend": "PyTorch",
            "device": self.device,
            "gpu": self.device_info,
            "precision": str(next(self._model.model.parameters()).dtype).removeprefix("torch."),
            "packages": self.versions,
            "upstream_commit": UPSTREAM_COMMIT,
            "weights": [self.checkpoint],
            "input_size": [self.input_size, self.input_size],
            "rect": False,
            "nms": False,
            "confidence_threshold": 0.25,
            "keypoint_threshold": KEYPOINT_THRESHOLD,
            "license": "AGPL-3.0",
            "gpu_peak_memory_bytes": peak,
            "memory_scope": "current process PyTorch allocator since adapter initialization",
            "warmup": "official first-predict internal warmup; no input frames discarded",
        }

    def close(self) -> None:
        self._model = None
        self.closed = True
        gc.collect()
