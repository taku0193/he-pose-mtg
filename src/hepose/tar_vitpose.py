"""Official TAR-ViTPose integration, with explicit native joints and original-frame windows."""

import gc
import importlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

from hepose.adapters import Prediction
from hepose.runtime import validate_runtime_device, verify_asset
from hepose.schema import Keypoint, Person, TemporalContext, coco17, finite
from hepose.timing import summarize

UPSTREAM_COMMIT = "42164c2c789a5297544dd836846e509459b0f1f3"
SOURCE_CONFIG = "configs/posetrack17/configTARViTPoseVitB.yaml"
VIT_CONFIG = "models/vitpose/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
ARTIFACTS = {
    "weights": {
        "filename": "tarvitpose_b_17.pt",
        "size": 1619532723,
        "sha256": "89387c8aded9044471163ae52f2ec970c23ccd7f6c7fff5f47dd6cdb33e87c47",
        "url": "https://drive.google.com/file/d/1mn-P-5bDa0OrM4ctc8PYrMuh40o3l-8J/view",
    },
    "pretrained": {
        "filename": "pretrain_model_b.pth",
        "size": 360049073,
        "sha256": "13926fe2a435db7752a7da5e4f288df97dec9d07501b192793b4cdf97cbd07ff",
        "url": "https://drive.google.com/file/d/1S6e-bFgp1qYermBJXoI9Pwt9-Qg1DBuq/view",
    },
    "detector_weights": {
        "filename": "yolo11x.pt",
        "size": 114636239,
        "sha256": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    },
}
PACKAGE_VERSIONS = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
    "mmcv": "2.1.0",
    "mmengine": "0.10.4",
    "mmpose": "1.3.1",
    "mmpretrain": "1.2.0",
    "mmdet": "3.2.0",
    "ultralytics": "8.4.202",
    "numpy": "1.26.4",
    "setuptools": "75.8.0",
}

NATIVE_JOINTS = (
    "nose",
    "head_bottom",
    "head_top",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
USED_JOINTS = (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def decode_person(heatmaps, crop_xyxy, detector_bbox) -> Person:
    import numpy as np

    if (
        len(heatmaps.shape) != 4
        or heatmaps.shape[:2] != (1, 17)
        or not heatmaps.shape[2]
        or not heatmaps.shape[3]
    ):
        raise ValueError("TARのヒートマップは1×17×H×Wです。")
    if not np.isfinite(heatmaps).all():
        raise ValueError("TARのヒートマップに非有限値があります。")
    x1, y1, x2, y2 = crop_xyxy
    if not np.isfinite(crop_xyxy).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("TARのクロップ領域が不正です。")
    height, width = heatmaps.shape[2:]
    native = []
    for index, name in enumerate(NATIVE_JOINTS):
        if index not in USED_JOINTS:
            native.append(Keypoint(name))
            continue
        position = int(heatmaps[0, index].argmax())
        x = x1 + (position % width) * (x2 - x1) / width
        y = y1 + (position // width) * (y2 - y1) / height
        native.append(Keypoint(name, float(x), float(y)))
    points = tuple(native)
    return Person(
        tuple(detector_bbox),
        coco17(points),
        native_keypoints=points,
        native_joint_definition="posetrack17_17slots",
        crop_xyxy=tuple(crop_xyxy),
    )


def validate_window(frames, target_frame_index):
    expected = list(range(target_frame_index - 2, target_frame_index + 3))
    if len(frames) != 5 or [frame.frame_index for frame in frames] != expected:
        raise ValueError("TARには中心の前後2枚を含む連続5フレームが必要です。")
    center = frames[2]
    if any(
        frame.width != center.width
        or frame.height != center.height
        or frame.image.shape != (center.height, center.width, 3)
        for frame in frames
    ):
        raise ValueError("TARの参照画像サイズが一致しません。")
    return center


def effective_config(repository: Path, pretrained: Path) -> dict:
    """Use the official Base settings, with recorded defaults and prepared local paths."""
    import yaml

    source = yaml.safe_load((repository / SOURCE_CONFIG).read_text())
    keys = (
        "METHOD",
        "NUM_JOINTS",
        "IMAGE_SIZE",
        "EMBED_DIM",
        "HEATMAP_SIZE",
        "MASK_THRESHOLD",
        "NUM_LAYERS",
    )
    model = {key: source["MODEL"][key] for key in keys}
    model.update(
        {
            "CONFIG_FILE": str((repository / VIT_CONFIG).resolve()),
            "CHECKPOINT_FILE": str(pretrained.resolve()),
            "WINDOWS_SIZE": 5,
            "USE_MASK": True,
            "USE_FEATURE_FUSION": False,
        }
    )
    return {
        "SEED": source.get("SEED", 42),
        "MODEL": model,
        "DATASET": {"BBOX_ENLARGE_FACTOR": source["DATASET"]["BBOX_ENLARGE_FACTOR"]},
    }


def verify_repository(repository: Path) -> dict:
    if not repository.is_dir():
        raise FileNotFoundError(f"TARの公式リポジトリがありません: {repository}")
    command = ["git", "-C", str(repository)]
    commit = subprocess.check_output([*command, "rev-parse", "HEAD"], timeout=10).decode().strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"TARのコミットが一致しません: {commit}")
    diff = subprocess.check_output(
        [*command, "diff", "--no-ext-diff", "--no-textconv", "HEAD", "--"], timeout=10
    )
    if diff:
        raise ValueError("TARの公式ソースに未記録の変更があります。")
    unknown = subprocess.check_output(
        [*command, "ls-files", "--others", "--exclude-standard"], timeout=10
    )
    if unknown.strip():
        raise ValueError("TARの公式ソースに未記録のファイルがあります。")
    return {"path": str(repository.resolve()), "commit": commit, "patches": []}


def local_assets(settings: dict) -> tuple[dict, Path, dict]:
    for name in (*ARTIFACTS, "repository", "config"):
        if not settings.get(name):
            raise FileNotFoundError(f"TARのローカル{name}を設定してください。")
    if "input_size" in settings:
        raise ValueError("TARの入力は公式Base設定の288×384固定です。")
    records = {
        key: verify_asset(Path(settings[key]), value["sha256"], value["url"])
        for key, value in ARTIFACTS.items()
    }
    repository = Path(settings["repository"])
    source = verify_repository(repository)
    import yaml

    config_path = Path(settings["config"])
    config = yaml.safe_load(config_path.read_text())
    if config != effective_config(repository, Path(settings["pretrained"])):
        raise ValueError("TARの推論設定が記録した公式Base設定と一致しません。")
    import hashlib

    records["config"] = {
        "path": str(config_path.resolve()),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    return records, repository, source


def runtime_versions() -> dict:
    result = {}
    for name in PACKAGE_VERSIONS:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def check_versions(versions: dict):
    for name, expected in PACKAGE_VERSIONS.items():
        actual = versions.get(name)
        if actual is None or actual.split("+")[0] != expected:
            raise RuntimeError(f"TARの依存版が一致しません: {name}={actual}, expected={expected}")


def inspect_tar(settings: dict, device: str) -> dict:
    report = {"status": "weights_missing", "dependencies": runtime_versions()}
    try:
        records, _, source = local_assets(settings)
        report.update(artifacts=records, upstream=source)
        check_versions(report["dependencies"])
        runtime = importlib.import_module("torch")
        report["device_info"] = validate_runtime_device(device, runtime, settings.get("gpu_uuid"))
        report["status"] = "ready"
    except FileNotFoundError as exc:
        report["detail"] = str(exc)
    except (RuntimeError, ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
        report.update(status="unavailable", detail=str(exc))
    return report


class TarVitposeAdapter:
    model_id = "tar-vitpose"
    temporal = TemporalContext(window_size=5, step=1, future_frames=2)

    def __init__(self, device: str, settings: dict):
        self.artifacts, repository, self.source = local_assets(settings)
        self.versions = runtime_versions()
        check_versions(self.versions)
        state = Path(settings["weights"]).parent / "runtime"
        state.mkdir(parents=True, exist_ok=True)
        for name, directory in {
            "MPLCONFIGDIR": "matplotlib",
            "YOLO_CONFIG_DIR": "yolo",
            "TORCH_HOME": "torch",
            "CUDA_CACHE_PATH": "cuda-cache",
        }.items():
            os.environ[name] = str(state / directory)
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        self.runtime = importlib.import_module("torch")
        self.device_info = validate_runtime_device(device, self.runtime, settings.get("gpu_uuid"))
        self.device = device
        self.closed = False
        self._model = self._detector = None
        self._model_calls = 0
        self._detector_times = []
        self.runtime.default_generator.manual_seed(42)
        self.runtime.backends.cudnn.deterministic = True
        self.runtime.backends.cudnn.benchmark = False
        if device.startswith("cuda:"):
            with self.runtime.cuda.device(self.runtime.device(device)):
                self.runtime.cuda.manual_seed(42)
        sys.path.insert(0, str(repository))
        try:
            if device.startswith("cuda:"):
                self.runtime.cuda.reset_peak_memory_stats(self.runtime.device(device))
            spec = importlib.util.spec_from_file_location(
                "_hepose_tar_inference", repository / "tools/inference.py"
            )
            self._upstream = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._upstream)
            model_module = importlib.import_module(self._upstream.TAR_ViTPose.__module__)
            if (
                Path(model_module.__file__).resolve()
                != (repository / "models/best/TAR_ViTPose.py").resolve()
            ):
                raise ValueError("ロードしたTARモデルの場所が公式ソースと一致しません。")
            cfg = self._upstream.load_cfg(settings["config"])
            self._model = self._upstream.TAR_ViTPose(cfg, device=device, phase="test")
            checkpoint = self.runtime.load(
                settings["weights"], map_location="cpu", weights_only=True
            )
            self._model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            self._model.to(device).float().eval()
            self._detector = importlib.import_module("ultralytics").YOLO(
                settings["detector_weights"], task="detect"
            )
            self._detector.to(self.runtime.device(device))
            if device.startswith("cuda:"):
                self.runtime.cuda.synchronize(self.runtime.device(device))
        except BaseException:
            self.close()
            raise
        finally:
            sys.path.remove(str(repository))

    def predict(self, frames, target_frame_index):
        if self.closed:
            raise RuntimeError("TARアダプタは閉じられています。")
        center = validate_window(frames, target_frame_index)
        detections = self._detector.predict(
            source=center.image,
            device=self.runtime.device(self.device),
            classes=[0],
            imgsz=640,
            rect=True,
            quantize=32,
            verbose=False,
            save=False,
            show=False,
        )
        if len(detections) != 1 or tuple(detections[0].orig_shape) != (center.height, center.width):
            raise ValueError("TAR人物検出の元画像サイズが一致しません。")
        self._detector_times.append((detections[0].speed or {}).get("inference"))
        people, model_times = [], []
        boxes = detections[0].boxes.xyxy.detach().cpu().tolist()
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            for value in bbox:
                finite(value, "detector bbox")
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = (x2 - x1) * 1.25, (y2 - y1) * 1.25
            crop = (
                int(max(cx - width / 2, 0)),
                int(max(cy - height / 2, 0)),
                int(min(cx + width / 2, center.width)),
                int(min(cy + height / 2, center.height)),
            )
            if crop[2] <= crop[0] or crop[3] <= crop[1]:
                raise ValueError("TARの人物クロップが空です。")
            images = [
                self._upstream.preprocess_frame(
                    frame.image[crop[1] : crop[3], crop[0] : crop[2]], (288, 384)
                )
                for frame in frames
            ]
            data = self.runtime.stack(images).unsqueeze(0).to(self.device)
            with self.runtime.inference_mode():
                if self.device.startswith("cuda:"):
                    stream = self.runtime.cuda.current_stream(self.runtime.device(self.device))
                    start, end = (
                        self.runtime.cuda.Event(enable_timing=True),
                        self.runtime.cuda.Event(enable_timing=True),
                    )
                    start.record(stream)
                    heatmaps = self._model(data)
                    end.record(stream)
                    end.synchronize()
                    model_times.append(start.elapsed_time(end))
                else:
                    started = perf_counter_ns()
                    heatmaps = self._model(data)
                    model_times.append((perf_counter_ns() - started) / 1e6)
            self._model_calls += 1
            people.append(decode_person(heatmaps.detach().cpu().numpy(), crop, bbox))
        method = (
            "CUDA events around TAR forwards summed over persons; detector/crops excluded"
            if self.device.startswith("cuda:")
            else "host clock around TAR forwards summed over persons"
        )
        return Prediction(
            tuple(people),
            sum(model_times) if model_times else None,
            method if model_times else None,
        )

    def environment(self) -> dict:
        peak = (
            int(self.runtime.cuda.max_memory_allocated(self.runtime.device(self.device)))
            if self.device.startswith("cuda:")
            else None
        )
        return {
            "model_id": self.model_id,
            "architecture": "TAR-ViTPose Base",
            "backend": "PyTorch",
            "device": self.device,
            "gpu": self.device_info,
            "packages": self.versions,
            "precision": str(next(self._model.parameters()).dtype).removeprefix("torch."),
            "upstream": self.source,
            "artifacts": self.artifacts,
            "input_size": [288, 384],
            "temporal": {
                "window_size": 5,
                "step": 1,
                "future_frames": 2,
                "boundary": "not_estimated",
            },
            "bbox_enlarge_factor": 1.25,
            "num_heads": 4,
            "num_layers": 6,
            "use_mask": True,
            "use_feature_fusion": "false; loader field unused by official model",
            "seed": 42,
            "seed_scope": "torch CPU and assigned CUDA device",
            "cudnn_deterministic": bool(self.runtime.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(self.runtime.backends.cudnn.benchmark),
            "joint_definition": "posetrack17_17slots",
            "joint_confidence": "unavailable: null",
            "visibility": "unavailable: null",
            "code_license": "Apache-2.0",
            "weights_license": "no separate declaration found in official Drive folder",
            "detector": {
                "architecture": "YOLO11x",
                "input_size": 640,
                "rect": True,
                "precision": "float32",
                "confidence": 0.25,
                "iou": 0.7,
                "license": "AGPL-3.0",
            },
            "model_forward_calls": self._model_calls,
            "detector_inference": summarize(self._detector_times),
            "timing_scope": (
                "inference_ms=TAR forward only; adapter_ms also includes detector and crops"
            ),
            "gpu_peak_memory_bytes": peak,
            "memory_scope": "current process PyTorch allocator since adapter initialization",
            "warmup": "detector official first-predict warmup only; no input frames discarded",
        }

    def close(self):
        self._model = self._detector = None
        self.closed = True
        gc.collect()
