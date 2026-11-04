"""Pinned official DETRPose-N with local safetensors; model imports remain lazy."""

import gc
import hashlib
import importlib
import importlib.metadata
import os
import subprocess
from pathlib import Path
from time import perf_counter_ns

from hepose.adapters import Prediction
from hepose.runtime import validate_runtime_device, verify_asset
from hepose.schema import COCO17, Keypoint, Person, TemporalContext, finite

UPSTREAM_COMMIT = "da12e0cd29864b1d3a289c562c34d423371a4157"
HF_REVISION = "8e11f255908c7869d9a257b2669a9012957b6d0a"
HF_BASE = f"https://huggingface.co/SebasJanampa/DETRPose_N_COCO/resolve/{HF_REVISION}"
WEIGHTS_URL = f"{HF_BASE}/model.safetensors"
WEIGHTS_SHA256 = "b618ccec659b42e92372290645c3c4712813571ecda914bf3ffa52d1173e20fc"
WEIGHTS_SIZE = 16677992
CONFIG_URL = f"{HF_BASE}/config.json"
CONFIG_SHA256 = "eb2801c0575dfff75b9d18a5c88aae6a94f2db7b91e29b40718cee171c8f0343"
PATCH_SHA256 = "20dbd6a0e13e4cd496cee5a7c7f90428848db376bcc07eece0f43091adef32eb"
CONFIG_PATH = "detrpose/configs/detrpose/detrpose_hgnetv2_n.py"
PERSON_THRESHOLD = 0.5


def verify_repository(path: Path) -> dict:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"DETRPoseの公式リポジトリがありません: {path}")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], timeout=10)

    commit = git("rev-parse", "HEAD").decode().strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"DETRPoseのコミットが一致しません: {commit}")
    diff = git("diff", "--no-ext-diff", "--no-textconv", "HEAD", "--")
    patch_hash = hashlib.sha256(diff).hexdigest()
    if patch_hash != PATCH_SHA256:
        raise ValueError("DETRPoseの上流パッチが一致しません。")
    unknown = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    if any(".egg-info/" not in name and "__pycache__/" not in name for name in unknown):
        raise ValueError("DETRPoseに未記録のファイルがあります。")
    return {"path": str(path.resolve()), "commit": commit, "patch_sha256": patch_hash}


def local_settings(settings: dict) -> tuple[Path, Path]:
    if not settings.get("weights") or not settings.get("repository"):
        raise FileNotFoundError("DETRPoseのローカルweightsとrepositoryを設定してください。")
    if settings.get("input_size", 640) != 640:
        raise ValueError("固定したDETRPose-Nの入力サイズは640です。")
    repo = Path(settings["repository"])
    if (
        settings.get("config")
        and Path(settings["config"]).resolve() != (repo / CONFIG_PATH).resolve()
    ):
        raise ValueError("DETRPose-Nの公式COCO設定を指定してください。")
    return Path(settings["weights"]), repo


def runtime_versions() -> dict:
    result = {}
    for name in ("detrpose", "torch", "torchvision", "safetensors", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def check_versions(versions: dict) -> None:
    for name, expected in {"detrpose": "0.0.1", "torch": "2.14.1", "torchvision": "0.29.1"}.items():
        actual = versions.get(name)
        if actual is None or actual.split("+")[0] != expected:
            raise RuntimeError(
                f"DETRPoseの依存版が一致しません: {name}={actual}, expected={expected}"
            )
    if any(value is None for value in versions.values()):
        raise RuntimeError("DETRPoseの依存パッケージが不足しています。")


def inspect_detrpose(settings: dict, device: str) -> dict:
    report = {"status": "weights_missing", "dependencies": runtime_versions()}
    try:
        weights, repo = local_settings(settings)
        report["weights"] = verify_asset(weights, WEIGHTS_SHA256, WEIGHTS_URL)
        report["repository"] = verify_repository(repo)
        check_versions(report["dependencies"])
        runtime = importlib.import_module("torch")
        report["device_info"] = validate_runtime_device(device, runtime, settings.get("gpu_uuid"))
        report["status"] = "ready"
    except FileNotFoundError as exc:
        report["detail"] = str(exc)
    except (RuntimeError, ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
        report.update(status="unavailable", detail=str(exc))
    return report


def _as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def parse_result(result: dict, width: int, height: int, *, inference_ms=None, method=None):
    if tuple(result["original_size"]) != (width, height):
        raise ValueError("DETRPose結果の元画像サイズが一致しません。")
    scores = _as_list(result["scores"])
    labels = _as_list(result["labels"])
    keypoints = _as_list(result["keypoints"])
    if len(scores) != len(labels) or len(scores) != len(keypoints):
        raise ValueError("DETRPose結果の件数が一致しません。")
    people = []
    for score, label, joints in zip(scores, labels, keypoints, strict=True):
        finite(float(score), "person score")
        finite(float(label), "person label")
        if len(joints) != len(COCO17):
            raise ValueError("DETRPose-Nの関節数がCOCO17と一致しません。")
        points = []
        for name, raw in zip(COCO17, joints, strict=True):
            if len(raw) != 2:
                raise ValueError("DETRPose-Nの関節座標は2次元です。")
            points.append(Keypoint(name, float(raw[0]), float(raw[1])))
        if float(score) > PERSON_THRESHOLD:
            people.append(Person(None, tuple(points)))
    return Prediction(tuple(people), inference_ms, method)


class DetrposeAdapter:
    model_id = "detrpose"
    temporal = TemporalContext()

    def __init__(self, device: str, settings: dict):
        weights, repo = local_settings(settings)
        self.checkpoint = verify_asset(weights, WEIGHTS_SHA256, WEIGHTS_URL)
        self.repository = verify_repository(repo)
        self.versions = runtime_versions()
        check_versions(self.versions)
        state = weights.parent / "runtime"
        state.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["HF_HOME"] = str(state / "huggingface")
        os.environ["MPLCONFIGDIR"] = str(state / "matplotlib")
        os.environ["CUDA_CACHE_PATH"] = str(state / "cuda-cache")
        self.runtime = importlib.import_module("torch")
        self.device_info = validate_runtime_device(device, self.runtime, settings.get("gpu_uuid"))
        self.device = device
        self.closed = False
        self._model = None
        try:
            if device.startswith("cuda:"):
                self.runtime.cuda.reset_peak_memory_stats(self.runtime.device(device))
            upstream = importlib.import_module("detrpose")
            if Path(upstream.__file__).resolve() != (repo / "detrpose/__init__.py").resolve():
                raise ValueError("インストール済みDETRPoseの場所が公式リポジトリと一致しません。")
            self._model = upstream.DETR(
                model=str(repo / CONFIG_PATH), device=device, resume=str(weights)
            )
            self._model.model.float().eval()
            transforms = importlib.import_module("torchvision.transforms")
            self.transforms = transforms.Compose(
                [transforms.Resize((640, 640)), transforms.ToTensor()]
            )
            self.to_pil = importlib.import_module("detrpose.engine.io").InputHandler.to_pil_image
            if device.startswith("cuda:"):
                self.runtime.cuda.synchronize(self.runtime.device(device))
        except BaseException:
            self.close()
            raise

    def predict(self, frames, target_frame_index: int) -> Prediction:
        if self.closed:
            raise RuntimeError("DETRPoseアダプタは閉じられています。")
        if len(frames) != 1 or frames[0].frame_index != target_frame_index:
            raise ValueError("DETRPoseには推定対象の1フレームを渡してください。")
        frame = frames[0]
        if frame.image.shape != (frame.height, frame.width, 3):
            raise ValueError("入力画像サイズが一致しません。")
        runtime = self.runtime
        data = self.transforms(self.to_pil(frame.image)).unsqueeze(0).to(self.device)
        original_size = runtime.tensor([[frame.width, frame.height]], device=self.device)
        with runtime.inference_mode():
            if self.device.startswith("cuda:"):
                stream = runtime.cuda.current_stream(runtime.device(self.device))
                start, end = (
                    runtime.cuda.Event(enable_timing=True),
                    runtime.cuda.Event(enable_timing=True),
                )
                start.record(stream)
                outputs = self._model.model.model(data)
                end.record(stream)
                end.synchronize()
                model_ms = start.elapsed_time(end)
                method = "CUDA events around model forward; preprocessing/postprocessing excluded"
            else:
                start_ns = perf_counter_ns()
                outputs = self._model.model.model(data)
                model_ms = (perf_counter_ns() - start_ns) / 1e6
                method = "host perf_counter_ns around model forward"
            scores, labels, keypoints = self._model.model.postprocessor(outputs, original_size)
        return parse_result(
            {
                "scores": scores[0],
                "labels": labels[0],
                "keypoints": keypoints[0],
                "original_size": (frame.width, frame.height),
            },
            frame.width,
            frame.height,
            inference_ms=model_ms,
            method=method,
        )

    def environment(self) -> dict:
        peak = (
            int(self.runtime.cuda.max_memory_allocated(self.runtime.device(self.device)))
            if self.device.startswith("cuda:")
            else None
        )
        return {
            "model_id": self.model_id,
            "architecture": "DETRPose-N (COCO)",
            "backend": "PyTorch",
            "device": self.device,
            "gpu": self.device_info,
            "precision": str(next(self._model.model.parameters()).dtype).removeprefix("torch."),
            "packages": self.versions,
            "upstream": self.repository,
            "hf_revision": HF_REVISION,
            "weights": [self.checkpoint],
            "license": "Apache-2.0",
            "input_size": [640, 640],
            "batch_size": 1,
            "confidence_threshold": PERSON_THRESHOLD,
            "bbox": "unavailable: null",
            "joint_confidence": "unavailable: null",
            "visibility": "unavailable: null",
            "gpu_peak_memory_bytes": peak,
            "memory_scope": "current process PyTorch allocator since adapter initialization",
            "warmup": "none; all input frames retained; not comparative benchmark",
        }

    def close(self) -> None:
        self._model = None
        self.closed = True
        gc.collect()
