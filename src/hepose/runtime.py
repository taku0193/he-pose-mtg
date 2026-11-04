"""Shared runtime checks; importing this module never imports a model or PyTorch."""

import hashlib
from pathlib import Path

from hepose.config import validate_device


def validate_runtime_device(device: str, runtime, expected_uuid: str | None = None) -> dict:
    validate_device(device)
    if device == "cpu":
        return {"device": "cpu", "gpu_uuid": None, "gpu_name": None, "memory_total_bytes": None}
    if not runtime.cuda.is_available():
        raise RuntimeError("指定したCUDAデバイスを利用できません。CPUへ自動で切り替えません。")
    index = int(device.split(":")[1])
    if index >= runtime.cuda.device_count():
        raise RuntimeError(f"指定したCUDAデバイスがありません: {device}")
    properties = runtime.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    uuid = str(uuid) if uuid is not None else None
    uuid = ("GPU-" + uuid.removeprefix("GPU-")) if uuid is not None else None
    if expected_uuid and (
        uuid is None
        or uuid.lower().removeprefix("gpu-") != expected_uuid.lower().removeprefix("gpu-")
    ):
        raise RuntimeError(f"割当GPUのUUIDが一致しません: expected={expected_uuid}, actual={uuid}")
    return {
        "device": device,
        "gpu_uuid": uuid,
        "gpu_name": properties.name,
        "memory_total_bytes": int(properties.total_memory),
    }


def verify_asset(path: Path, expected_sha256: str, source: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ローカルファイルがありません: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"SHA256が一致しません: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size": path.stat().st_size,
        "source": source,
    }
