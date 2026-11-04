"""Shared runtime checks; importing this module never imports a model or PyTorch."""

import hashlib
from pathlib import Path

from hepose.config import validate_device


def _cuda_runtime_uuid(runtime, index: int) -> str | None:
    """Read the process-visible CUDA device UUID on older PyTorch, without guessing indices."""
    import ctypes
    from uuid import UUID

    module_path = getattr(runtime, "__file__", None)
    if not module_path:
        return None
    libraries = sorted((Path(module_path).parent / "lib").glob("libcudart*.so*"))
    if not libraries:
        raise RuntimeError("CUDA UUID確認用のランタイムが見つかりません。")

    class CudaUUID(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    library = ctypes.CDLL(str(libraries[0]))
    bus_id = ctypes.create_string_buffer(32)
    pci_function = library.cudaDeviceGetPCIBusId
    pci_function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    pci_function.restype = ctypes.c_int
    if pci_function(bus_id, len(bus_id), index) != 0:
        raise RuntimeError("CUDAデバイスのPCI IDを確認できません。")
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    if driver.cuInit(0) != 0:
        raise RuntimeError("CUDAドライバを初期化できません。")
    physical = ctypes.c_int()
    driver.cuDeviceGetByPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]
    if driver.cuDeviceGetByPCIBusId(ctypes.byref(physical), bus_id) != 0:
        raise RuntimeError("CUDAのPCI IDに対応するデバイスを確認できません。")
    function = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
    function.argtypes = [ctypes.POINTER(CudaUUID), ctypes.c_int]
    function.restype = ctypes.c_int
    value = CudaUUID()
    if function(ctypes.byref(value), physical) != 0:
        raise RuntimeError("CUDAデバイスのUUIDを確認できません。")
    return "GPU-" + str(UUID(bytes=bytes(value.bytes)))


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
    if uuid is None:
        uuid = _cuda_runtime_uuid(runtime, index)
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
