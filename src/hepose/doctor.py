"""Read-only checks: never install, download, open a camera, or start inference."""

import importlib.util
import os
import platform
import shutil
import subprocess
import sys

from hepose.schema import MODEL_IDS


def module_check(name: str) -> dict:
    found = importlib.util.find_spec(name) is not None
    return {"status": "ok" if found else "missing", "detail": name}


def inspect(role: str, model: str = "all", device: str = "cuda:0") -> dict:
    checks = {"python": {"status": "ok", "detail": sys.version.split()[0]}}
    checks["opencv"] = module_check("cv2")
    result = {"role": role, "checks": checks, "camera_opened": False}
    if role == "client":
        checks["ssh"] = {"status": "ok" if shutil.which("ssh") else "missing"}
        display = platform.system() != "Linux" or bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        checks["display"] = {
            "status": "available_unverified" if display else "missing",
            "detail": "表示・カメラの実動作は未検証",
        }
    else:
        selected = MODEL_IDS if model == "all" else (model,)
        result["device"] = device
        result["models"] = {
            name: {"status": "not_implemented", "detail": "モデルアダプタは未実装"}
            for name in selected
        }
        checks["torch"] = module_check("torch")
        if device.startswith("cuda:"):
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                checks["cuda"] = {
                    "status": "visible_unverified" if proc.returncode == 0 else "unavailable",
                    "detail": (proc.stdout or proc.stderr).strip(),
                }
            except (OSError, subprocess.TimeoutExpired) as exc:
                checks["cuda"] = {"status": "unavailable", "detail": str(exc)}
    result["ready"] = role == "client" and all(
        item["status"] in {"ok", "available_unverified"} for item in checks.values()
    )
    return result
