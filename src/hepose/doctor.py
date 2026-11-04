"""Read-only checks: never install, download, open a camera, or start inference."""

import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from hepose import __version__
from hepose.config import Configuration
from hepose.schema import MODEL_IDS


def module_check(name: str) -> dict:
    found = importlib.util.find_spec(name) is not None
    return {"status": "ok" if found else "missing", "detail": name}


def gui_check() -> dict:
    try:
        cv2 = importlib.import_module("cv2")
        line = next(
            (
                line.strip()
                for line in cv2.getBuildInformation().splitlines()
                if line.strip().startswith("GUI:")
            ),
            "GUI: UNKNOWN",
        )
        available = line.split(":", 1)[1].strip() not in {"NONE", "UNKNOWN", ""}
        return {"status": "ok" if available else "missing", "detail": line}
    except (ImportError, OSError) as exc:
        return {"status": "missing", "detail": str(exc)}


def inspect(
    role: str, model: str = "all", device: str = "cuda:0", settings: dict | None = None
) -> dict:
    checks = {"python": {"status": "ok", "detail": sys.version.split()[0]}}
    checks["opencv"] = module_check("cv2")
    result = {
        "role": role,
        "checks": checks,
        "camera_opened": False,
        "application_version": __version__,
    }
    if role == "client":
        checks["opencv_gui"] = gui_check()
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
        if "yolo26" in selected:
            from hepose.yolo26 import inspect_yolo26

            result["models"]["yolo26"] = inspect_yolo26(settings or {}, device)
        if "detrpose" in selected:
            from hepose.detrpose import inspect_detrpose

            result["models"]["detrpose"] = inspect_detrpose(settings or {}, device)
        checks["torch"] = module_check("torch")
        if device.startswith("cuda:"):
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                checks["cuda"] = {
                    "status": "visible_unverified" if proc.returncode == 0 else "unavailable",
                    "detail": (proc.stdout or proc.stderr).strip(),
                }
            except (OSError, subprocess.TimeoutExpired) as exc:
                checks["cuda"] = {"status": "unavailable", "detail": str(exc)}
    if role == "client":
        result["ready"] = all(
            item["status"] in {"ok", "available_unverified"} for item in checks.values()
        )
    else:
        result["ready"] = all(item["status"] == "ready" for item in result["models"].values())
    return result


def inspect_environments(model: str, device: str | None, configuration: Configuration) -> dict:
    if not configuration.models:
        return inspect("worker", model, device or "cuda:0")
    selected = MODEL_IDS if model == "all" else (model,)
    result = {
        "role": "worker",
        "application_version": __version__,
        "camera_opened": False,
        "checks": {},
        "models": {},
        "ready": False,
    }
    for name in selected:
        settings = configuration.model(name)
        selected_device = device or settings.get("device", "cuda:0")
        python = settings.get("python", sys.executable)
        if os.path.abspath(python) == os.path.abspath(sys.executable):
            payload = inspect("worker", name, selected_device, settings)
        else:
            command = [
                python,
                "-m",
                "hepose.cli",
                "doctor",
                "--local",
                "--model",
                name,
                "--device",
                selected_device,
                "--json",
            ]
            if configuration.path:
                command.extend(["--config", str(configuration.path)])
            try:
                process = subprocess.run(
                    command, capture_output=True, text=True, timeout=30, check=False
                )
                payload = json.loads(process.stdout)
                if not isinstance(payload, dict):
                    raise ValueError("worker response must be an object")
                if payload.get("application_version") != __version__:
                    raise ValueError("worker application version differs")
                if not isinstance(payload.get("models"), dict):
                    raise ValueError("worker response has no model table")
                if not isinstance(payload["models"].get(name), dict):
                    raise ValueError("worker response has no requested model")
                if not isinstance(payload.get("checks"), dict):
                    raise ValueError("worker response has no environment checks")
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                result["models"][name] = {"status": "environment_error", "detail": str(exc)}
                continue
        item = dict(payload["models"][name])
        item.update({"python": python, "device": selected_device, "checks": payload["checks"]})
        weights = settings.get("weights")
        item["weights"] = {
            "path": weights,
            "status": "present" if weights and Path(weights).is_file() else "missing",
        }
        result["models"][name] = item
    result["ready"] = all(item["status"] == "ready" for item in result["models"].values())
    return result
