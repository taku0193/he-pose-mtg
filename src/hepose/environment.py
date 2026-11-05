"""Collect reproducibility metadata without importing or initializing model packages."""

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from hepose import __version__


def source_revision() -> dict:
    project = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args], cwd=project, capture_output=True, text=True, timeout=5, check=False
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    dirty = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(dirty) if dirty is not None else None,
    }


def collect_environment(input_path: Path | None, model_id: str, device: str | None) -> dict:
    versions = {}
    for name in ("numpy", "opencv-python-headless", "opencv-python", "torch", "ultralytics"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    digest = None
    if input_path is not None and input_path.is_file():
        with input_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "application_version": __version__,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "source": source_revision(),
        "input": {"path": str(input_path.resolve()), "sha256": digest} if input_path else None,
        "model_id": model_id,
        "requested_device": device,
        "device": None,
        "packages": versions,
    }
