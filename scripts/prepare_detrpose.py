"""Explicit DETRPose preparation: pinned repository, declared patch, runtime and weights."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare_yolo26 import fetch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-environment", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from hepose.detrpose import (
        CONFIG_SHA256,
        CONFIG_URL,
        PATCH_SHA256,
        UPSTREAM_COMMIT,
        WEIGHTS_SHA256,
        WEIGHTS_SIZE,
        WEIGHTS_URL,
        verify_repository,
    )

    repo = ROOT / "third_party/DETRPose"
    patch = ROOT / "patches/detrpose/local-safetensors.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != PATCH_SHA256:
        raise ValueError("記録したパッチのSHA256が一致しません。")
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/SebastianJanampa/DETRPose.git",
                str(repo),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "switch", "--detach", UPSTREAM_COMMIT], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("既存リポジトリの版が異なります。上書きしません。")
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "--no-ext-diff", "HEAD", "--"])
    if not diff:
        subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(repo), "apply", str(patch)], check=True)
    source = verify_repository(repo)
    if not args.skip_environment:
        env = dict(
            os.environ,
            UV_CACHE_DIR=str(ROOT / "work/uv-cache"),
            UV_PROJECT_ENVIRONMENT=str(ROOT / ".venv-detrpose"),
        )
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(ROOT / "environments/detrpose"),
                "--locked",
                "--python",
                "3.11",
                "--no-python-downloads",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    weights = fetch(WEIGHTS_URL, ROOT / "models/detrpose/model.safetensors", WEIGHTS_SHA256)
    if weights["size"] != WEIGHTS_SIZE:
        raise ValueError("公式重みのサイズが一致しません。")
    config = fetch(CONFIG_URL, ROOT / "models/detrpose/config.json", CONFIG_SHA256)
    manifest = {
        "upstream": source,
        "weights": weights,
        "hf_config": config,
        "patch_reason": "load official local safetensors without network; no numerical changes",
    }
    (ROOT / "models/detrpose/preparation.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
