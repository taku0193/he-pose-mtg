"""Explicit preparation only: install the pinned environment and fetch official artifacts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_URL = (
    "https://raw.githubusercontent.com/ultralytics/ultralytics/"
    "ae725d5cb20c2d849b864cd34fc15f5c55f877a5/ultralytics/assets/bus.jpg"
)


def fetch(url: str, target: Path, expected_sha256: str | None = None) -> dict:
    sidecar = target.with_name(target.name + ".source.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        expected = expected_sha256 or previous.get("sha256")
        if not expected or (previous and previous.get("url") != url):
            raise FileExistsError(f"出典を確認できない既存ファイルです: {target}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"既存ファイルのSHA256が一致しません: {target}")
    else:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "he-pose-mtg/0.1"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                stream.flush()
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                if expected_sha256 and digest != expected_sha256:
                    raise ValueError(f"取得ファイルのSHA256が一致しません: {url}")
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    info = {
        "url": url,
        "sha256": digest,
        "size": target.stat().st_size,
        "path": str(target.relative_to(ROOT)),
    }
    sidecar.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    return info


def main():
    parser = argparse.ArgumentParser(description="YOLO26の明示的な準備工程")
    parser.add_argument("--skip-environment", action="store_true", help="取得・検証のみ行う")
    args = parser.parse_args()
    if not args.skip_environment:
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = str(ROOT / "work/uv-cache")
        env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv-yolo26")
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(ROOT / "environments/yolo26"),
                "--locked",
                "--python",
                "3.11",
                "--no-python-downloads",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    sys.path.insert(0, str(ROOT / "src"))
    from hepose.yolo26 import UPSTREAM_COMMIT, WEIGHTS_SHA256, WEIGHTS_SIZE, WEIGHTS_URL

    weights = fetch(WEIGHTS_URL, ROOT / "models/yolo26/yolo26n-pose.pt", WEIGHTS_SHA256)
    if weights["size"] != WEIGHTS_SIZE:
        raise ValueError("公式重みのサイズが一致しません。")
    image = fetch(SAMPLE_URL, ROOT / "work/yolo26-smoke/bus.jpg")
    manifest = {
        "upstream_commit": UPSTREAM_COMMIT,
        "weights": weights,
        "sample_image": image,
        "sample_purpose": "official image smoke; not a recorded motion sample",
    }
    target = ROOT / "models/yolo26/preparation.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
