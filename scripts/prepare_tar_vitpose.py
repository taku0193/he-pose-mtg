"""Explicit preparation of official TAR Base, ViTPose and YOLO11x weights and runtime."""

import argparse
import hashlib
import http.cookiejar
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from prepare_yolo26 import fetch

ROOT = Path(__file__).resolve().parents[1]


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
        if tag == "input" and attrs.get("type") == "hidden":
            self.values[attrs.get("name")] = attrs.get("value", "")


def fetch_drive(record: dict, target: Path) -> dict:
    from hepose.runtime import verify_asset

    target.parent.mkdir(parents=True, exist_ok=True)
    file_id = urllib.parse.urlsplit(record["url"]).path.split("/")[3]
    if not target.exists():
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
            {"export": "download", "id": file_id}
        )
        response = opener.open(url, timeout=60)
        if "text/html" in response.headers.get("Content-Type", ""):
            form = DownloadForm()
            form.feed(response.read().decode())
            response.close()
            if (
                form.action != "https://drive.usercontent.google.com/download"
                or form.values.get("id") != file_id
            ):
                raise ValueError("公式Driveのダウンロード応答を確認できません。")
            response = opener.open(
                form.action + "?" + urllib.parse.urlencode(form.values), timeout=60
            )
        if "text/html" in response.headers.get("Content-Type", ""):
            response.close()
            raise ValueError("Driveから重みのバイナリを取得できません。")
        digest, count = hashlib.sha256(), 0
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                while chunk := response.read(4 * 1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if count > record["size"]:
                        raise ValueError("公式重みのサイズが一致しません。")
                stream.flush()
                if count != record["size"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError("公式重みのサイズまたはSHA256が一致しません。")
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
                response.close()
    result = verify_asset(target, record["sha256"], record["url"])
    if result["size"] != record["size"]:
        raise ValueError("公式重みのサイズが一致しません。")
    target.with_name(target.name + ".source.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-environment", action="store_true")
    parser.add_argument("--write-config", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from hepose.tar_vitpose import ARTIFACTS, UPSTREAM_COMMIT, effective_config, verify_repository

    repository = ROOT / "third_party/TARViTPose"
    weights_dir = ROOT / "models/tar-vitpose"
    if args.write_config:
        config = effective_config(repository, weights_dir / "pretrain_model_b.pth")
        path = weights_dir / "inference.yaml"
        text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        if path.exists() and path.read_text() != text:
            raise FileExistsError("既存の推論設定が異なります。上書きしません。")
        path.write_text(text)
        return
    if not repository.exists():
        repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/zgspose/TARViTPose.git",
                str(repository),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "switch", "--detach", UPSTREAM_COMMIT], check=True
        )
    source = verify_repository(repository)
    if not args.skip_environment:
        env = dict(
            os.environ,
            UV_CACHE_DIR=str(ROOT / "work/uv-cache"),
            UV_PROJECT_ENVIRONMENT=str(ROOT / ".venv-tar"),
        )
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(ROOT / "environments/tar-vitpose"),
                "--locked",
                "--python",
                "3.11",
                "--no-python-downloads",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    records = {}
    for key, record in ARTIFACTS.items():
        target = weights_dir / record["filename"]
        records[key] = (
            fetch_drive(record, target)
            if "drive.google.com" in record["url"]
            else fetch(record["url"], target, record["sha256"])
        )
    subprocess.run(
        [str(ROOT / ".venv-tar/bin/python"), str(Path(__file__).resolve()), "--write-config"],
        cwd=ROOT,
        check=True,
    )
    manifest = {
        "upstream": source,
        "artifacts": records,
        "upstream_patches": [],
        "effective_config": str(weights_dir / "inference.yaml"),
    }
    (weights_dir / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
