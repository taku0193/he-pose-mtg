"""Fetch an attributed CC BY-SA recording and prepare a CFR validation clip."""

import hashlib
import json
import subprocess

from prepare_yolo26 import ROOT, fetch

SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/5/57/Jumping_jacks_and_burpees.webm"
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm"
SOURCE_SHA1 = "de0408ee2f90afba376f23d86f48d6c98d4eb42b"
CREDIT = "Jumping jacks and burpees — Taco fleur — CC BY-SA 4.0"


def main():
    original = ROOT / "work/validation-video/jumping-jacks.webm"
    source = fetch(SOURCE_URL, original)
    if hashlib.sha1(original.read_bytes()).hexdigest() != SOURCE_SHA1:
        raise ValueError("source recording differs from the Commons file description")
    target = ROOT / "assets/sample/jumping-jacks-validation.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(original),
        "-t",
        "15",
        "-an",
        "-vf",
        "fps=25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        f"comment={CREDIT}; first 15 seconds, CFR 25, no audio",
        "-n",
        str(target),
    ]
    subprocess.run(command, check=True)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(target),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    info = {
        "title": "Jumping jacks and burpees",
        "author": "Taco fleur",
        "source_page": SOURCE_PAGE,
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "source": {**source, "commons_sha1": SOURCE_SHA1},
        "derived": {
            "path": str(target.relative_to(ROOT)),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "changes": "first 15 seconds; CFR 25 fps; H.264 MP4; audio removed",
            "probe": json.loads(probe.stdout),
        },
        "attribution": CREDIT,
        "pose_output_changes": "pose skeleton and frame labels added",
        "derived_video_license": "CC BY-SA 4.0",
        "purpose": "recorded motion validation; not the final three MTG sample set or a benchmark",
    }
    manifest = ROOT / "docs/validation-assets/jumping-jacks.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    (target.parent / "jumping-jacks-ATTRIBUTION.md").write_text(
        f"# {CREDIT}\n\nSource: {SOURCE_PAGE}\n\n"
        "License: https://creativecommons.org/licenses/by-sa/4.0/\n\n"
        "Changes: first 15 seconds; 25 fps CFR; H.264 MP4; audio removed. "
        "Pose output additionally overlays inferred skeletons and frame labels.\n"
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
