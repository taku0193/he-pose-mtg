"""Common command interface. Model adapters are added on their dedicated branches."""

import argparse
import json
import sys
from pathlib import Path

from hepose import __version__
from hepose.doctor import inspect
from hepose.schema import MODEL_IDS


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="HE姿勢推定デモ：共通CLI")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="インストールや推論を行わず環境を確認")
    doctor.add_argument("--role", choices=("client", "worker"), default="worker")
    doctor.add_argument("--model", choices=(*MODEL_IDS, "all"), default="all")
    doctor.add_argument("--device", default="cuda:0")
    doctor.add_argument("--json", action="store_true")
    run = commands.add_parser("run", help="録画動画の推論（モデル対応は後続工程）")
    run.add_argument("--model", choices=MODEL_IDS, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    live = commands.add_parser("live", help="Macからの遠隔ライブ（未実装）")
    live.add_argument("--model", choices=MODEL_IDS, required=True)
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--remote", required=True)
    live.add_argument("--remote-root", required=True)
    live.add_argument("--device", default="cuda:0")
    live.add_argument("--show-fps", action="store_true")
    benchmark = commands.add_parser("benchmark", help="モデル別の順次計測（未実装）")
    benchmark.add_argument("--models", required=True)
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda:0")
    compare = commands.add_parser("compare", help="元フレームで揃えた比較動画（未実装）")
    compare.add_argument("--input-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "doctor":
        result = inspect(args.role, args.model, args.device)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        else:
            print(f"doctor / {args.role} / ready={result['ready']}")
            for name, item in result["checks"].items():
                print(f"  {name}: {item['status']} {item.get('detail', '')}")
            for name, item in result.get("models", {}).items():
                print(f"  {name}: {item['status']}")
        return 0 if result["ready"] else 1
    print(f"{args.command}: この機能またはモデルアダプタは未実装です。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
