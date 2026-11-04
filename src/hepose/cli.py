"""Common command interface. Model adapters are added on their dedicated branches."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from hepose import __version__
from hepose.adapters import load_adapter
from hepose.config import load_config, validate_device
from hepose.doctor import inspect, inspect_environments
from hepose.schema import MODEL_IDS


def device_argument(value: str) -> str:
    try:
        return validate_device(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def record_worker_failure(args, settings, device, error):
    from hepose.results import RunWriter

    if args.output.exists():
        return
    environment = {
        "model_id": args.model,
        "requested_device": device,
        "device": None,
        "input": {"path": str(args.input.resolve()), "sha256": None},
        "configuration": settings,
        "failure_origin": "controller",
    }
    with RunWriter(args.output, uuid4().hex, environment) as results:
        results.record_failure(error)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="HE姿勢推定デモ：共通CLI")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="インストールや推論を行わず環境を確認")
    doctor.add_argument("--role", choices=("client", "worker"), default="worker")
    doctor.add_argument("--model", choices=(*MODEL_IDS, "all"), default="all")
    doctor.add_argument("--device", type=device_argument)
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    run = commands.add_parser("run", help="録画動画の推論（モデル対応は後続工程）")
    run.add_argument("--model", choices=MODEL_IDS, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", type=device_argument)
    run.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    live = commands.add_parser("live", help="Macからの遠隔ライブ（未実装）")
    live.add_argument("--model", choices=MODEL_IDS, required=True)
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--remote", required=True)
    live.add_argument("--remote-root", required=True)
    live.add_argument("--device", type=device_argument, default="cuda:0")
    live.add_argument("--show-fps", action="store_true")
    benchmark = commands.add_parser("benchmark", help="モデル別の順次計測（未実装）")
    benchmark.add_argument("--models", required=True)
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", type=device_argument, default="cuda:0")
    compare = commands.add_parser("compare", help="元フレームで揃えた比較動画（未実装）")
    compare.add_argument("--input-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    for command in (doctor, run, live, benchmark, compare):
        command.add_argument("--config", type=Path, help="Git管理外のローカルTOML設定")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except KeyboardInterrupt:
        print("中断しました。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 1


def execute(args) -> int:
    configuration = load_config(args.config)
    explicit_device = getattr(args, "device", None)
    if explicit_device:
        validate_device(explicit_device)
    if args.command == "doctor":
        if args.role == "client" or args.local:
            result = inspect(
                args.role, args.model, args.device or "cuda:0", configuration.model(args.model)
            )
        else:
            result = inspect_environments(args.model, args.device, configuration)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        else:
            print(f"doctor / {args.role} / ready={result['ready']}")
            for name, item in result["checks"].items():
                print(f"  {name}: {item['status']} {item.get('detail', '')}")
            for name, item in result.get("models", {}).items():
                print(f"  {name}: {item['status']}")
        return 0 if result["ready"] else 1
    if args.command == "run":
        from hepose.pipeline import run_video

        settings = configuration.model(args.model)
        device = args.device or settings.get("device", "cuda:0")
        python = settings.get("python", sys.executable)
        if not args.local and os.path.abspath(python) != os.path.abspath(sys.executable):
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(args.output)
            command = [
                python,
                "-m",
                "hepose.cli",
                "run",
                "--local",
                "--model",
                args.model,
                "--input",
                str(args.input.resolve()),
                "--output",
                str(args.output.resolve()),
                "--device",
                device,
            ]
            if configuration.path:
                command.extend(["--config", str(configuration.path)])
            try:
                code = subprocess.run(command, check=False).returncode
            except OSError as exc:
                record_worker_failure(args, settings, device, exc)
                raise
            code = 128 - code if code < 0 else code
            if code:
                error = (
                    KeyboardInterrupt("worker interrupted")
                    if code == 130
                    else RuntimeError(f"worker exited with code {code}")
                )
                record_worker_failure(args, settings, device, error)
            return code
        summary = run_video(
            args.input,
            args.output,
            lambda: load_adapter(args.model, device, settings),
            model_id=args.model,
            device=device,
            configuration=settings,
        )
        return 0 if summary["status"] == "success" else 1
    print(f"{args.command}: この機能またはモデルアダプタは未実装です。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
