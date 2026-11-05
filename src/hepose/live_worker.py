"""One authenticated live session on an ephemeral Linux loopback port."""

import argparse
import os
import secrets
import signal
import socket
import sys
from dataclasses import asdict
from pathlib import Path
from time import monotonic, perf_counter_ns

from hepose import __version__
from hepose.adapters import InputFrame, load_adapter
from hepose.config import load_config, validate_device
from hepose.environment import collect_environment
from hepose.live_protocol import WIRE, Channel, ProtocolError, decode_image, identity
from hepose.schema import integer


def serve_session(
    channel, token, source, adapter_factory, *, timeout=10, environment=None, frame_watchdog=None
):
    hello, data = channel.receive(timeout=timeout)
    session = hello.get("session")
    presented = hello.get("token")
    identity(hello, "hello", session)
    if (
        data
        or not isinstance(session, str)
        or not 0 < len(session) <= 64
        or not isinstance(presented, str)
        or not presented.isascii()
        or not secrets.compare_digest(presented, token)
    ):
        raise ProtocolError("invalid live authentication")
    if hello.get("app") != __version__ or hello.get("revision") != source.get("git_commit"):
        raise ProtocolError("live client and worker must use the same version and commit")

    started = perf_counter_ns()
    adapter = adapter_factory()
    try:
        channel.send(
            {
                "kind": "ready",
                "wire": WIRE,
                "session": session,
                "app": __version__,
                "source": source,
                "model": adapter.environment(),
                "environment": environment,
                "model_load_ms": (perf_counter_ns() - started) / 1e6,
            },
            timeout=timeout,
        )
        previous = -1
        while True:
            if frame_watchdog is not None:
                frame_watchdog()
            try:
                message, data = channel.receive(timeout=timeout)
            except EOFError:
                return
            if frame_watchdog is not None:
                frame_watchdog()
            identity(message, "frame", session)
            frame = message.get("frame")
            try:
                integer(frame, "frame")
            except ValueError as exc:
                raise ProtocolError("invalid live frame index") from exc
            if frame <= previous:
                raise ProtocolError("live frame index is not increasing")
            previous = frame
            width, height = message.get("width"), message.get("height")
            started = perf_counter_ns()
            image = decode_image(data, width, height)
            decoded = perf_counter_ns()
            result = adapter.predict([InputFrame(frame, 0, width, height, image)], frame)
            adapter_ms = (perf_counter_ns() - decoded) / 1e6
            channel.send(
                {
                    "kind": "result",
                    "wire": WIRE,
                    "session": session,
                    "frame": frame,
                    "width": width,
                    "height": height,
                    "people": [
                        {"bbox": person.bbox, "keypoints": [asdict(p) for p in person.keypoints]}
                        for person in result.people
                    ],
                    "decode_ms": (decoded - started) / 1e6,
                    "adapter_ms": adapter_ms,
                    "inference_ms": result.inference_ms,
                    "measurement_method": result.measurement_method,
                },
                timeout=timeout,
            )
    finally:
        adapter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=validate_device)
    parser.add_argument("--startup-timeout", type=int, required=True)
    parser.add_argument("--io-timeout", type=int, required=True)
    parser.add_argument("--session-timeout", type=int, required=True)
    args = parser.parse_args(argv)
    if sys.platform != "linux" or not (
        1 <= args.io_timeout <= 120
        and 1 <= args.startup_timeout <= 300
        and max(args.io_timeout, args.startup_timeout) < args.session_timeout <= 7200
    ):
        parser.error("live worker requires Linux and finite session/IO timeouts")
    # The OS ends this dedicated process even if a driver or native call stalls.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    deadline = monotonic() + args.session_timeout

    def watchdog(seconds):
        signal.setitimer(signal.ITIMER_REAL, max(0.001, min(seconds, deadline - monotonic())))

    watchdog(args.startup_timeout)
    try:
        settings = load_config(args.config).model("yolo26")
        device = args.device or settings.get("device", "cuda:0")
        if not device.startswith("cuda:") or not settings.get("gpu_uuid"):
            raise ValueError("live worker requires an assigned CUDA device and gpu_uuid")
        environment = collect_environment(None, "yolo26", device)
        token = secrets.token_hex(32)
        print(f"live worker PID: {os.getpid()}", file=sys.stderr, flush=True)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(args.io_timeout)
            Channel(0, 1).send(
                {"kind": "listen", "wire": WIRE, "port": listener.getsockname()[1], "token": token},
                timeout=args.io_timeout,
            )
            # Only the bootstrap packet belongs on stdout.
            os.dup2(2, 1)
            with listener.accept()[0] as connection:
                listener.close()
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                serve_session(
                    Channel(connection.fileno(), connection.fileno()),
                    token,
                    environment["source"],
                    lambda: load_adapter("yolo26", device, settings),
                    timeout=args.io_timeout,
                    environment=environment,
                    frame_watchdog=lambda: watchdog(args.io_timeout + 1),
                )
        return 0
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        print(f"live worker: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
