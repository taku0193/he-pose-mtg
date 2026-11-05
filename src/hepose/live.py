"""Run camera capture, SSH GPU inference, and matching frame display."""

import csv
import re
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import perf_counter_ns
from uuid import uuid4

from hepose.config import validate_device
from hepose.environment import collect_environment
from hepose.live_capture import Capture
from hepose.live_capture import LatestFrame as LatestFrame
from hepose.live_protocol import WIRE, ProtocolError, check_size, validate_reply
from hepose.live_ssh import SshSession
from hepose.render import draw_poses
from hepose.results import write_json
from hepose.schema import FrameResult, finite, integer
from hepose.timing import summarize, throughput
from hepose.video import VideoError, VideoWriter, backend

WINDOW = "HE Pose | YOLO26 (q: quit)"
METRICS = (
    "encode_ms",
    "round_trip_ms",
    "worker_decode_ms",
    "worker_adapter_ms",
    "worker_inference_ms",
    "render_ms",
    "capture_to_render_ms",
    "display_latency_ms",
)


@dataclass(frozen=True)
class LiveOptions:
    remote: str
    remote_root: str
    remote_python: str
    remote_config: str
    device: str | None
    camera: int
    video: Path | None
    width: int
    height: int
    fps: float
    jpeg_quality: int
    connect_timeout: int
    startup_timeout: int
    io_timeout: int
    session_timeout: int
    output: Path
    record: bool
    show_fps: bool
    headless: bool
    limit: int | None
    network_note: str | None


def live_options(args, config):
    def value(name, default=None):
        explicit = getattr(args, name, None)
        return explicit if explicit is not None else config.get(name, default)

    remote, root = value("remote"), value("remote_root")
    if not isinstance(remote, str) or not re.fullmatch(
        r"[\w][\w.%-]*(?:@[\w][\w.:%-]*)?", remote, re.ASCII
    ):
        raise ValueError("有効なSSH接続先を --remote または [live].remote で指定してください。")
    if not isinstance(root, str) or not root.startswith("/") or "\x00" in root:
        raise ValueError("--remote-root または [live].remote_root はLinux上の絶対パスが必要です。")
    python = value("remote_python", ".venv-yolo26/bin/python")
    worker_config = value("remote_config", "hepose.local.toml")
    for name, path in (("remote_python", python), ("remote_config", worker_config)):
        if not isinstance(path, str) or not path or path.startswith("-") or "\x00" in path:
            raise ValueError(f"invalid live {name}")
    device = value("device")
    if device is not None and not validate_device(device).startswith("cuda:"):
        raise ValueError("liveは割当済みCUDA GPUが必要です。")
    camera, width, height = value("camera", 0), value("width", 640), value("height", 480)
    integer(camera, "camera")
    check_size(width, height)
    fps, quality = value("fps", 25), value("jpeg_quality", 80)
    finite(fps, "fps", 1)
    integer(quality, "jpeg_quality", 1)
    if fps > 120 or quality > 100:
        raise ValueError("fps must be 1–120 and jpeg_quality must be 1–100")
    connect, startup = value("connect_timeout", 5), value("startup_timeout", 45)
    io, session = value("io_timeout", 15), value("session_timeout", 1800)
    for label, number, maximum in (
        ("connect_timeout", connect, 60),
        ("startup_timeout", startup, 300),
        ("io_timeout", io, 120),
        ("session_timeout", session, 7200),
    ):
        integer(number, label, 1)
        if number > maximum:
            raise ValueError(f"{label} must be <= {maximum}")
    if not connect <= startup < session or io >= session:
        raise ValueError(
            "startup_timeout must cover connect_timeout; session must exceed startup and IO"
        )
    limit = value("limit")
    if limit is not None:
        integer(limit, "limit", 1)
    record, show_fps, headless = (value(name, False) for name in ("record", "show_fps", "headless"))
    if any(type(flag) is not bool for flag in (record, show_fps, headless)):
        raise ValueError("record, show_fps and headless must be booleans")
    video = value("video")
    if headless and not video:
        raise ValueError("--headless requires --video")
    if getattr(args, "camera", None) is not None and video:
        raise ValueError("--camera and --video are mutually exclusive")
    note = value("network_note")
    if note is not None and not isinstance(note, str):
        raise ValueError("network_note must be a string")
    output = value("output", Path("outputs") / f"live-{uuid4().hex[:12]}")
    return LiveOptions(
        remote,
        root,
        python,
        worker_config,
        device,
        camera,
        Path(video) if video else None,
        width,
        height,
        fps,
        quality,
        connect,
        startup,
        io,
        session,
        Path(output),
        record,
        show_fps,
        headless,
        limit,
        note,
    )


class Exchange:
    def __init__(self, session, frame, options):
        self.frame = frame
        self.sent = Event()
        self.result = Future()
        self.thread = Thread(
            target=self._exchange, args=(session, options), name="hepose-request", daemon=True
        )
        self.thread.start()

    def _exchange(self, session, options):
        try:
            cv2 = backend()
            height, width = self.frame.image.shape[:2]
            start = perf_counter_ns()
            ok, data = cv2.imencode(
                ".jpg", self.frame.image, [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality]
            )
            if not ok:
                raise VideoError("JPEG圧縮に失敗しました。")
            jpeg = data.tobytes()
            encoded = perf_counter_ns()
            session.channel.send(
                {
                    "kind": "frame",
                    "wire": WIRE,
                    "session": session.session,
                    "frame": self.frame.index,
                    "width": width,
                    "height": height,
                },
                jpeg,
                timeout=options.io_timeout,
            )
            self.sent.set()
            remaining = options.io_timeout - (perf_counter_ns() - encoded) / 1e9
            reply, payload = session.channel.receive(timeout=remaining)
            received = perf_counter_ns()
            if payload:
                raise ProtocolError("unexpected binary data in result")
            people = validate_reply(reply, session.session, self.frame.index, width, height)
            self.result.set_result(
                (
                    people,
                    {
                        "encode_ms": (encoded - start) / 1e6,
                        "round_trip_ms": (received - encoded) / 1e6,
                        "worker_decode_ms": reply["decode_ms"],
                        "worker_adapter_ms": reply["adapter_ms"],
                        "worker_inference_ms": reply["inference_ms"],
                        "measurement_method": reply["measurement_method"],
                    },
                )
            )
        except EOFError:
            self.result.set_exception(ProtocolError("SSH stream closed before the frame result"))
        except Exception as exc:
            self.result.set_exception(exc)


def run_live(options):
    output = options.output
    output.mkdir(parents=True, exist_ok=False)
    with (
        (output / "run.log").open("a", encoding="utf-8", buffering=1) as log,
        (output / "timings.csv").open("x", newline="", encoding="utf-8") as stream,
    ):
        return _run(options, log, stream)


def _run(options, log, stream):
    run_id = uuid4().hex
    capture = Capture(options)
    settings = asdict(options)
    for key in ("video", "output"):
        settings[key] = str(settings[key]) if settings[key] is not None else None
    environment = {
        "wire": WIRE,
        "client": None,
        "worker": None,
        "configuration": settings,
        "transport": "OpenSSH direct-tcpip to per-session Linux loopback listener",
        "started_at": datetime.now(UTC).isoformat(),
    }
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "frame_index",
            "capture_elapsed_ms",
            "people_count",
            "displayed",
            "recorded",
            *METRICS,
            "measurement_method",
        ),
    )
    writer.writeheader()
    samples = {metric: [] for metric in METRICS}
    counts = dict(dispatched=0, sent=0, received=0, rendered=0, displayed=0, recorded=0)
    ssh = flight = recorder = cv2 = None
    capture_started = window_open = False
    active_ns = elapsed_ms = None
    error = None
    status, reason = "success", "end_of_input"

    def failure(exc):
        nonlocal error, status, reason
        log.write(f"{type(exc).__name__}: {exc}\n")
        if error is None:
            error = f"{type(exc).__name__}: {exc}"
            status, reason = (
                ("interrupted", "ctrl_c")
                if isinstance(exc, KeyboardInterrupt)
                else ("failed", "error")
            )

    def cleanup(operation):
        try:
            return operation()
        except (Exception, KeyboardInterrupt) as exc:
            failure(exc)
            return None

    try:
        if options.video and not options.video.is_file():
            raise FileNotFoundError(options.video)
        cv2 = backend()
        if not options.headless:
            gui = getattr(cv2, "currentUIFramework", lambda: None)()
            if gui == "":
                raise VideoError("GUI版OpenCVが必要です。Macでは client 依存を使用してください。")
        environment["client"] = collect_environment(options.video, "yolo26", options.device)
        ssh = SshSession(options, run_id, environment["client"]["source"], log)
        log.write(f"Connecting to {options.remote} (session {run_id})\n")
        ready, environment["startup_ms"] = ssh.open()
        environment["worker"] = {k: ready[k] for k in ("environment", "model", "model_load_ms")}
        log.write("Worker ready; starting capture\n")
        if not options.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            window_open = True
        active_ns = perf_counter_ns()
        capture.thread.start()
        capture_started = True
        while True:
            if window_open:
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    reason = "q"
                    break
            elif flight is not None and not flight.result.done():
                capture.stop.wait(0.002)
            if capture.error:
                raise capture.error
            if flight is not None and flight.result.done():
                counts["sent"] += int(flight.sent.is_set())
                frame, result = flight.frame, flight.result
                flight = None
                people, row = result.result()
                counts["received"] += 1
                height, width = frame.image.shape[:2]
                age = (frame.at_ns - active_ns) / 1e6
                rendered_at = perf_counter_ns()
                image = draw_poses(
                    frame.image,
                    FrameResult(
                        run_id,
                        "yolo26",
                        frame.index,
                        age,
                        width,
                        height,
                        people,
                        source_time_basis="client capture clock since session capture start",
                        reference_frame_indices=(frame.index,),
                    ),
                )
                if window_open and options.show_fps:
                    fps = throughput(counts["displayed"], (rendered_at - active_ns) / 1e6)
                    cv2.putText(
                        image,
                        f"Display: {fps or 0:.1f} fps",
                        (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                before_display = perf_counter_ns()
                row.update(
                    render_ms=(before_display - rendered_at) / 1e6,
                    capture_to_render_ms=(before_display - frame.at_ns) / 1e6,
                    display_latency_ms=None,
                )
                counts["rendered"] += 1
                displayed = recorded = False
                if window_open:
                    enqueue_ns = perf_counter_ns()
                    cv2.imshow(WINDOW, image)
                    row["display_latency_ms"] = (enqueue_ns - frame.at_ns) / 1e6
                    counts["displayed"] += 1
                    displayed = True
                try:
                    if options.record:
                        if recorder is None:
                            recorder = VideoWriter(
                                options.output / "annotated.mp4", options.fps, (width, height)
                            )
                            recorder.__enter__()
                        recorder.write(image)
                        counts["recorded"] += 1
                        recorded = True
                finally:
                    writer.writerow(
                        {
                            "frame_index": frame.index,
                            "capture_elapsed_ms": age,
                            "people_count": len(people),
                            "displayed": int(displayed),
                            "recorded": int(recorded),
                            **row,
                        }
                    )
                    stream.flush()
                    for metric in METRICS:
                        if row[metric] is not None:
                            samples[metric].append(row[metric])
            ssh.check()
            if flight is None:
                frame = capture.slot.take()
                if frame is not None:
                    counts["dispatched"] += 1
                    flight = Exchange(ssh, frame, options)
                elif capture.done.is_set():
                    # The producer can finish between take() and done.is_set().
                    if capture.slot.counts()[2]:
                        continue
                    if capture.error:
                        raise capture.error
                    if not counts["received"]:
                        raise VideoError("推論結果がありません。")
                    break
                elif not window_open:
                    capture.done.wait(0.002)
            if (perf_counter_ns() - active_ns) / 1e9 >= options.session_timeout:
                raise TimeoutError("live session deadline exceeded")
    except (Exception, KeyboardInterrupt) as exc:
        failure(exc)
    finally:
        if active_ns is not None:
            elapsed_ms = (perf_counter_ns() - active_ns) / 1e6
        capture.stop.set()
        if window_open:
            cleanup(lambda: cv2.destroyWindow(WINDOW))
        if ssh:
            worker_code = cleanup(lambda: ssh.close(in_flight=flight is not None))
            environment["worker_ssh_exit_code"] = worker_code
            if status == "success" and reason != "q" and worker_code != 0:
                failure(ProtocolError("worker did not stop cleanly; see run.log"))
        if flight:
            cleanup(lambda: flight.thread.join(timeout=1))
            counts["sent"] += int(flight.sent.is_set())
            if flight.thread.is_alive():
                failure(TimeoutError("request thread did not stop within deadline"))
        if ssh:
            cleanup(ssh.close_pipes)
        if capture_started:
            cleanup(lambda: capture.thread.join(timeout=2))
            if capture.thread.is_alive():
                failure(TimeoutError("capture backend did not release within deadline"))
        if recorder and not recorder.closed:
            cleanup(lambda: recorder.__exit__(None, None, None))
    acquired, replaced, pending = capture.slot.counts()
    counts.update(
        acquired=acquired,
        replaced=replaced,
        pending_at_stop=pending,
        dispatched_not_sent=counts["dispatched"] - counts["sent"],
        sent_without_result=counts["sent"] - counts["received"],
        sent_without_display=counts["sent"] - counts["displayed"] if window_open else None,
    )
    environment["capture"] = capture.metadata()
    summary = {
        "status": status,
        "stop_reason": reason,
        "error": error,
        "run_id": run_id,
        "source": "replay" if options.video else "camera",
        "counts": counts,
        "active_ms": elapsed_ms,
        "result_fps": throughput(counts["received"], elapsed_ms) if elapsed_ms else None,
        "display_fps": throughput(counts["displayed"], elapsed_ms)
        if window_open and elapsed_ms
        else None,
        "timing": {metric: summarize(values) for metric, values in samples.items()},
    }
    log.write(f"Session ended: {status} ({reason}), counts={counts}\n")
    write_json(options.output / "environment.json", environment)
    write_json(options.output / "summary.json", summary)
    return summary
