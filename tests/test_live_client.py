from hepose.live import live_options
from hepose.live_capture import LatestFrame


def test_latest_slot_discards_only_unsent_frames():
    slot = LatestFrame()
    slot.put("first")
    assert slot.take() == "first"
    slot.put("second")
    slot.put("third")
    assert slot.take() == "third"
    assert slot.take() is None
    assert slot.dropped == 1


def test_live_explicit_values_override_config_and_defaults():
    from hepose.cli import parser

    args = parser().parse_args(
        ["live", "--model", "yolo26", "--fps", "20", "--remote", "lab", "--remote-root", "/project"]
    )
    options = live_options(args, {"fps": 12, "width": 800, "remote": "other"})
    assert options.fps == 20
    assert options.width == 800
    assert options.remote == "lab"


def test_camera_quit_releases_owned_resources_and_saves_only_stats(monkeypatch, tmp_path):
    import csv
    from threading import Event

    import cv2
    import numpy as np

    from hepose import live, live_capture
    from hepose.cli import parser

    events = []
    shown = Event()
    image = np.full((24, 32, 3), 75, dtype=np.uint8)

    class Camera:
        def __init__(self, source):
            assert source == 0

        def isOpened(self):
            return True

        def getBackendName(self):
            return "test camera"

        def set(self, prop, value):
            pass

        def get(self, prop):
            return 25.0

        def read(self):
            shown.wait(0.005)
            return True, image.copy()

        def release(self):
            events.append("camera closed")

    class Ssh:
        def __init__(self, options, session, source, log):
            self.session = session
            self.channel = self

        def open(self):
            return dict(environment={}, model={}, model_load_ms=1), 2

        def send(self, message, payload, *, timeout):
            assert payload.startswith(b"\xff\xd8")
            self.frame = message

        def receive(self, *, timeout):
            return (
                dict(
                    self.frame,
                    kind="result",
                    people=[],
                    inference_ms=None,
                    adapter_ms=1,
                    decode_ms=0.1,
                    measurement_method=None,
                ),
                b"",
            )

        def check(self):
            pass

        def close(self, *, in_flight):
            events.append("SSH closed")
            return 0

        def close_pipes(self):
            pass

    def display(window, rendered):
        assert rendered.shape == image.shape
        assert np.array_equal(rendered[-1], image[-1])
        shown.set()

    monkeypatch.setattr(live, "SshSession", Ssh)
    monkeypatch.setattr(live, "backend", lambda: cv2)
    monkeypatch.setattr(live_capture, "backend", lambda: cv2)
    monkeypatch.setattr(cv2, "VideoCapture", Camera)
    monkeypatch.setattr(cv2, "currentUIFramework", lambda: "TEST")
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", display)
    monkeypatch.setattr(cv2, "waitKey", lambda *_: ord("q") if shown.wait(0.005) else -1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_: events.append("window closed"))

    out = tmp_path / "live"
    args = parser().parse_args(
        [
            "live",
            "--remote",
            "lab",
            "--remote-root",
            "/project",
            "--output",
            str(out),
            "--limit",
            "10",
        ]
    )
    summary = live.run_live(live.live_options(args, {}))
    assert summary["status"] == "success", summary["error"]
    assert summary["stop_reason"] == "q"
    assert summary["counts"]["displayed"] == 1
    assert summary["timing"]["display_latency_ms"]["count"] == 1
    assert sorted(events) == ["SSH closed", "camera closed", "window closed"]
    assert sorted(path.name for path in out.iterdir()) == [
        "environment.json",
        "run.log",
        "summary.json",
        "timings.csv",
    ]
    with (out / "timings.csv").open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    assert row["displayed"] == "1" and float(row["display_latency_ms"]) >= 0
