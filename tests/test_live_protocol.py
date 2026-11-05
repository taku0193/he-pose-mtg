import os
import struct
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from hepose.live_protocol import Channel, ProtocolError, decode_image, validate_reply
from hepose.live_worker import serve_session


def jpeg(width=32, height=24):
    ok, data = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
    assert ok
    return data.tobytes()


def test_bounded_packet_round_trip():
    read, write = os.pipe()
    try:
        channel = Channel(read, write)
        channel.send({"kind": "frame"}, jpeg(), timeout=1)
        assert channel.receive(timeout=1) == ({"kind": "frame"}, jpeg())
    finally:
        os.close(read)
        os.close(write)


def test_oversize_header_rejected_before_body_read():
    read, write = os.pipe()
    try:
        os.write(write, struct.pack("!II", 4_000_000, 0))
        with pytest.raises(ProtocolError):
            Channel(read, write).receive(timeout=1)
    finally:
        os.close(read)
        os.close(write)


def test_partial_packet_has_deadline():
    read, write = os.pipe()
    try:
        os.write(write, b"\x00")
        with pytest.raises(TimeoutError):
            Channel(read, write).receive(timeout=0.02)
    finally:
        os.close(read)
        os.close(write)


def test_jpeg_dimensions_are_checked_before_opencv_decode(monkeypatch):
    data = jpeg()
    monkeypatch.setattr(cv2, "imdecode", lambda *_: pytest.fail("unexpected decode"))
    with pytest.raises(ProtocolError, match="size"):
        decode_image(data, 31, 24)


@pytest.mark.parametrize("field,value", [("session", "wrong"), ("frame", 8), ("width", 33), ("wire", 2)])
def test_reply_cannot_be_rendered_on_another_frame(field, value):
    reply = {"kind": "result", "session": "session", "frame": 7, "width": 32, "height": 24,
             "wire": 1, "people": [], "inference_ms": None, "adapter_ms": 1.0, "decode_ms": 0.3,
             "measurement_method": None}
    reply[field] = value
    with pytest.raises(ProtocolError):
        validate_reply(reply, "session", 7, 32, 24)


@dataclass
class FakeChannel:
    packets: list
    replies: list

    def receive(self, *, timeout):
        if not self.packets:
            raise EOFError
        return self.packets.pop(0)

    def send(self, message, payload=b"", *, timeout):
        self.replies.append(message)


def test_handshake_mismatch_rejected_before_loading_model():
    channel = FakeChannel([({"kind": "hello", "wire": 1, "app": "bad", "session": "x",
                             "token": "t", "revision": "abc"}, b"")], [])
    with pytest.raises(ProtocolError):
        serve_session(channel, "t", {"git_commit": "abc"}, lambda: pytest.fail("model loaded"))


def test_disconnect_closes_worker_adapter():
    from hepose import __version__

    class Adapter:
        closed = False

        def environment(self):
            return {"device": "cpu"}

        def close(self):
            self.closed = True

    adapter = Adapter()
    channel = FakeChannel([({"kind": "hello", "wire": 1, "app": __version__, "session": "x",
                             "token": "t", "revision": "abc"}, b"")], [])
    serve_session(channel, "t", {"git_commit": "abc"}, lambda: adapter)
    assert adapter.closed
    assert channel.replies[0]["kind"] == "ready"
