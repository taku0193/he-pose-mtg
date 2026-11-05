"""Bounded JSON/JPEG packets over an SSH byte stream."""

import json
import os
import select
import struct
from time import monotonic

from hepose.schema import COCO17, Keypoint, Person, finite, integer

WIRE = 1
MAX_JSON = 1024 * 1024
MAX_JPEG = 2 * 1024 * 1024
HEADER = struct.Struct("!II")
SOF = {0xC0, 0xC1, 0xC2}


class ProtocolError(RuntimeError):
    pass


def identity(message, kind, session):
    if (
        message.get("kind") != kind
        or type(message.get("wire")) is not int
        or message["wire"] != WIRE
        or message.get("session") != session
    ):
        raise ProtocolError(f"invalid {kind} packet identity")


class Channel:
    def __init__(self, read_fd, write_fd):
        self.read_fd = read_fd
        self.write_fd = write_fd
        os.set_blocking(read_fd, False)
        if write_fd != read_fd:
            os.set_blocking(write_fd, False)

    @staticmethod
    def _wait(fd, deadline, *, write=False):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("live packet deadline exceeded")
        readable, writable, _ = select.select(
            [] if write else [fd], [fd] if write else [], [], remaining
        )
        if not (readable or writable):
            raise TimeoutError("live packet deadline exceeded")

    def _read(self, size, deadline):
        data = bytearray()
        while len(data) < size:
            self._wait(self.read_fd, deadline)
            try:
                chunk = os.read(self.read_fd, size - len(data))
            except BlockingIOError:
                continue
            if not chunk:
                if data:
                    raise ProtocolError("truncated live packet")
                raise EOFError("live stream closed")
            data.extend(chunk)
        return bytes(data)

    def send(self, message, payload=b"", *, timeout):
        deadline = monotonic() + timeout
        header = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
        if not 0 < len(header) <= MAX_JSON or len(payload) > MAX_JPEG:
            raise ProtocolError("live packet exceeds size limit")
        for part in (HEADER.pack(len(header), len(payload)), header, payload):
            data = memoryview(part)
            while data:
                self._wait(self.write_fd, deadline, write=True)
                try:
                    data = data[os.write(self.write_fd, data) :]
                except BlockingIOError:
                    continue

    def receive(self, *, timeout):
        deadline = monotonic() + timeout
        head, body = HEADER.unpack(self._read(HEADER.size, deadline))
        if not 0 < head <= MAX_JSON or body > MAX_JPEG:
            raise ProtocolError("invalid live packet size")
        try:
            message = json.loads(self._read(head, deadline), parse_constant=_reject_constant)
            if not isinstance(message, dict):
                raise ValueError("JSON header is not an object")
            payload = self._read(body, deadline)
        except EOFError as exc:
            raise ProtocolError("truncated live packet") from exc
        except (ValueError, RecursionError) as exc:
            raise ProtocolError("invalid live packet JSON") from exc
        return message, payload


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def check_size(width, height):
    try:
        integer(width, "width", 1)
        integer(height, "height", 1)
    except ValueError as exc:
        raise ProtocolError("invalid image size") from exc
    if max(width, height) > 1920 or width * height > 1920 * 1080:
        raise ProtocolError("image size exceeds live limit (1920 x 1080 pixels)")


def jpeg_size(data):
    if not data.startswith(b"\xff\xd8"):
        raise ProtocolError("expected JPEG image")
    i = 2
    while i + 4 < len(data) and data[i] == 0xFF:
        i += 1
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i + 3 >= len(data) or data[i] in {0, 0xDA, 0xD9}:
            break
        marker = data[i]
        length = int.from_bytes(data[i + 1 : i + 3], "big")
        if length < 2 or i + 1 + length > len(data):
            break
        if marker in SOF and length >= 8:
            height, width = struct.unpack_from("!HH", data, i + 4)
            return width, height
        i += 1 + length
    raise ProtocolError("JPEG size marker is missing or invalid")


def decode_image(data, width, height):
    import cv2
    import numpy as np

    check_size(width, height)
    if len(data) > MAX_JPEG or jpeg_size(data) != (width, height):
        raise ProtocolError("JPEG size does not match frame metadata")
    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None or image.shape != (height, width, 3):
        raise ProtocolError("invalid decoded JPEG size")
    return image


def _person(item):
    if not isinstance(item, dict):
        raise ValueError("invalid person")
    points = item["keypoints"]
    if not isinstance(points, list) or len(points) > len(COCO17):
        raise ValueError("invalid keypoints")
    keypoints = tuple(Keypoint(**point) for point in points)
    if any(point.name not in COCO17 for point in keypoints):
        raise ValueError("invalid joint name")
    return Person(bbox=item["bbox"], keypoints=keypoints)


def validate_reply(reply, session, frame, width, height):
    identity(reply, "result", session)
    for name, expected in (("frame", frame), ("width", width), ("height", height)):
        value = reply.get(name)
        if type(value) is not int or value != expected:
            raise ProtocolError(f"mismatched live result {name}")
    try:
        for name in ("decode_ms", "adapter_ms"):
            finite(reply[name], name, 0)
        method = reply["measurement_method"]
        if method is not None and (not isinstance(method, str) or not method):
            raise ValueError("invalid measurement method")
        if reply["inference_ms"] is not None:
            finite(reply["inference_ms"], "inference_ms", 0)
            if method is None:
                raise ValueError("missing inference measurement method")
        people = reply["people"]
        if not isinstance(people, list) or len(people) > 300:
            raise ValueError("invalid people list")
        return tuple(_person(person) for person in people)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ProtocolError("invalid pose data or timings in live result") from exc
