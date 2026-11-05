"""Start an owned worker and connect to its loopback listener with OpenSSH -W."""

import re
import shlex
import subprocess
from time import monotonic

from hepose import __version__
from hepose.live_protocol import WIRE, Channel, ProtocolError, identity
from hepose.schema import finite


class SshSession:
    def __init__(self, options, session, source, log):
        self.options = options
        self.session = session
        self.source = source
        self.log = log
        self.worker = self.relay = None

    def _start(self, arguments, *, relay=False):
        return subprocess.Popen(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "UpdateHostKeys=no",
                "-o",
                "ControlPath=none",
                "-o",
                "ClearAllForwardings=yes",
                "-o",
                f"ConnectTimeout={self.options.connect_timeout}",
                "-o",
                "ServerAliveInterval=5",
                "-o",
                "ServerAliveCountMax=2",
                *arguments,
            ],
            stdin=subprocess.PIPE if relay else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=self.log,
        )

    def open(self):
        options = self.options
        command = [
            options.remote_python,
            "-m",
            "hepose.live_worker",
            "--config",
            options.remote_config,
            "--startup-timeout",
            str(options.startup_timeout),
            "--io-timeout",
            str(options.io_timeout),
            "--session-timeout",
            str(options.session_timeout),
        ]
        if options.device:
            command.extend(("--device", options.device))
        started = monotonic()
        self.worker = self._start(
            [options.remote, f"cd {shlex.quote(options.remote_root)} && exec {shlex.join(command)}"]
        )
        try:
            boot, data = Channel(self.worker.stdout.fileno(), self.worker.stdout.fileno()).receive(
                timeout=options.startup_timeout
            )
            token, port = boot.get("token"), boot.get("port")
            identity(boot, "listen", None)
            if (
                data
                or type(port) is not int
                or not 1 <= port <= 65535
                or not isinstance(token, str)
                or not re.fullmatch(r"[0-9a-f]{64}", token)
            ):
                raise ProtocolError("invalid worker bootstrap")
            self.relay = self._start(["-W", f"127.0.0.1:{port}", options.remote], relay=True)
            self.channel = Channel(self.relay.stdout.fileno(), self.relay.stdin.fileno())
            self.channel.send(
                {
                    "kind": "hello",
                    "wire": WIRE,
                    "app": __version__,
                    "revision": self.source["git_commit"],
                    "session": self.session,
                    "token": token,
                },
                timeout=options.startup_timeout - (monotonic() - started),
            )
            ready, data = self.channel.receive(
                timeout=options.startup_timeout - (monotonic() - started)
            )
        except EOFError as exc:
            raise ProtocolError("SSH or worker exited during startup; see run.log") from exc
        identity(ready, "ready", self.session)
        source, model = ready.get("source"), ready.get("model")
        if (
            data
            or ready.get("app") != __version__
            or not isinstance(source, dict)
            or source.get("git_commit") != self.source["git_commit"]
            or not isinstance(model, dict)
            or model.get("model_id") != "yolo26"
            or not isinstance(model.get("gpu"), dict)
            or not model["gpu"].get("gpu_uuid")
        ):
            raise ProtocolError("invalid worker version or GPU metadata")
        finite(ready.get("model_load_ms"), "model_load_ms", 0)
        return ready, (monotonic() - started) * 1000

    def check(self):
        if self.worker.poll() is not None or self.relay.poll() is not None:
            raise ProtocolError("SSH connection or live worker exited; see run.log")

    @staticmethod
    def _stop(process, *, graceful):
        if process is None:
            return None
        if graceful:
            try:
                return process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        process.terminate()
        try:
            return process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=1)

    def close(self, *, in_flight=False):
        try:
            if self.relay and not in_flight:
                self.relay.stdin.close()  # EOF reaches the worker through direct-tcpip.
            self._stop(self.relay, graceful=not in_flight)
        finally:
            return_code = self._stop(self.worker, graceful=True)
        return return_code

    def close_pipes(self):
        for process in (self.relay, self.worker):
            if process:
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        pipe.close()
