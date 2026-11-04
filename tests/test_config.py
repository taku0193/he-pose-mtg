import json
import subprocess
import sys

import pytest

from hepose import __version__
from hepose.cli import main
from hepose.config import load_config


def test_local_config_resolves_paths_relative_to_its_file(tmp_path):
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        "schema_version = 1\n[models.yolo26]\n"
        'python = "env/bin/python"\nweights = "weights/model.pt"\n'
        'device = "cuda:0"\n'
    )
    configuration = load_config(config_path)
    assert configuration.model("yolo26")["python"] == str(tmp_path / "env/bin/python")
    assert configuration.model("yolo26")["weights"] == str(tmp_path / "weights/model.pt")


def test_config_rejects_unknown_models_and_invalid_devices(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("schema_version = 1\n[models.other]\n")
    with pytest.raises(ValueError, match="model"):
        load_config(path)
    path.write_text('schema_version = 1\n[models.yolo26]\ndevice = "automatic"\n')
    with pytest.raises(ValueError, match="device"):
        load_config(path)


def test_doctor_dispatches_model_environments_sequentially(tmp_path, monkeypatch, capsys):
    path = tmp_path / "settings.toml"
    path.write_text(
        'schema_version = 1\n[models.yolo26]\npython = "yolo/bin/python"\n'
        '[models.detrpose]\npython = "detr/bin/python"\n'
        '[models.tar-vitpose]\npython = "tar/bin/python"\n'
    )
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        model = command[command.index("--model") + 1]
        payload = {
            "application_version": __version__,
            "role": "worker",
            "ready": False,
            "checks": {},
            "models": {model: {"status": "not_implemented"}},
        }
        return subprocess.CompletedProcess(command, 1, json.dumps(payload), "")

    monkeypatch.setattr("hepose.doctor.subprocess.run", worker)
    assert main(["doctor", "--config", str(path), "--model", "all", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert calls[0][0] == str(tmp_path / "yolo/bin/python")
    assert calls[1][0] == str(tmp_path / "detr/bin/python")
    assert all("--local" in command for command in calls[:2])
    assert result["models"]["yolo26"]["status"] == "not_implemented"


def test_run_failure_writes_summary_without_model_download(sample_video, tmp_path, capsys):
    output = tmp_path / "failed-run"
    assert (
        main(["run", "--model", "yolo26", "--input", str(sample_video), "--output", str(output)])
        == 1
    )
    assert "重み" in capsys.readouterr().err
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
    assert not (output / "annotated.mp4").exists()


def test_run_config_dispatches_worker_with_explicit_device_override(tmp_path, monkeypatch):
    path = tmp_path / "settings.toml"
    path.write_text(
        'schema_version = 1\n[models.yolo26]\npython = "env/bin/python"\ndevice = "cuda:0"\n'
    )
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr("hepose.cli.subprocess.run", worker)
    result = main(
        [
            "run",
            "--config",
            str(path),
            "--model",
            "yolo26",
            "--input",
            "input.mp4",
            "--output",
            str(tmp_path / "run"),
            "--device",
            "cpu",
        ]
    )
    assert result == 1
    command = calls[0]
    assert command[0] != sys.executable
    assert command[command.index("--device") + 1] == "cpu"
    assert "--local" in command


def test_doctor_rejects_malformed_worker_payload(tmp_path, monkeypatch, capsys):
    path = tmp_path / "settings.toml"
    path.write_text('schema_version = 1\n[models.yolo26]\npython = "env/bin/python"\n')
    monkeypatch.setattr(
        "hepose.doctor.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "[]", ""),
    )
    assert main(["doctor", "--config", str(path), "--model", "yolo26", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["models"]["yolo26"]["status"] == "environment_error"


def test_virtualenv_python_symlink_is_preserved(tmp_path):
    executable = tmp_path / "env/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    path = tmp_path / "settings.toml"
    path.write_text('schema_version = 1\n[models.yolo26]\npython = "env/bin/python"\n')
    assert load_config(path).model("yolo26")["python"] == str(executable)


def test_virtualenvs_sharing_base_python_still_dispatch(tmp_path, monkeypatch, capsys):
    executable = tmp_path / "env/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    path = tmp_path / "settings.toml"
    path.write_text('schema_version = 1\n[models.yolo26]\npython = "env/bin/python"\n')
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        payload = {
            "application_version": __version__,
            "checks": {},
            "models": {"yolo26": {"status": "not_implemented"}},
        }
        return subprocess.CompletedProcess(command, 1, json.dumps(payload), "")

    monkeypatch.setattr("hepose.doctor.subprocess.run", child)
    monkeypatch.setattr(
        "hepose.doctor.inspect",
        lambda *args: pytest.fail("model environment was collapsed into the controller"),
    )
    assert main(["doctor", "--config", str(path), "--model", "yolo26", "--json"]) == 1
    assert calls[0][0] == str(executable)
    capsys.readouterr()


def test_headless_opencv_cannot_mark_the_display_client_ready(monkeypatch):
    import cv2

    from hepose.doctor import inspect

    monkeypatch.setattr(cv2, "getBuildInformation", lambda: "GUI: NONE\n")
    report = inspect("client")
    assert report["checks"]["opencv_gui"]["status"] == "missing"
    assert not report["ready"]


def test_missing_worker_python_still_records_startup_failure(sample_video, tmp_path, capsys):
    path = tmp_path / "settings.toml"
    path.write_text('schema_version = 1\n[models.yolo26]\npython = "missing/bin/python"\n')
    output = tmp_path / "failed-worker"
    assert (
        main(
            [
                "run",
                "--config",
                str(path),
                "--model",
                "yolo26",
                "--input",
                str(sample_video),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "missing/bin/python" in capsys.readouterr().err
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "failed"
    assert "missing/bin/python" in report["error"]


def test_worker_sigint_is_normalized_to_130(sample_video, tmp_path, monkeypatch):
    path = tmp_path / "settings.toml"
    path.write_text('schema_version = 1\n[models.yolo26]\npython = "env/bin/python"\n')
    monkeypatch.setattr(
        "hepose.cli.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, -2),
    )
    output = tmp_path / "cancelled-worker"
    assert (
        main(
            [
                "run",
                "--config",
                str(path),
                "--model",
                "yolo26",
                "--input",
                str(sample_video),
                "--output",
                str(output),
            ]
        )
        == 130
    )
    assert json.loads((output / "summary.json").read_text())["status"] == "interrupted"
