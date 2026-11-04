"""Local paths and execution environments; credentials never belong in this configuration."""

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from hepose.schema import MODEL_IDS


def validate_device(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"cpu|cuda:\d+", value):
        raise ValueError("device must be cpu or an explicit cuda:N")
    return value


@dataclass(frozen=True)
class Configuration:
    path: Path | None = None
    models: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)

    def model(self, model_id: str) -> dict:
        return dict(self.models.get(model_id, {}))


def load_config(path: Path | None = None) -> Configuration:
    if path is None:
        candidate = Path("hepose.local.toml")
        if not candidate.exists():
            return Configuration()
        path = candidate
    path = Path(path).resolve()
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("configuration schema_version must be 1")
    if data.keys() - {"schema_version", "models", "live"}:
        raise ValueError("unknown configuration section")
    models = data.get("models", {})
    live = data.get("live", {})
    if not isinstance(models, dict) or not isinstance(live, dict):
        raise ValueError("models and live must be tables")
    if models.keys() - set(MODEL_IDS):
        raise ValueError("unknown model in configuration")
    resolved = {}
    path_fields = {"python", "weights", "repository", "config"}
    for name, model in models.items():
        if not isinstance(model, dict) or model.keys() - {*path_fields, "device", "gpu_uuid"}:
            raise ValueError(f"invalid model configuration: {name}")
        item = dict(model)
        for key in path_fields & item.keys():
            value = item[key]
            if not isinstance(value, str) or not value or "://" in value:
                raise ValueError(f"{name}.{key} must be a local path")
            value = Path(value).expanduser()
            # Invoking a virtualenv's symlink sets sys.prefix; resolving it selects the base env.
            item[key] = (
                os.path.abspath(path.parent / value)
                if key == "python"
                else str((path.parent / value).resolve())
            )
        validate_device(item.get("device", "cuda:0"))
        if "gpu_uuid" in item and not isinstance(item["gpu_uuid"], str):
            raise ValueError("gpu_uuid must be a string")
        resolved[name] = item
    return Configuration(path, resolved, live)
