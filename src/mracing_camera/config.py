from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class AppConfig:
    detect: bool = False
    source: str = "usb"
    camera: str = "0"
    serial: str | None = None
    input: str | None = None
    weights: str = "models/best.pt"
    device: str = "cpu"
    confidence: float = 0.25
    imgsz: int = 640
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    preview: str = "window"
    host: str = "127.0.0.1"
    port: int = 8080
    record: str | None = None
    snapshot: str | None = None
    loop: bool = False
    model_backend: str = "ultralytics"

    def validate(self) -> AppConfig:
        if self.source not in {"usb", "basler", "image", "video"}:
            raise ConfigurationError("source must be usb, basler, image, or video")
        if self.preview not in {"window", "web", "none"}:
            raise ConfigurationError("preview must be window, web, or none")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ConfigurationError("device must be cpu, cuda, or auto")
        if self.model_backend != "ultralytics":
            raise ConfigurationError("model-backend currently supports only ultralytics")
        if not 0 <= self.confidence <= 1:
            raise ConfigurationError("confidence must be between 0 and 1")
        if self.imgsz <= 0 or self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ConfigurationError("imgsz, width, height, and fps must be positive")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        if self.source in {"image", "video"} and not self.input:
            raise ConfigurationError(f"--input is required when source is {self.source}")
        if self.source == "basler" and self.camera != "0":
            raise ConfigurationError("use --serial to select a Basler camera, not --camera")
        return self


def load_config_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"could not read config file {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in config file {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("config file must contain a JSON object")
    valid_keys = {field.name for field in fields(AppConfig)}
    unknown = sorted(set(raw) - valid_keys)
    if unknown:
        raise ConfigurationError(f"unknown config option(s): {', '.join(unknown)}")
    return raw


def build_config(config_path: str | None, cli_values: dict[str, Any]) -> AppConfig:
    values = load_config_file(config_path)
    values.update({key: value for key, value in cli_values.items() if value is not None})
    values.pop("config", None)
    return AppConfig(**values).validate()
