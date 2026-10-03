from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .sources import Frame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ModelError(RuntimeError):
    """Raised when model loading or inference cannot proceed."""


def parse_result(result: Any, names: dict[int, str] | list[str] | tuple[str, ...]) -> list[Detection]:
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    for coordinates, raw_id, raw_confidence in zip(xyxy, class_ids, confidences):
        class_id = int(raw_id)
        class_name = names.get(class_id) if isinstance(names, dict) else (
            names[class_id] if 0 <= class_id < len(names) else None
        )
        if class_name is None:
            raise ModelError(f"model result used class ID {class_id} absent from model metadata")
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        detections.append(Detection(class_id, str(class_name), float(raw_confidence), x1, y1, x2, y2))
    return detections


class UltralyticsAdapter:
    def __init__(self, weights: str, device: str, confidence: float, imgsz: int) -> None:
        self.weights = Path(weights)
        self.requested_device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self.model = None
        self.names: dict[int, str] | list[str] | tuple[str, ...] | None = None
        self.device = ""

    def load(self) -> None:
        if not self.weights.is_file():
            raise ModelError(
                f"weights not found: {self.weights}. Supply the team's original best.pt; "
                "the checkpoint is intentionally not downloaded or replaced."
            )
        try:
            import torch
        except ImportError as exc:
            raise ModelError("model inference requires PyTorch; install a compatible PyTorch build first") from exc

        if self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise ModelError("CUDA was requested but PyTorch cannot access a CUDA device")
            selected_device = "cuda"
        elif self.requested_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = "cpu"

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError("this loader requires the optional 'ultralytics' package") from exc
        try:
            model = YOLO(str(self.weights))
            names = model.names
            if not names:
                raise ModelError("loaded model does not expose class names in its metadata")
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(
                f"Ultralytics could not load {self.weights}. A .pt suffix does not identify its model format; "
                "provide the training/export library and version used to create this checkpoint."
            ) from exc

        self.model = model
        self.names = names
        self.device = selected_device
        LOGGER.info(
            "Loaded Ultralytics %s on %s with classes %s",
            getattr(__import__("ultralytics"), "__version__", "unknown"),
            self.device,
            names,
        )

    def infer(self, frame: Frame) -> list[Detection]:
        if self.model is None or self.names is None:
            raise ModelError("model has not been loaded")
        try:
            results = self.model.predict(
                source=frame.image,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            if not results:
                return []
            return parse_result(results[0], self.names)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"inference failed for frame {frame.frame_id}: {exc}") from exc


def load_model(config: AppConfig) -> UltralyticsAdapter:
    model = UltralyticsAdapter(config.weights, config.device, config.confidence, config.imgsz)
    model.load()
    return model


def resolve_model(config: AppConfig) -> UltralyticsAdapter:
    return load_model(config)
