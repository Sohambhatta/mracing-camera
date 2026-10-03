from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .model import Detection
from .sources import Frame


@dataclass(frozen=True)
class Performance:
    inference_ms: float | None
    process_fps: float
    capture_fps: float
    host_age_ms: float


def annotate(
    frame: Frame,
    detections: list[Detection],
    performance: Performance,
) -> np.ndarray:
    image = frame.image.copy()
    height, width = image.shape[:2]
    for detection in detections:
        x1 = max(0, min(width - 1, round(detection.x1)))
        y1 = max(0, min(height - 1, round(detection.y1)))
        x2 = max(0, min(width - 1, round(detection.x2)))
        y2 = max(0, min(height - 1, round(detection.y2)))
        color = (0, 220, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    inference = "n/a" if performance.inference_ms is None else f"{performance.inference_ms:.1f} ms"
    lines = (
        f"{width}x{height} | process {performance.process_fps:.1f} FPS | capture {performance.capture_fps:.1f} FPS",
        f"inference {inference} | host frame age {performance.host_age_ms:.1f} ms",
    )
    for line_index, line in enumerate(lines):
        y = 22 + line_index * 22
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return image


def host_frame_age_ms(frame: Frame, now: float | None = None) -> float:
    return max(0.0, ((time.monotonic() if now is None else now) - frame.acquired_monotonic) * 1000)
