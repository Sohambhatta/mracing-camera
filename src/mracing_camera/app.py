from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .annotation import Performance, annotate, host_frame_age_ms
from .config import AppConfig
from .model import Detection, UltralyticsAdapter, resolve_model
from .preview import PreviewPublisher, PreviewServer
from .sources import Frame, FrameSource, create_source

LOGGER = logging.getLogger(__name__)


class LatestFrameBuffer:
    """A single-slot handoff; publishing a frame replaces any unconsumed older frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: Frame | None = None
        self._error: BaseException | None = None
        self._ended = False

    def publish(self, frame: Frame) -> None:
        with self._condition:
            self._frame = frame
            self._condition.notify_all()

    def finish(self, error: BaseException | None = None) -> None:
        with self._condition:
            self._error = error
            self._ended = True
            self._condition.notify_all()

    def next(self, after_frame_id: int, timeout: float = 0.25) -> Frame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: (self._frame is not None and self._frame.frame_id > after_frame_id)
                or self._ended,
                timeout,
            )
            if self._error is not None:
                raise RuntimeError(f"camera capture failed: {self._error}") from self._error
            if self._frame is not None and self._frame.frame_id > after_frame_id:
                return self._frame
            return None

    @property
    def ended(self) -> bool:
        with self._condition:
            return self._ended


class CaptureWorker:
    def __init__(self, source: FrameSource) -> None:
        self.source = source
        self.buffer = LatestFrameBuffer()
        self._stop = threading.Event()
        self.capture_fps = 0.0
        self._thread = threading.Thread(target=self._capture, name="camera-capture", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _capture(self) -> None:
        error: Exception | None = None
        last_timestamp: float | None = None
        try:
            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:
                    break
                if last_timestamp is not None and frame.acquired_monotonic > last_timestamp:
                    measured_fps = 1 / (frame.acquired_monotonic - last_timestamp)
                    self.capture_fps = measured_fps if self.capture_fps == 0 else (
                        self.capture_fps * 0.8 + measured_fps * 0.2
                    )
                last_timestamp = frame.acquired_monotonic
                self.buffer.publish(frame)
        except Exception as exc:
            error = exc
        finally:
            try:
                self.source.close()
            except Exception as exc:
                error = error or exc
            self.buffer.finish(error)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6)
        if self._thread.is_alive():
            raise RuntimeError("camera capture worker did not stop within 6 seconds")


def _save_snapshot(path: str | Path, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"could not save snapshot to {output}")
    LOGGER.info("Saved snapshot to %s", output)


def run_application(
    config: AppConfig,
    *,
    source: FrameSource | None = None,
    model: UltralyticsAdapter | None = None,
    model_factory: Callable[[AppConfig], UltralyticsAdapter] = resolve_model,
) -> None:
    live_source: FrameSource | None = None
    worker: CaptureWorker | None = None
    capture_open = False
    preview_server: PreviewServer | None = None
    writer: cv2.VideoWriter | None = None
    window_open = False
    try:
        if model is None and config.detect:
            model = model_factory(config)
        live_source = source or create_source(config)
        capture_open = True
        live_source.open()

        if config.preview == "web":
            publisher = PreviewPublisher()
            preview_server = PreviewServer(publisher, config.host, config.port)
            preview_server.start()
            host, port = preview_server.address
            LOGGER.info("Browser preview: http://%s:%s", host, port)

        if getattr(live_source, "is_live", False):
            worker = CaptureWorker(live_source)
            worker.start()

        last_frame_id = 0
        previous_process_time: float | None = None
        previous_capture_time: float | None = None
        process_fps = 0.0
        capture_fps = 0.0
        if config.snapshot:
            one_shot_snapshot = config.snapshot
        else:
            one_shot_snapshot = None

        while True:
            if worker is not None:
                frame = worker.buffer.next(last_frame_id, timeout=0.05)
                if frame is None:
                    if worker.buffer.ended:
                        break
                    if config.preview == "window":
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    continue
            else:
                frame = live_source.read()
                if frame is None:
                    break
            last_frame_id = frame.frame_id

            now = time.monotonic()
            if previous_process_time is not None and now > previous_process_time:
                process_fps = 1 / (now - previous_process_time)
            if worker is None and previous_capture_time is not None and frame.acquired_monotonic > previous_capture_time:
                capture_fps = 1 / (frame.acquired_monotonic - previous_capture_time)
            elif worker is not None:
                capture_fps = worker.capture_fps
            previous_process_time = now
            previous_capture_time = frame.acquired_monotonic

            infer_start = time.perf_counter()
            detections: list[Detection] = model.infer(frame) if model is not None else []
            inference_ms = (time.perf_counter() - infer_start) * 1000 if model is not None else None
            output = annotate(
                frame,
                detections,
                Performance(inference_ms, process_fps, capture_fps, host_frame_age_ms(frame)),
            )

            if config.record:
                if writer is None:
                    target = Path(config.record)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(target), fourcc, config.fps, (output.shape[1], output.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open video writer for {target}")
                writer.write(output)

            if one_shot_snapshot is not None:
                _save_snapshot(one_shot_snapshot, output)
                one_shot_snapshot = None

            if config.preview == "window":
                cv2.imshow("MRacing Camera", output)
                window_open = True
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    _save_snapshot(Path("runs") / f"snapshot-{stamp}-{frame.frame_id}.jpg", output)
            elif config.preview == "web":
                assert preview_server is not None
                preview_server.publisher.publish(
                    output,
                    {
                        "frame_id": frame.frame_id,
                        "resolution": f"{output.shape[1]}x{output.shape[0]}",
                        "process_fps": round(process_fps, 2),
                        "capture_fps": round(capture_fps, 2),
                        "inference_ms": None if inference_ms is None else round(inference_ms, 2),
                        "host_frame_age_ms": round(host_frame_age_ms(frame), 2),
                        "detections": len(detections),
                    },
                )
            if config.source == "image":
                break
    finally:
        cleanup_actions: list[tuple[str, Callable[[], None]]] = []
        if worker is not None:
            cleanup_actions.append(("camera worker", worker.close))
        elif capture_open and live_source is not None:
            cleanup_actions.append(("camera source", live_source.close))
        if writer is not None:
            cleanup_actions.append(("video writer", writer.release))
        if window_open:
            cleanup_actions.append(("preview window", cv2.destroyAllWindows))
        if preview_server is not None:
            cleanup_actions.append(("preview server", preview_server.close))
        for resource_name, cleanup in cleanup_actions:
            try:
                cleanup()
            except Exception:
                LOGGER.exception("Failed to clean up %s", resource_name)
