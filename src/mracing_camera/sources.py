from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import AppConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    frame_id: int
    acquired_monotonic: float


class FrameSource(Protocol):
    is_live: bool

    def open(self) -> None: ...

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


class OpenCvSource:
    is_live = False

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._frame_id = 0

    def _open_capture(self, source: int | str, config: AppConfig) -> None:
        if os.name == "nt":
            backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
        elif os.name == "posix":
            backends = (cv2.CAP_V4L2, cv2.CAP_ANY)
        else:
            backends = (cv2.CAP_ANY,)

        for backend in backends:
            capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                self._capture = capture
                break
            capture.release()
        if self._capture is None:
            raise RuntimeError(f"could not open camera/video source {source!r}")

        if isinstance(source, int):
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            self._capture.set(cv2.CAP_PROP_FPS, config.fps)
            actual = (
                int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                self._capture.get(cv2.CAP_PROP_FPS),
            )
            LOGGER.info("Camera negotiated %sx%s at %.2f FPS", *actual)
            if actual[0] != config.width or actual[1] != config.height:
                LOGGER.warning(
                    "Requested %sx%s; camera reports %sx%s",
                    config.width,
                    config.height,
                    actual[0],
                    actual[1],
                )
            if not actual[2]:
                LOGGER.warning("Requested %.2f FPS; camera did not report a capture rate", config.fps)
            elif abs(actual[2] - config.fps) > 0.5:
                LOGGER.warning("Requested %.2f FPS; camera reports %.2f FPS", config.fps, actual[2])

    def _read(self) -> Frame | None:
        if self._capture is None:
            raise RuntimeError("source is not open")
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        self._frame_id += 1
        return Frame(image, self._frame_id, time.monotonic())

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class UsbCameraSource(OpenCvSource):
    is_live = True

    def __init__(self, camera: str, config: AppConfig) -> None:
        super().__init__()
        self.camera = int(camera) if camera.isdecimal() else camera
        self.config = config

    def open(self) -> None:
        self._open_capture(self.camera, self.config)

    def read(self) -> Frame:
        frame = self._read()
        if frame is None:
            raise RuntimeError("USB camera stopped producing frames or disconnected")
        return frame


class ImageSource(OpenCvSource):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = Path(path)
        self._returned = False

    def open(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"image not found: {self.path}")

    def read(self) -> Frame | None:
        if self._returned:
            return None
        image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode image: {self.path}")
        self._returned = True
        self._frame_id = 1
        return Frame(image, 1, time.monotonic())


class VideoSource(OpenCvSource):
    def __init__(self, path: str, loop: bool = False) -> None:
        super().__init__()
        self.path = Path(path)
        self.loop = loop

    def open(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        self._open_capture(str(self.path), AppConfig())

    def read(self) -> Frame | None:
        frame = self._read()
        if frame is not None or not self.loop:
            return frame
        assert self._capture is not None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = self._read()
        if frame is None:
            raise RuntimeError(f"video contains no readable frames: {self.path}")
        return frame


class BaslerSource:
    is_live = True

    def __init__(self, serial: str | None, config: AppConfig) -> None:
        self.serial = serial
        self.config = config
        self._camera = None
        self._pylon = None
        self._converter = None
        self._frame_id = 0

    @staticmethod
    def enumerate_devices() -> list[dict[str, str]]:
        try:
            from pypylon import pylon
        except (ImportError, OSError) as exc:
            raise RuntimeError("Basler support requires the optional 'pypylon' package and pylon runtime") from exc
        result = []
        for device in pylon.TlFactory.GetInstance().EnumerateDevices():
            result.append(
                {
                    "serial": device.GetSerialNumber(),
                    "model": device.GetModelName(),
                    "name": device.GetFriendlyName(),
                }
            )
        return result

    def open(self) -> None:
        try:
            from pypylon import pylon
        except (ImportError, OSError) as exc:
            raise RuntimeError("Basler support requires the optional 'pypylon' package and pylon runtime") from exc

        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        if not devices:
            raise RuntimeError("no Basler cameras found; check the pylon runtime, connection, and camera power")
        selected = next((item for item in devices if item.GetSerialNumber() == self.serial), None)
        if self.serial is None:
            if len(devices) != 1:
                serials = ", ".join(item.GetSerialNumber() for item in devices)
                raise RuntimeError(f"multiple Basler cameras found ({serials}); select one with --serial")
            selected = devices[0]
        elif selected is None:
            raise RuntimeError(f"Basler camera with serial {self.serial!r} was not found")

        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(selected))
        camera.Open()
        self._camera = camera
        self._pylon = pylon
        for feature, value in (
            ("Width", self.config.width),
            ("Height", self.config.height),
            ("AcquisitionFrameRate", self.config.fps),
        ):
            node = getattr(camera, feature, None)
            try:
                if node is not None and node.IsWritable():
                    node.SetValue(value)
                else:
                    LOGGER.warning("Basler feature %s is not writable; using camera default", feature)
            except (AttributeError, RuntimeError, ValueError) as exc:
                LOGGER.warning("Could not set Basler feature %s: %s", feature, exc)

        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        LOGGER.info(
            "Opened Basler %s (%s), %sx%s at %s FPS",
            selected.GetSerialNumber(),
            selected.GetModelName(),
            camera.Width.GetValue() if hasattr(camera, "Width") else "unknown",
            camera.Height.GetValue() if hasattr(camera, "Height") else "unknown",
            camera.AcquisitionFrameRate.GetValue()
            if hasattr(camera, "AcquisitionFrameRate")
            else "unknown",
        )

    def read(self) -> Frame:
        if self._camera is None:
            raise RuntimeError("Basler source is not open")
        result = self._camera.RetrieveResult(5000, self._pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f"Basler grab failed: {result.ErrorDescription}")
            image = self._converter.Convert(result).GetArray().copy()
        finally:
            result.Release()
        self._frame_id += 1
        return Frame(image, self._frame_id, time.monotonic())

    def close(self) -> None:
        if self._camera is not None:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
            self._camera.Close()
            self._camera = None


def create_source(config: AppConfig) -> FrameSource:
    if config.source == "usb":
        return UsbCameraSource(config.camera, config)
    if config.source == "basler":
        return BaslerSource(config.serial, config)  # type: ignore[return-value]
    if config.source == "image":
        return ImageSource(config.input or "")
    if config.source == "video":
        return VideoSource(config.input or "", config.loop)
    raise ValueError(f"unsupported source: {config.source}")
