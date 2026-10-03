from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .app import run_application
from .config import AppConfig, ConfigurationError, build_config
from .model import ModelError
from .sources import BaslerSource


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON config file; explicit CLI options override it")
    parser.add_argument("--source", choices=("usb", "basler", "image", "video"))
    parser.add_argument("--camera", help="USB camera index or device path (default: 0)")
    parser.add_argument("--serial", help="Basler camera serial number")
    parser.add_argument("--input", help="image or video file path")
    parser.add_argument("--weights", help="model checkpoint path")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--preview", choices=("window", "web", "none"))
    parser.add_argument("--host", help="web preview bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int)
    parser.add_argument("--record", help="write annotated video to this path")
    parser.add_argument("--snapshot", help="save the first processed frame to this image path")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-backend", choices=("ultralytics",))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mracing-camera")
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="show camera/file frames without loading a model")
    _add_run_options(preview)
    detect = commands.add_parser("detect", help="run cone detection and optionally show annotated frames")
    _add_run_options(detect)
    cameras = commands.add_parser("list-cameras", help="discover connected Basler cameras")
    cameras.add_argument("--source", choices=("basler",), default="basler")
    diagnostics = commands.add_parser("diagnostics", help="report Python, dependencies, model and cameras")
    diagnostics.add_argument("--weights", default="models/best.pt")
    diagnostics.add_argument("--usb-probe-count", type=int, default=4)
    return parser


def _runtime_diagnostics(weights: str, usb_probe_count: int) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    for package in ("numpy", "opencv-python", "pypylon", "torch", "ultralytics"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package}: {version}")
    print(f"Model path {weights}: {'present' if Path(weights).is_file() else 'missing'}")

    try:
        import torch

        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    except (ImportError, OSError) as exc:
        print(f"PyTorch CUDA available: unavailable ({exc})")

    try:
        devices = BaslerSource.enumerate_devices()
        print(f"Basler cameras: {devices if devices else 'none found'}")
    except (RuntimeError, OSError) as exc:
        print(f"Basler cameras: unavailable ({exc})")

    import cv2

    usb_devices: list[int] = []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    for index in range(max(0, usb_probe_count)):
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    usb_devices.append(index)
        finally:
            capture.release()
    print(f"USB camera indices that returned a frame: {usb_devices if usb_devices else 'none found'}")
    return 0


def _run_config(args: argparse.Namespace) -> AppConfig:
    cli_values: dict[str, Any] = vars(args).copy()
    cli_values.pop("command", None)
    config_path = cli_values.pop("config", None)
    config = build_config(config_path, cli_values)
    return replace(config, detect=args.command == "detect")


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "list-cameras":
            devices = BaslerSource.enumerate_devices()
            if not devices:
                print("No Basler cameras found.")
            else:
                for device in devices:
                    print(f"{device['serial']}: {device['model']} ({device['name']})")
            return 0
        if args.command == "diagnostics":
            return _runtime_diagnostics(args.weights, args.usb_probe_count)
        config = _run_config(args)
        run_application(config)
        return 0
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 130
    except (ConfigurationError, ModelError, RuntimeError, OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
