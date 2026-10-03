from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from mracing_camera.app import LatestFrameBuffer, run_application
from mracing_camera.config import AppConfig, build_config
from mracing_camera.model import ModelError, load_model, parse_result
from mracing_camera.sources import Frame, ImageSource


class ConfigTests(unittest.TestCase):
    def test_cli_values_override_config_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.json"
            config_file.write_text('{"camera": "2", "width": 1280, "preview": "none"}', encoding="utf-8")
            config = build_config(str(config_file), {"camera": "Logitech HD Pro", "width": None})
        self.assertEqual(config.camera, "Logitech HD Pro")
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.preview, "none")


class ModelTests(unittest.TestCase):
    def test_missing_weights_have_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(weights=str(Path(directory) / "missing.pt"))
            with self.assertRaisesRegex(ModelError, "weights not found"):
                load_model(config)

    def test_result_coordinates_and_names_are_preserved(self) -> None:
        boxes = SimpleNamespace(
            xyxy=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[10, 20, 90, 80]]))),
            cls=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([1]))),
            conf=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([0.875]))),
        )
        detections = parse_result(SimpleNamespace(boxes=boxes), {0: "blue", 1: "yellow"})
        self.assertEqual(len(detections), 1)
        self.assertEqual((detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2), (10, 20, 90, 80))
        self.assertEqual(detections[0].class_name, "yellow")
        self.assertAlmostEqual(detections[0].confidence, 0.875)


class FrameTests(unittest.TestCase):
    def test_latest_frame_buffer_replaces_older_unconsumed_frame(self) -> None:
        buffer = LatestFrameBuffer()
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        buffer.publish(Frame(image, 1, 1.0))
        buffer.publish(Frame(image, 2, 2.0))
        newest = buffer.next(after_frame_id=0, timeout=0)
        self.assertIsNotNone(newest)
        self.assertEqual(newest.frame_id, 2)

    def test_image_source_returns_one_frame_then_eof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((8, 12, 3), dtype=np.uint8)))
            source = ImageSource(str(path))
            source.open()
            frame = source.read()
            self.assertEqual(frame.image.shape, (8, 12, 3))
            self.assertIsNone(source.read())

    def test_application_closes_source_at_end_of_file(self) -> None:
        class FakeSource:
            is_live = False

            def __init__(self) -> None:
                self.closed = False
                self.read_count = 0

            def open(self) -> None:
                pass

            def read(self) -> Frame | None:
                self.read_count += 1
                if self.read_count > 1:
                    return None
                return Frame(np.zeros((16, 16, 3), dtype=np.uint8), 1, 1.0)

            def close(self) -> None:
                self.closed = True

        source = FakeSource()
        config = AppConfig(source="image", input="unused.png", preview="none")
        run_application(config, source=source)
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()
