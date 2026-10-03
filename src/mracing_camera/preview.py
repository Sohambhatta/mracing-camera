from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np


class PreviewPublisher:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._status: dict[str, Any] = {"state": "starting"}
        self._sequence = 0

    def publish(self, image: np.ndarray, status: dict[str, Any]) -> None:
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("could not encode preview frame as JPEG")
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._status = dict(status)
            self._sequence += 1
            self._condition.notify_all()

    def snapshot(self, after: int = 0, timeout: float = 5.0) -> tuple[int, bytes | None, dict[str, Any]]:
        with self._condition:
            if self._sequence <= after:
                self._condition.wait_for(lambda: self._sequence > after, timeout)
            return self._sequence, self._jpeg, dict(self._status)


class PreviewServer:
    def __init__(self, publisher: PreviewPublisher, host: str, port: int) -> None:
        self.publisher = publisher
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    body = (
                        "<!doctype html><title>MRacing Camera</title>"
                        "<h1>MRacing Camera</h1><pre id='status'>Connecting…</pre>"
                        "<img src='/mjpeg' style='max-width:100%;height:auto'>"
                        "<script>setInterval(async()=>{try{let r=await fetch('/status');"
                        "document.querySelector('#status').textContent=JSON.stringify(await r.json(),null,2)}"
                        "catch(e){}},1000)</script>"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/status":
                    _, _, status = owner.publisher.snapshot(timeout=0)
                    body = json.dumps(status).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/mjpeg":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    sequence = 0
                    try:
                        while True:
                            sequence, image, _ = owner.publisher.snapshot(sequence, timeout=15)
                            if image is None:
                                continue
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(image)).encode()
                                + b"\r\n\r\n"
                                + image
                                + b"\r\n"
                            )
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        return
                else:
                    self.send_error(404)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="preview-http", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address[:2]

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
