# MRacing Camera

Project workspace for initial live camera capture and cone detection. Structure and documentation only; camera/model code has not been implemented.

## Current scope

- Basler a2A1920 family for initial tests; full suffix/interface pending.
- Previously linked Arducam B0353 is the intended camera reference; confirm SKU and Jetson compatibility.
- Capture target: 1920 × 1080 at 30 FPS; achieved performance must be measured.
- Compute: Jetson Orin Nano Super. First development/test host (Windows or Jetson) pending.
- Team-reported pylon Viewer: 26.08. Python capture will use a compatible pypylon version; Viewer can remain closed.
- Team-reported model: YOLO v27 medium. Supplied PyTorch checkpoint: best.pt. Confirm the compatible inference package and checkpoint metadata before loading.
- LiDAR, fusion, planning and driving are deferred.

## Open in VS Code

Open this folder as the workspace. Select a project Python environment when we confirm the Python version. No dependencies have been installed or pinned yet.

## Structure

- `docs/SETUP.md`: decisions needed before installation and first hardware test.
- `src/`: reserved for acquisition, preview and inference code.
- `configs/`: reserved for camera/model settings.
- `models/`: local model weights; ignored by Git except the README.
- `data/`: local recordings/datasets; ignored by Git except the README.

## First implementation sequence

1. Discover the Basler camera and report its model/serial number.
2. Display live frames and achieved FPS; save sample images.
3. Load compatible model weights and overlay detections.
4. Record capture FPS, inference throughput and end-to-end delay separately.
5. Integrate the intended Arducam acquisition path later.

## Project reference

[ALL INFO](https://docs.google.com/document/d/1r2eZJjpH60jpXqhf9EQnMrlORoWUxrvEbyuic_40h4A/edit)

The parent MRACING resources/presentations remain outside this source repository.
