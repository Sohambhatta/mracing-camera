# MRacing Camera

Project workspace for initial live camera capture and cone detection. The capabilities below are the first set of coding goals, not the final autonomous-driving system. Structure and documentation only; camera/model code has not been implemented.

## Initial coding capabilities

| Capability | Where the software runs | Camera connection | Intended output |
| --- | --- | --- | --- |
| Laptop development | Laptop | Logitech USB webcam plugged into the laptop | Live camera preview and YOLO cone-detection overlay; also evaluate the model on saved images/video. |
| Orin with Logitech | Jetson Orin Nano Super, accessed from the laptop through SSH / VS Code Remote SSH | Logitech USB webcam plugged into the Orin | Capture and inference on the Orin, with a live preview accessible from the laptop. |
| Orin with Basler | Jetson Orin Nano Super, accessed from the laptop through SSH / VS Code Remote SSH | Basler camera connected to the Orin through its confirmed interface | Basler frame acquisition and YOLO inference on the Orin, with a live preview accessible from the laptop. |

Keep camera acquisition separate from model inference and visualization so these modes can share the same downstream pipeline. Each host will need compatible dependencies; moving to the Orin does not imply copying the laptop's Python environment unchanged. The remote preview mechanism will be selected during implementation; SSH alone does not display live video.

For each capability, first verify stable camera frames, then add model inference and detection overlays. Record achieved capture FPS, inference throughput, and end-to-end delay separately. The original Jetson Nano is not a required development target.

## Next goal after live perception

Use cone observations to estimate the track layout and generate a candidate path through it. Localization estimates the vehicle's position and orientation relative to that layout; path planning uses the layout and vehicle state to choose a route. These are related but separate tasks, so tracing an ideal path is broader than localization alone.

Before this stage, decide how image detections become cone positions and how vehicle motion is measured. Start with track boundaries and a candidate centerline; an optimized racing line is a later refinement. This next goal is exploratory and does not yet include vehicle actuation or a complete autonomous-driving capability.

## Current scope

- Basler a2A1920 family for initial tests; full suffix/interface pending.
- Previously linked Arducam B0353 is the intended camera reference; confirm SKU and Jetson compatibility.
- Capture target: 1920 × 1080 at 30 FPS; achieved performance must be measured.
- Development hosts: laptop for the first capability, then Jetson Orin Nano Super for the two remote capabilities.
- Team-reported pylon Viewer: 26.08. Python capture will use a compatible pypylon version; Viewer can remain closed.
- Team-reported model: YOLO v27 medium. Supplied PyTorch checkpoint: best.pt. Confirm the compatible inference package and checkpoint metadata before loading.
- Immediate work is live camera capture and model inference. Track estimation, localization and path generation follow afterward; LiDAR, fusion and vehicle control are deferred.

## Open in VS Code

Open this folder as the workspace. Select a project Python environment when we confirm the Python version. No dependencies have been installed or pinned yet.

## Structure

- `docs/SETUP.md`: decisions needed before installation and first hardware test.
- `src/`: reserved for acquisition, preview and inference code.
- `configs/`: reserved for camera/model settings.
- `models/`: local model weights; ignored by Git except the README.
- `data/`: local recordings/datasets; ignored by Git except the README.

## First implementation sequence

1. Confirm the model-loading package and run saved-image/video inference on the laptop.
2. Add laptop Logitech capture, live preview and detection overlays.
3. Set up Orin network access, VS Code Remote SSH and compatible dependencies; repeat the Logitech pipeline on the Orin.
4. Confirm the full Basler model/interface, add Basler acquisition on the Orin and reuse inference/preview logic.
5. Measure performance for each mode and document reproducible setup steps.
6. Begin track estimation, localization and candidate path generation after the initial live-perception capabilities work.

The previously discussed Arducam remains a later hardware-integration consideration, outside these three initial capabilities.

## Project reference

[ALL INFO](https://docs.google.com/document/d/1r2eZJjpH60jpXqhf9EQnMrlORoWUxrvEbyuic_40h4A/edit)

The parent MRACING resources/presentations remain outside this source repository.
