# MRacing Camera

Python application for local live camera capture and cone detection. The capabilities below are the first set of coding goals, not the final autonomous-driving system.

## Implementation status — 2026-10-03

The first application pass is implemented. The intended order is still **Windows Logitech preview first, then model inference**; neither has been verified against physical hardware or the team's real checkpoint yet.

| Area | Status | Notes |
| --- | --- | --- |
| Python package and CLI | Implemented | Installable `src/mracing_camera/` package with `preview`, `detect`, `list-cameras`, and `diagnostics` commands. |
| Logitech USB preview | Ready for first hardware check | OpenCV capture supports Windows backends and configurable camera index/path. Run `python -m mracing_camera preview --source usb --camera 0`. |
| Image/video sources | Implemented | Image EOF and video EOF are handled; looping is opt-in. |
| Shared preview pipeline | Implemented, not hardware-tested | Window and browser MJPEG previews, status, snapshots, optional annotated-video recording, and a bounded latest-frame handoff. |
| Basler acquisition | Implemented, not hardware-tested | Optional pypylon import, device listing/serial selection, BGR conversion, and guarded feature setting. Needs compatible pylon runtime and confirmed camera/interface. |
| Model inference | Adapter implemented; actual checkpoint unverified | Ultralytics adapter reports metadata class names and returns original-frame boxes. The checkpoint's provenance is not established, so this adapter may not load it. |
| Automated tests | Verified | Six hardware-independent unit tests pass using the configured Python 3.13.5 environment. |
| Orin modes | Not hardware-tested | JetPack/Ubuntu/Python/CUDA details and matching packages must be confirmed on the board. |

### Current blockers and next actions

1. **Try Windows Logitech preview.** Install the package using the instructions below, connect the camera, run the preview command, and confirm live frames, snapshot save, and clean `q` exit. No physical camera test has been performed yet.
2. **Resolve checkpoint compatibility.** `models/best.pt` is not present in this checkout. Supply the team's checkpoint and its original training/export library, version, and inference command. Do not infer compatibility from `.pt` or the reported “YOLO v27 medium” name.
3. **Run saved-image inference, then live CPU inference.** After the loader and checkpoint are confirmed, verify model classes and boxes on a sample image before measuring laptop capture/inference throughput.
4. **Verify remote and Basler modes later.** Record the Orin software stack first; confirm the full Basler suffix, interface, connection, and pylon runtime before selecting those dependencies.

**Development environment observed:** Windows 11, Python 3.13.5, NumPy 2.2.6, OpenCV 4.12.0.88, PyTorch 2.8.0; Ultralytics and pypylon are not installed. CUDA was unavailable in the local environment. These are development-machine observations, not a tested environment lock.

## Initial coding capabilities

| Capability | Where the software runs | Camera connection | Intended output |
| --- | --- | --- | --- |
| Laptop development | Laptop | Logitech USB webcam plugged into the laptop | Live camera preview and YOLO cone-detection overlay; also evaluate the model on saved images/video. |
| Orin with Logitech | Jetson Orin Nano Super, accessed from the laptop through SSH / VS Code Remote SSH | Logitech USB webcam plugged into the Orin | Capture and inference on the Orin, with a live preview accessible from the laptop. |
| Orin with Basler | Jetson Orin Nano Super, accessed from the laptop through SSH / VS Code Remote SSH | Basler camera connected to the Orin through its confirmed interface | Basler frame acquisition and YOLO inference on the Orin, with a live preview accessible from the laptop. |

Keep camera acquisition separate from model inference and visualization so these modes can share the same downstream pipeline. Each host will need compatible dependencies; moving to the Orin does not imply copying the laptop's Python environment unchanged. Use a browser preview forwarded through SSH for the initial remote modes, as specified below; SSH alone does not display live video.

For each capability, first verify stable camera frames, then add model inference and detection overlays. Record achieved capture FPS, inference throughput, and end-to-end delay separately. The original Jetson Nano is not a required development target.

## Next goal after live perception

Use cone observations to estimate the track layout and generate a candidate path through it. Localization estimates the vehicle's position and orientation relative to that layout; path planning uses the layout and vehicle state to choose a route. These are related but separate tasks, so tracing an ideal path is broader than localization alone.

Before this stage, decide how image detections become cone positions and how vehicle motion is measured. Start with track boundaries and a candidate centerline; an optimized racing line is a later refinement. This next goal is exploratory and does not yet include vehicle actuation or a complete autonomous-driving capability.

## Current scope

- Basler a2A1920 family for initial tests; full suffix/interface pending.
- Previously linked Arducam B0353 is the intended camera reference; confirm SKU and Jetson compatibility.
- Capture target: 1920 × 1080 at 30 FPS; achieved performance must be measured.
- Development hosts: laptop for the first capability, then Jetson Orin Nano Super for the two remote capabilities.
- Laptop baseline: Windows, Intel Core Ultra 7, CPU inference. Use the latest stable Python release supported by the selected inference/camera packages; verify compatibility before pinning dependencies.
- Team-reported pylon Viewer: 26.08. Python capture will use a compatible pypylon version; Viewer can remain closed.
- Team-reported model: YOLO v27 medium. Supplied PyTorch checkpoint: best.pt. Confirm the compatible inference package and checkpoint metadata before loading.
- Immediate work is live camera capture and model inference. Track estimation, localization and path generation follow afterward; LiDAR, fusion and vehicle control are deferred.

## Dataset and weights selection

The preferred dataset candidate for now is [UTSMA FSAE cones](https://universe.roboflow.com/utsma/fsae-cones-dataset). Its project page lists 9,521 images, four dataset versions and six classes: `blue_cone`, `yellow_cone`, `orange_cone`, `large_orange_cone`, `unknown_cone`, and `objects`. It lists CC BY 4.0; preserve attribution when using it. Select a specific version and inspect its labels/splits before training. Project image count is not the same as a version's exported count, which may include augmentations.

The [University of Michigan cones dataset v2](https://universe.roboflow.com/university-of-michigan-znckk/cones-4b8zo/dataset/2) is a possible source for the existing checkpoint, but that relationship is unconfirmed.

Changing dataset configuration is straightforward. Changing the model requires compatible trained weights or training/fine-tuning on the selected dataset, plus verifying class mappings and preprocessing. More images alone do not establish better performance. The UTSMA page advertises a Roboflow 3.0 Object Detection (Fast) model; it is not verified as a downloadable checkpoint compatible with our reported YOLO model. Keep the supplied `best.pt` unchanged until its provenance/loading package is confirmed.

## Open in VS Code

Open this folder as the workspace, create a `.venv`, and select it as the VS Code interpreter. See [docs/SETUP.md](docs/SETUP.md) for Windows laptop and Jetson setup. The tested local environment is Python 3.13.5 with NumPy 2.2.6, GUI-capable OpenCV 4.12.0.88 and PyTorch 2.8.0; Ultralytics and pypylon are not installed. No camera or model hardware test has been performed.

## Structure

- `docs/SETUP.md`: environment setup, current verification status and hardware checks.
- `src/mracing_camera/`: installable package, source adapters, model adapter, preview and app.
- `configs/default.json`: sample defaults, overridable by command-line options.
- `models/`: local model weights; ignored by Git except the README.
- `data/`: local recordings/datasets; ignored by Git except the README.

## Run the application

Install the package and its preview dependencies in the selected environment:

```powershell
python -m pip install -e .
```

On Windows, start with Logitech preview; this command does not load model weights:

```powershell
python -m mracing_camera preview --source usb --camera 0
```

Press `q` to exit or `s` to save a snapshot under `runs/`. Requested capture defaults are 1920 × 1080 at 30 FPS; the application logs negotiated values because the camera may use different settings.

After confirming checkpoint provenance and installing its compatible inference dependencies, run:

```powershell
python -m mracing_camera detect --source usb --camera 0 --weights models/best.pt --device cpu
python -m mracing_camera detect --source image --input data/sample.jpg --weights models/best.pt --device cpu --preview none
```

`best.pt` is not present in this checkout. The model loader defaults to the explicitly integrated Ultralytics adapter but does not assume all PyTorch checkpoints use that format; unsupported checkpoints produce an actionable error. Inference is not verified until the actual checkpoint successfully runs on an image.

Other available commands:

```powershell
python -m mracing_camera diagnostics
python -m mracing_camera list-cameras --source basler
python -m mracing_camera detect --source video --input data/recording.mp4 --weights models/best.pt --preview window
python -m mracing_camera detect --source usb --camera 0 --weights models/best.pt --preview web --port 8080
```

Use `--preview window|web|none`, `--record runs/session.mp4`, `--snapshot runs/frame.jpg`, or `--loop` for a video source. A JSON config may be selected with `--config configs/default.json`; explicitly supplied CLI options override it. The full option list is available from `python -m mracing_camera preview --help` and `python -m mracing_camera detect --help`.

Web preview binds to `127.0.0.1` by default. Forward port 8080 from the Orin using VS Code Remote SSH or `ssh -L 8080:127.0.0.1:8080 USER@ORIN_HOST`, then open `http://127.0.0.1:8080` on the laptop. Placeholders are user supplied. Annotated video playback uses the configured FPS and is not a timing-accurate representation when processing FPS varies.

## Verification status

- Hardware-independent pipeline tests cover config precedence, missing weights, detection coordinates, latest-frame buffering, image EOF, and source cleanup.
- Local Python/OpenCV availability is verified, but a physical Logitech/Orin/Basler test has not been run.
- `models/best.pt` is absent, and its training/export library and package version are unknown. Inference must be re-tested with the team's actual checkpoint before use.
- Orin setup still depends on the board's JetPack/Ubuntu/Python/CUDA versions and matching packages. Basler needs pypylon plus the camera-compatible pylon runtime and confirmed hardware details.

Do not implement localization, path planning, LiDAR, ROS, Arducam, training or vehicle control in this stage.

The previously discussed Arducam remains a later hardware-integration consideration, outside these three initial capabilities.

## Project reference

[ALL INFO](https://docs.google.com/document/d/1r2eZJjpH60jpXqhf9EQnMrlORoWUxrvEbyuic_40h4A/edit)

The parent MRACING resources/presentations remain outside this source repository.

## Implementation brief for GitHub Copilot

This section is the requested specification for the first implementation. Read this README and `docs/SETUP.md`, inspect the repository, then implement the three initial capabilities. The next track/localization/path goal is context only: do not implement it in this first stage. Do not add LiDAR, ROS, Arducam support, training pipelines, cloud inference, vehicle commands or autonomous driving yet.

The initial application must acquire frames and run inference locally on the selected host. It must work in preview-only mode before model compatibility is resolved. It should be understandable to a student team: prefer a small Python application with clear modules over a complex framework.

### Architecture and frame handling

Create an installable Python package under `src/mracing_camera/`, with a CLI runnable as `python -m mracing_camera`. Provide separate modules for configuration, camera/file sources, the model adapter, annotation, preview serving and the application lifecycle.

- Source interface: open, read a frame, and close. Standardize on BGR image arrays plus a frame ID and host-side monotonic acquisition timestamp. Distinguish host timestamps from hardware exposure timestamps; do not imply exact sensor latency from a host timestamp alone.
- Logitech source: use OpenCV USB camera capture. Make the device index/path configurable; support Windows and Linux through appropriate backends, rather than assuming device index 0 always identifies the desired camera.
- Basler source: use the official pypylon API, support listing devices and selecting a serial number, and convert output to the same BGR representation. Import pypylon only when this source is selected. Check supported camera features before setting them, release grab results and close the camera on exit. Viewer must not need to be open.
- File sources: accept a saved image or video for repeatable model testing without hardware. End-of-file should exit normally unless looping is explicitly requested.
- Model adapter: accept a configurable weights path, device (`cpu`, `cuda`, or `auto`), confidence threshold and inference image size. Return detections in original-frame pixel coordinates with class ID, class name, confidence and bounding box. Use the selected model library's documented preprocessing/postprocessing; do not duplicate letterboxing or non-maximum suppression blindly.
- Overlay: show detections, current processing/capture rates, source resolution and inference duration. Label latency measurements honestly; host-side processing age is not physical camera-to-screen latency.

Use a bounded latest-frame handoff for live sources so slow CPU inference does not build an unbounded queue of old frames. Keep result overlays associated with the frame actually inferred; never draw stale boxes on an unrelated newer frame. Preview-only mode must bypass model loading entirely.

### Model compatibility: resolve, do not guess

`models/best.pt` is intentionally ignored by Git and is not present in the current checkout. A local copy must be supplied using the link in `models/README.md`. Preserve that checkpoint; do not overwrite it or download substitute weights silently.

“YOLO v27 medium” is an unverified team-reported identifier, not authority to invent an API or package. Inspect checkpoint metadata safely where possible, and determine the actual producing library and version. Do not assume every `.pt` file loads with Ultralytics. If the checkpoint is compatible with a documented loader, integrate it and record the exact package/version and class mapping. Otherwise leave preview fully usable, provide an actionable inference error and ask for the original training/export command or environment. Never report inference as working until a real image succeeds.

Read class names from verified model metadata; do not hard-code UTSMA class IDs into the existing checkpoint. The chosen dataset link does not establish the provenance of best.pt. Training or switching weights is a separate future task.

### Configuration and proposed CLI

The implemented CLI supports these commands:

```powershell
# Logitech preview on the Windows laptop
python -m mracing_camera preview --source usb --camera 0

# CPU detections from the laptop's Logitech camera
python -m mracing_camera detect --source usb --camera 0 --weights models/best.pt --device cpu

# Reproducible inference from a saved image
python -m mracing_camera detect --source image --input data/sample.jpg --weights models/best.pt --device cpu

# Discover Basler devices
python -m mracing_camera list-cameras --source basler
```

Support USB, Basler, image and video sources. Require a serial number if multiple Basler devices are connected and none has been explicitly selected. Provide a config file plus CLI overrides; CLI values take precedence. Document defaults, validation and every option.

Suggested defaults: requested camera width 1920, height 1080 and FPS 30; model confidence 0.25; inference size 640 if the verified model supports it. These are starting settings, not measured guarantees. Log the actual negotiated camera settings and report unsupported requests clearly. Use CPU by default for laptop examples; Orin `auto` may choose CUDA only after verifying availability and logging the selected device.

Support `--preview window|web|none`, optional annotated video recording, and a snapshot action. Create recording directories when needed, verify the video writer opened successfully, and document recording playback FPS so it does not imply accurate timing if processing FPS varies. Generated files must stay ignored by Git.

### Orin access and remote live preview

Implement a lightweight browser preview with an MJPEG endpoint and a basic page showing live annotated frames and status. Bind to `127.0.0.1` by default, with a configurable port (default 8080). Multiple viewers must share one capture/inference pipeline. A disconnected browser must not stall acquisition or cause a new camera instance per client.

Example intended workflow once implemented:

```bash
# On the Orin through VS Code Remote SSH or an SSH shell
python -m mracing_camera detect --source usb --camera 0 --weights models/best.pt --device auto --preview web --port 8080

# Basler alternative on the Orin
python -m mracing_camera detect --source basler --serial CAMERA_SERIAL --weights models/best.pt --device auto --preview web --port 8080
```

Forward remote port 8080 in VS Code, or run `ssh -L 8080:127.0.0.1:8080 USER@ORIN_HOST` from the laptop, then open `http://127.0.0.1:8080` on the laptop. USER, ORIN_HOST and CAMERA_SERIAL are user-supplied placeholders. Do not embed credentials or invent network addresses. SSH setup and real-camera testing require access to the actual board.

### Dependencies, setup and diagnostics

Choose the newest stable Python for which the full selected stack has compatible packages. Re-check current package support during implementation; the research notes in SETUP.md are not a dependency lock. Record the tested Python and package versions. Keep Basler support optional so laptop USB preview works without pypylon. Install a GUI-capable OpenCV package for desktop preview; avoid conflicting OpenCV distributions in one environment.

Provide Windows PowerShell setup/run instructions, `.venv`/VS Code interpreter selection, and separate Jetson Linux instructions. Determine the Orin's JetPack/Ubuntu/Python/CUDA versions before choosing its PyTorch installation; do not assume a generic desktop CUDA wheel works on Jetson. No flashing or system-wide upgrades are authorized by this brief.

Add a diagnostics command reporting Python/OS, installed package versions, model path existence, device availability and camera discovery without requiring all optional dependencies. Report useful errors for missing weights, incompatible checkpoints, unavailable CUDA, camera-open failure, timeout/disconnect and invalid configuration. On `q`/Ctrl+C or failure, release camera resources, stop workers, close windows/writers and stop the server. For this first version, fail clearly on disconnect rather than silently replaying stale frames.

### Validation and completion criteria

Add meaningful automated tests using fake frame sources/model results for configuration precedence, missing-model behavior, original-frame box coordinates, bounded frame handoff, end-of-file and cleanup. Tests must run without a physical camera, weights, Basler SDK or GPU. Use saved/generated test frames where appropriate; avoid committing private recordings or large artifacts.

Document manual checks separately:

1. Windows Logitech preview opens, displays current frames, saves a sample and exits cleanly.
2. Verified best.pt inference succeeds on a saved image, reports real model classes and overlays detections on the corresponding frame.
3. Windows Logitech CPU inference runs with measured performance; no 30 FPS inference requirement is assumed.
4. Orin Logitech capture/inference and browser preview work through an SSH tunnel.
5. Orin Basler discovery, capture/inference and browser preview work with the full confirmed camera model.

Only mark a mode hardware-verified after running it on that hardware. Implement and test what is possible locally; document the exact unresolved dependency or hardware check for the rest. Do not present mocks as real camera/model validation. Update the README and SETUP.md with actual executable commands, package versions and results when the code is implemented.

### Prompt to start implementation

> Read README.md and docs/SETUP.md. Implement the initial camera application following the Implementation brief for GitHub Copilot. Start with Windows Logitech preview, then verified model inference, and provide the shared architecture and optional Orin/Basler modes. Do not implement the later localization/path goal yet. Resolve model compatibility from evidence, keep preview working if it is unresolved, run hardware-independent tests, and document what still needs actual hardware verification.
