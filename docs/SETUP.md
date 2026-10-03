# Setup and verification

## Windows laptop: Logitech preview

The local workspace environment used for development is Python 3.13.5, NumPy 2.2.6, `opencv-python` 4.12.0.88 and PyTorch 2.8.0 on Windows. `ultralytics` and `pypylon` are not installed. These package versions are observed in the development interpreter, not a complete dependency lock. Create a project-local environment rather than installing packages into the system interpreter:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If `py -3.13` is not available, install a stable Python version supported by the project's OpenCV wheel and the selected inference stack, then substitute that version. In VS Code use **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`.

Connect the Logitech camera to the Windows laptop and run:

```powershell
python -m mracing_camera preview --source usb --camera 0
```

Set `--camera` to the Windows camera index or a device path where supported. OpenCV's DirectShow and Media Foundation backends are attempted on Windows. Press `q` to close the preview; press `s` to save a frame under `runs/`. The camera can negotiate a different mode than the requested 1920 × 1080 at 30 FPS; compare the logged values with the actual preview.

## Model inference

The `models/best.pt` checkpoint is intentionally excluded from Git and is not present in the current checkout. The team reports YOLO v27 medium, but the actual model format, training/export package, version and class metadata have not been confirmed. Do not install or assume Ultralytics compatibility based only on a `.pt` extension.

Ask the checkpoint provider for the producing library/version, its original inference command, and environment/export details. Then install a compatible PyTorch/model-loader combination in the `.venv`. The optional `inference` extra is available as `python -m pip install -e ".[inference]"`, but it installs general PyTorch and Ultralytics packages and is not a compatibility guarantee for this checkpoint.

Once the package has been confirmed, test a saved frame before live inference:

```powershell
python -m mracing_camera detect --source image --input data/sample.jpg --weights models/best.pt --device cpu --preview none
python -m mracing_camera detect --source usb --camera 0 --weights models/best.pt --device cpu
```

Class names are read from model metadata. Inference results use the model library's original-frame box coordinates; no project-specific class IDs are hard-coded. The reported host frame age is a host-side timestamp delta, not sensor exposure-to-display latency.

## Configuration and runtime options

The example `configs/default.json` sets the sample defaults. Supply it with `--config configs/default.json`; explicit CLI options win. Options include:

| Option | Default | Purpose |
| --- | --- | --- |
| `--source` | `usb` | `usb`, `basler`, `image` or `video` |
| `--camera` | `0` | USB index or device path |
| `--serial` | unset | Basler serial when selecting a Basler camera |
| `--input` | unset | Input image/video path |
| `--weights` | `models/best.pt` | Local model checkpoint |
| `--device` | `cpu` | `cpu`, `cuda` or `auto` |
| `--confidence` | `0.25` | Detection confidence threshold |
| `--imgsz` | `640` | Requested model inference size |
| `--width`, `--height`, `--fps` | `1920`, `1080`, `30` | Requested capture mode |
| `--preview` | `window` | `window`, `web` or `none` |
| `--host`, `--port` | `127.0.0.1`, `8080` | Web preview bind address and port |
| `--record` | unset | Annotated video output path |
| `--snapshot` | unset | Save the first processed frame to a file |
| `--loop` | false | Repeat a video after end-of-file |
| `--model-backend` | `ultralytics` | Currently integrated inference adapter |

Use `--no-loop` to override a JSON `loop` value. A Logitech failure to supply another frame is reported as a disconnect/error; the preview never silently redraws a stale frame. A Basler selection requires `--serial` if multiple devices are present.

Generated output lives in `runs/` or the user-selected path; local recordings, snapshots, datasets, and model weights are ignored by Git. Recorded video is written at the configured playback FPS; if processing is slower or variable, video timing does not reproduce real elapsed capture time.

## Diagnostics and Basler

```powershell
python -m mracing_camera diagnostics
python -m mracing_camera list-cameras --source basler
```

Diagnostics reports package/runtime versions, checkpoint presence, CUDA availability, USB indices that returned a frame, and Basler discovery where the optional package/runtime is available. Install pypylon separately with `python -m pip install -e ".[basler]"`; the Basler pylon runtime/driver must also match the camera/interface. pylon Viewer need not remain open while the application captures.

## Jetson Orin Nano Super (future hardware verification)

Use VS Code Remote SSH or an SSH shell on the board. First record the board's JetPack, Ubuntu, Python, CUDA and PyTorch versions. Install a PyTorch build compatible with that exact JetPack release using the NVIDIA/Jetson guidance; do not use a generic desktop CUDA wheel or copy the Windows environment. Install the application and its compatible OpenCV, model loader and (for Basler) pypylon/pylon runtime on the Orin.

Example once the Orin environment and checkpoint compatibility are established:

```bash
python -m mracing_camera detect --source usb --camera 0 --weights models/best.pt --device auto --preview web --port 8080
python -m mracing_camera detect --source basler --serial CAMERA_SERIAL --weights models/best.pt --device auto --preview web --port 8080
```

Forward port 8080 in VS Code, or run `ssh -L 8080:127.0.0.1:8080 USER@ORIN_HOST` from the laptop and browse to `http://127.0.0.1:8080`. `USER`, `ORIN_HOST`, and `CAMERA_SERIAL` are placeholders; no credentials or network addresses are embedded. No board setup or hardware test has been performed, and no system-wide upgrade or flashing is part of this setup.

## Automated and manual checks

Run hardware-independent unit tests with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The tests do not require a camera, checkpoint, Basler runtime or GPU. They do use the project's NumPy/OpenCV dependencies.

Still to verify on hardware:

1. Windows Logitech preview, snapshot and clean exit.
2. The actual `best.pt` loading and saved-image inference, with its metadata classes.
3. Windows Logitech CPU detections and measured capture/inference/end-to-end host timing.
4. Orin Logitech acquisition/inference through the browser preview and SSH tunnel.
5. Basler enumeration/acquisition on the confirmed full camera model/interface.

Only mark any of these as hardware-verified after running them on that hardware.
