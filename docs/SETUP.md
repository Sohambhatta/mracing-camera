# Setup decisions and first test

## Confirm before installation

- Full Basler model suffix, color/mono variant, interface and required cable/power.
- First test host confirmed: Windows laptop, Intel Core Ultra 7, CPU inference, Logitech USB camera.
- Python version and compatible pypylon/runtime installation.
- On Jetson: carrier board, JetPack, Ubuntu and compatible ARM64 packages.
- Inference library/version used to produce the reported YOLO v27 medium checkpoint.

Do not infer checkpoint compatibility from the `.pt` extension alone.

## Python preference

Use the latest stable Python supported by the chosen model-loading and camera packages. The Python downloads page currently lists 3.14.8 as the latest stable release; the PyTorch Windows documentation lists support for Python 3.10–3.14. The full dependency combination still needs verification, especially the unknown YOLO package behind `best.pt`. Do not use a prerelease or assume all dependencies support the newest Python. No Python installation was discoverable as `python` or `py` in the current shell; this does not prove none is installed elsewhere.

## Dataset preference

UTSMA FSAE cones is the preferred candidate, with the exact version/export still to be selected. Inspect `objects` and `unknown_cone` annotations before deciding the training class set. Existing best.pt remains a separate model artifact; changing dataset URLs does not retrain or replace it.

## Acquisition approach

Basler → pypylon → image frames → preview → model → cone overlay.

VS Code runs/debugs the Python program. The program opens the camera and displays live output in its own window. pylon Viewer is optional during program capture; runtime/driver requirements remain interface-dependent. Basler recommends installing the pylon suite.

Arducam needs a separate compatible acquisition path. Reuse downstream preview/inference logic and recheck formats, exposure, timing and performance after switching cameras.

## Future environment setup

Once Python is selected, create a `.venv` in this folder and select it in VS Code. Record exact installed package versions after the first working hardware test. Candidate packages include pypylon and a GUI-capable OpenCV build; the model loader/PyTorch build must match the checkpoint and host.

No install or camera-run commands are claimed to work yet; no hardware test has been performed.

## References

- [Official pypylon](https://github.com/basler/pypylon)
- [pylon downloads](https://www.baslerweb.com/en-us/software/pylon/)
- [Basler Linux installation](https://docs.baslerweb.com/software-installation-%28linux%29)
- [Python releases](https://www.python.org/downloads/)
- [PyTorch installation/support](https://pytorch.org/get-started/locally/)
- [Preferred UTSMA dataset candidate](https://universe.roboflow.com/utsma/fsae-cones-dataset)
- [Possible existing model dataset source, unconfirmed](https://universe.roboflow.com/university-of-michigan-znckk/cones-4b8zo/dataset/2)
