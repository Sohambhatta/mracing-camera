# Setup decisions and first test

## Confirm before installation

- Full Basler model suffix, color/mono variant, interface and required cable/power.
- First test host: Windows laptop or Jetson.
- Python version and compatible pypylon/runtime installation.
- On Jetson: carrier board, JetPack, Ubuntu and compatible ARM64 packages.
- Inference library/version used to produce the reported YOLO v27 medium checkpoint.

Do not infer checkpoint compatibility from the `.pt` extension alone.

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
