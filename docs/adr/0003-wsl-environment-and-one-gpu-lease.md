# The Environment goes into WSL when the GPU is reachable there, and GPU Jobs hold one GPU Lease

TensorFlow has had no GPU support on native Windows since 2.10. So when a WSL2
distribution exists and `nvidia-smi` works inside it, the Environment is
installed there automatically: uv brings its own Python 3.12, so no root or
apt is needed. Otherwise the Environment is native Windows, on the CPU, and the
setup screen explains why. CloudLabeller never installs WSL itself. A guided
button links to Microsoft's instructions instead, because enabling WSL changes
system settings and may need a reboot.

COLMAP and the CNNs must never use the GPU at the same time. A GPU Job may run
on either side of the Windows/WSL boundary, so the one **GPU Lease** is held by
the Tauri core, the only thing that starts Jobs on both sides. A second GPU
Job waits for the Lease and is shown as "waiting for GPU". Predicting a single
photo falls back to the CPU instead of queueing behind a long reconstruction.

## Considered Options

- **Refusing a second GPU Job**: the user has to retry by hand.
- **Silently falling back to the CPU for every GPU Job**: a reconstruction or
  training run could take ten times longer without the user noticing.
- **A lock inside Python**: it can't see a native `colmap.exe` and a WSL
  TensorFlow at the same time.
- **COLMAP native and TensorFlow in WSL**: two runtimes for every
  Project. Rejected, unless the spike below fails.

## Consequences

- In WSL, COLMAP runs from `pycolmap-cuda12` (BSD, Linux only, CUDA runtime
  from pip). pycolmap moves from 4.1 to 4.2. WSL has no OpenGL, and it is
  unverified whether COLMAP's GPU SIFT runs headless. A spike on real hardware
  decides this first. If SIFT fails there, only feature extraction falls back
  to the native CUDA `colmap.exe`. Matching and patch-match stereo stay in WSL.
- WSL COLMAP Jobs work in a **Workspace** on WSL's own disk. The Tauri core
  copies the inputs in from the Windows side, which also makes OneDrive
  download online-only files properly. The Job's products are copied back. This
  avoids `/mnt/c`, which Microsoft advises against for heavy I/O. Training and
  prediction read the Project straight through `/mnt/c`, because they read
  each image only once per epoch, or once per batch when predicting.
- TensorFlow stays pinned at 2.19. Moving to a newer version would touch the
  locked U-Net, so it is a separate decision.
