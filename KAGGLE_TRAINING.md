# Kaggle Diffusion Policy Training

This workflow keeps local Windows work limited to code editing and Git. Training happens in a Kaggle GPU notebook, and outputs are written under `/kaggle/working/diffusion_outputs`.

## Cell 1: Clone Repository

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY>.git
```

## Cell 2: Enter Repository

```bash
cd <REPOSITORY>
```

## Cell 3: Install Missing Requirements Without Replacing PyTorch

First check what is missing:

```bash
python - <<'PY'
import importlib.util
missing = []
for package, module in [
    ("diffusers", "diffusers"),
    ("nuscenes-devkit", "nuscenes"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
]:
    if importlib.util.find_spec(module) is None:
        missing.append(package)
print(" ".join(missing))
PY
```

If the previous command prints package names, install only those missing dependencies:

```bash
pip install diffusers nuscenes-devkit matplotlib "numpy>=1.26,<3"
```

Kaggle usually already has a CUDA-compatible PyTorch build. Do not replace it unless it is broken, and avoid `pip install -r requirements.txt` if it would replace Kaggle's working PyTorch/CUDA environment.

## Cell 4: Verify GPU

```bash
python - <<'PY'
import sys
import torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## Cell 5: Locate nuScenes Dataset

```bash
find /kaggle/input -maxdepth 3 -type d | head -100
```

Choose the folder that contains the nuScenes files. The training command accepts this path through `--dataroot`.

## Cell 6: Run Sanity Test

```bash
python kaggle_train.py \
  --dataroot /kaggle/input/<YOUR_NUSCENES_DATASET> \
  --version v1.0-mini \
  --output-dir /kaggle/working/diffusion_outputs \
  --sanity-test \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --num-workers 2
```

This runs a tiny optimization pass to verify dataset loading, normalization, conditioning, diffusion noise scheduling, forward pass, backward pass, and checkpoint writing.

## Cell 7: Run Full Training

```bash
python kaggle_train.py \
  --dataroot /kaggle/input/<YOUR_NUSCENES_DATASET> \
  --version v1.0-mini \
  --output-dir /kaggle/working/diffusion_outputs \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 4 \
  --num-workers 2
```

## Cell 8: Inspect Checkpoints

```bash
ls -lh /kaggle/working/diffusion_outputs
ls -lh /kaggle/working/diffusion_outputs/checkpoints
ls -lh /kaggle/working/diffusion_outputs/results
```

Expected checkpoint paths:

```text
/kaggle/working/diffusion_outputs/checkpoints/best.pt
/kaggle/working/diffusion_outputs/checkpoints/latest.pt
```

## Cell 9: Resume Training

```bash
python kaggle_train.py \
  --dataroot /kaggle/input/<YOUR_NUSCENES_DATASET> \
  --version v1.0-mini \
  --output-dir /kaggle/working/diffusion_outputs \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 4 \
  --num-workers 2 \
  --resume /kaggle/working/diffusion_outputs/checkpoints/latest.pt
```

## Cell 10: Export ZIP

```bash
python export_checkpoint.py \
  --output-dir /kaggle/working/diffusion_outputs
```

Expected ZIP:

```text
/kaggle/working/diffusion_outputs/diffusion_policy_checkpoint.zip
```

Files under `/kaggle/working/` can be downloaded from the Kaggle notebook Output/File interface.
