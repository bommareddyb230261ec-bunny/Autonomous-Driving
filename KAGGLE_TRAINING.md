# Kaggle Diffusion Policy Training And Test Inference

This workflow keeps local Windows work limited to code editing and Git. Training happens on Kaggle GPU using nuScenes `v1.0-trainval`. Final inference uses the official annotation-free nuScenes `v1.0-test` split and does not compute supervised loss.

Expected nuScenes dataroot structure:

```text
/kaggle/input/<TRAINVAL_DATASET>/
  samples/
  sweeps/
  maps/
  v1.0-trainval/

/kaggle/input/<TEST_DATASET>/
  samples/
  sweeps/
  maps/
  v1.0-test/
```

If Kaggle mounts either dataset differently, the scripts print the detected structure and fail clearly.

## Cell 1: Clone Repository

```bash
git clone https://github.com/bommareddyb230261ec-bunny/Autonomous-Driving.git
```

## Cell 2: Enter Repository

```bash
cd Autonomous-Driving
```

## Cell 3: Install Missing Requirements Without Replacing PyTorch

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

If package names are printed:

```bash
pip install diffusers nuscenes-devkit matplotlib "numpy>=1.26,<3"
```

Do not install a CPU PyTorch wheel on Kaggle. Keep Kaggle's CUDA PyTorch unless it is broken.

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

## Cell 5: Locate Datasets

```bash
find /kaggle/input -maxdepth 3 -type d | sort | head -200
```

Choose one dataroot containing `v1.0-trainval` and one dataroot containing `v1.0-test`.

## Cell 6: Run Sanity Test

```bash
python kaggle_train.py \
  --train-dataroot /kaggle/input/<TRAINVAL_DATASET> \
  --train-version v1.0-trainval \
  --output-dir /kaggle/working/diffusion_outputs \
  --sanity-test \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --num-workers 2
```

Expected terminal lines include:

```text
Dataset loaded successfully
Condition shape: ...
Target shape: ...
Predicted noise shape: ...
Loss: ...
CUDA: ...
Sanity test PASSED
```

## Cell 7: Run Full Training

```bash
python kaggle_train.py \
  --train-dataroot /kaggle/input/<TRAINVAL_DATASET> \
  --train-version v1.0-trainval \
  --output-dir /kaggle/working/diffusion_outputs \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 4 \
  --num-workers 2
```

The script fits normalization only on the train split from `v1.0-trainval`, validates scene-disjoint train/validation splits, saves `latest.pt` every epoch, saves `best.pt` only when validation improves, and automatically exports a ZIP after training completes.

## Cell 8: Inspect Training Outputs

```bash
ls -lh /kaggle/working/diffusion_outputs
ls -lh /kaggle/working/diffusion_outputs/checkpoints
ls -lh /kaggle/working/diffusion_outputs/results
```

Expected files:

```text
/kaggle/working/diffusion_outputs/checkpoints/best.pt
/kaggle/working/diffusion_outputs/checkpoints/latest.pt
/kaggle/working/diffusion_outputs/normalization_stats.json
/kaggle/working/diffusion_outputs/training_history.json
/kaggle/working/diffusion_outputs/loss_curve.png
/kaggle/working/diffusion_outputs/results/training_history.json
/kaggle/working/diffusion_outputs/results/loss_curve.png
/kaggle/working/diffusion_outputs/diffusion_policy_checkpoint.zip
```

## Cell 9: Resume Training

```bash
python kaggle_train.py \
  --train-dataroot /kaggle/input/<TRAINVAL_DATASET> \
  --train-version v1.0-trainval \
  --output-dir /kaggle/working/diffusion_outputs \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 4 \
  --num-workers 2 \
  --resume /kaggle/working/diffusion_outputs/checkpoints/latest.pt
```

## Cell 10: Run Official Test Inference

```bash
python kaggle_test.py \
  --test-dataroot /kaggle/input/<TEST_DATASET> \
  --test-version v1.0-test \
  --checkpoint /kaggle/working/diffusion_outputs/checkpoints/best.pt \
  --output-dir /kaggle/working/diffusion_test_outputs
```

This does not train, does not use labels, and does not compute supervised loss. It saves speed/curvature predictions and an integrated trajectory estimate for each valid test sample.

Expected test outputs:

```text
/kaggle/working/diffusion_test_outputs/test_predictions.json
/kaggle/working/diffusion_test_outputs/predictions/test_predictions.json
/kaggle/working/diffusion_test_outputs/test_summary.json
```

## Cell 11: Export Checkpoint ZIP Manually

```bash
python export_checkpoint.py \
  --checkpoint /kaggle/working/diffusion_outputs/checkpoints/best.pt \
  --output /kaggle/working/diffusion_outputs/diffusion_policy_checkpoint.zip
```

Files under `/kaggle/working/` can be downloaded from the Kaggle notebook Output/File interface.
