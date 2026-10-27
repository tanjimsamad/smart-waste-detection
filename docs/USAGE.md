# Usage Guide

## 1. Installation

Python 3.10 or newer. A CUDA GPU is strongly recommended for training (the reported runs used an
RTX 5060 Laptop GPU; YOLO11m took 3.2 h for 40 epochs). Inference runs fine on CPU.

```bash
git clone https://github.com/tanjimsamad/smart-waste-detection.git
cd smart-waste-detection

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# GPU users: install the matching CUDA build of PyTorch first, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**OpenCV and the webcam.** `ultralytics` can pull in `opencv-python-headless`, which has no GUI and
makes `cv2.imshow` fail. If the webcam window does not open:

```bash
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python
```

## 2. Roboflow API key

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Put your key in `.env` (`ROBOFLOW_API_KEY=...`). `.env` is git-ignored. If the variable is missing,
`main.py download` asks for the key interactively. Never commit the key.

## 3. Commands

```
python main.py [--project-dir DIR] [--seed 42] [--workers 0] <command> [options]
```

| Command | What it does | Main options | Writes |
|---|---|---|---|
| `download` | Download the Roboflow dataset (skipped if present) | — | `data/waste_dataset/` |
| `inspect` | Split sizes + 4 annotated samples | — | `outputs/dataset_annotated_samples.png` |
| `crops` | Build the CNN crop dataset | — | `data/cnn_crops/`, `outputs/crop_distribution.*` |
| `train-cnn` | Train one CNN config | `--lr 1e-3 --dropout 0.2 --weight-decay 1e-4 --epochs 50 --patience 8 --batch-size 32` | `models/cnn_best.pt`, `outputs/cnn_*` |
| `tune-cnn` | Grid of 3 CNN configs | `--epochs 50 --patience 8` | `models/cnn_tuning_{i}.pt`, `models/cnn_best_tuned.pt`, `outputs/cnn_hyperparameter_tuning.csv` |
| `eval-cnn` | Test-set accuracy, report, confusion matrix | `--checkpoint PATH` | `outputs/cnn_test_metrics.json`, `outputs/cnn_confusion_matrix.png` |
| `train-yolo` | YOLO baseline | `--model yolo11m.pt --epochs 40 --batch 16` | `outputs/yolo_runs/baseline/`, `models/yolo_baseline_best.pt` |
| `tune-yolo` | Grid of 3 YOLO configs | `--model yolo11n.pt --epochs 25` | `outputs/yolo_runs/tuning_{i}/`, `models/yolo_best.pt` |
| `eval-yolo` | Test-split P / R / mAP | `--weights PATH` | `outputs/yolo_test_metrics.json`, `outputs/yolo_evaluation/test/` |
| `compare` | CNN vs YOLO table and chart | — | `outputs/cnn_vs_yolo_metrics.csv`, `outputs/cnn_vs_yolo_comparison.png` |
| `predict` | Inference on an image or folder | `--source PATH --mode yolo/cnn/hybrid --weights --checkpoint --conf 0.25` | `outputs/predictions/` |
| `webcam` | Live inference, press **Q** to quit | `--mode yolo/cnn/hybrid --camera 0 --weights --checkpoint --conf 0.25` | — |
| `all` | download → inspect → crops → CNN baseline → tune → eval → YOLO baseline → tune → eval → compare | — | everything above |

Defaults for `--checkpoint` and `--weights` are picked automatically:
CNN `models/cnn_best_tuned.pt` → `cnn_tuning_1.pt` → `cnn_best.pt`;
YOLO `models/yolo_best.pt` → `yolo_baseline_best.pt`.

## 4. Common recipes

```bash
# Faster CPU-friendly check of the whole CNN path
python main.py download && python main.py crops
python main.py train-cnn --epochs 3

# Train YOLO11s instead of the defaults
python main.py train-yolo --model yolo11s.pt --epochs 60

# Use weights trained in the original notebook
python main.py eval-yolo --weights path/to/yolo_runs/tuning_2/weights/best.pt
python main.py webcam --mode hybrid \
    --weights path/to/yolo_runs/tuning_2/weights/best.pt \
    --checkpoint path/to/models/cnn_tuning_1.pt

# Keep data and outputs on another drive
python main.py --project-dir D:/waste-project all
```

## 5. Changing hyperparameter grids

The grids live in the `Config` dataclass at the top of [`main.py`](../main.py):

```python
cnn_tuning_grid = [
    {"lr": 1e-3, "dropout": 0.2, "weight_decay": 1e-4},
    ...
]
yolo_tuning_grid = [
    {"lr0": 1e-3, "weight_decay": 5e-4},
    ...
]
```

Add or remove dictionaries; `tune-cnn` / `tune-yolo` loop over them and keep the best automatically.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Not found: data/waste_dataset/data.yaml` | Run `python main.py download` first |
| `cv2.imshow` error / no window | Install the GUI build of OpenCV (see section 1) |
| `Could not open camera 0` | Try `--camera 1`; close other apps using the camera |
| CUDA out of memory with YOLO11m | `train-yolo --batch 8` or use `yolo11s.pt` |
| DataLoader hangs on Windows | Keep `--workers 0` (the default) |
| `Error(s) in loading state_dict` | The checkpoint is not a WasteCNN; check you passed a CNN file to `--checkpoint` |
