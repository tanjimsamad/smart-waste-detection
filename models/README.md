# models/

Weights are git-ignored because they are large binaries. `main.py` writes:

| File | Written by | What it is |
|---|---|---|
| `cnn_baseline.pt` | `main.py all` | Baseline CNN (dropout 0.27, wd 1e-3) |
| `cnn_tuning_{1,2,3}.pt` | `tune-cnn` | Best epoch of each CNN tuning config |
| `cnn_best_tuned.pt` | `tune-cnn` | Copy of the best tuning config (lowest val loss) |
| `cnn_best.pt` | `train-cnn` | Single-config CNN run |
| `yolo_baseline_best.pt` | `train-yolo` | YOLO11m baseline `best.pt` |
| `yolo_best.pt` | `tune-yolo` | Best YOLO11n tuning run `best.pt` |

## Sharing trained weights

Upload `yolo_best.pt` (≈5.5 MB) and `cnn_best_tuned.pt` (≈1.7 MB) as assets of a
**GitHub Release** (repo page → *Releases* → *Draft a new release*), then link them in the README.

## Checkpoints from the original notebook

The notebook saved raw `state_dict`s (no metadata). `main.py` still loads them:

```bash
python main.py eval-cnn --checkpoint models/cnn_tuning_1.pt
python main.py webcam --mode cnn --checkpoint models/cnn_tuning_1.pt
```

In the notebook run, `cnn_tuning_1.pt` was the best CNN (81.9 % test accuracy) and
`results/yolo_runs/tuning_2/weights/best.pt` was the best YOLO model.
