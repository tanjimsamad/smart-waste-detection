# From Notebook to Script

`main.py` reproduces the experiment in [`notebooks/waste_detection.ipynb`](../notebooks/waste_detection.ipynb)
as reusable functions behind a command-line interface. The original notebook is kept unchanged as the
record of the reported results.

## Cell → function map

| Notebook cells | Purpose | `main.py` |
|---|---|---|
| 0–1 | Installs, imports, seeding, device | `requirements.txt`, `set_seed()`, `get_device()` |
| 3 | Project folders | `Config` dataclass + `Config.make_dirs()` |
| 5 | Roboflow download | `download_dataset()` → `download` |
| 7 | Read `data.yaml` | `load_class_names()` |
| 11, 13 | Split counts, annotated samples | `inspect_dataset()`, `plot_annotated_samples()` → `inspect` |
| 15, 17 | Object crops, crop distribution | `create_crops()` → `crops` |
| 19 | Transforms and loaders | `get_transforms()`, `get_cnn_loaders()` |
| 21, 60 | `WasteCNN` class | `build_wastecnn()` (one definition only) |
| 23, 25 | Training with early stopping, curves | `train_cnn()`, `plot_cnn_history()` → `train-cnn` |
| 27, 29, 45 | Test evaluation, confusion matrix | `cmd_eval_cnn()` → `eval-cnn` |
| 31 | CNN tuning | `cmd_tune_cnn()` → `tune-cnn` |
| 33–36 | YOLO baseline + loss curves | `train_yolo()`, `cmd_train_yolo()`, `plot_yolo_losses()` → `train-yolo` |
| 38 | YOLO tuning | `cmd_tune_yolo()` → `tune-yolo` |
| 40, 42 | YOLO test evaluation | `cmd_eval_yolo()` → `eval-yolo` |
| 43 | Sample predictions | `cmd_predict()` → `predict` |
| 47 | CNN vs YOLO comparison | `cmd_compare()` → `compare` |
| 53, 62 | Webcam | `CNNClassifier`, `annotate_frame()`, `cmd_webcam()` → `webcam` |
| 8–9, 50–59, 61 | Debugging (paths, OpenCV reinstall, checkpoint inspection) | Not needed; covered by the fixes below |

## Bugs and inconsistencies fixed

| # | In the notebook | Effect | In `main.py` |
|---|---|---|---|
| 1 | Cell 60 re-defined `WasteCNN` with 3 conv blocks and a 4 × 4 pool before the webcam cell | Webcam inference crashed: `Unexpected key(s) … features.12 … size mismatch for classifier.2.weight` | One `build_wastecnn()` used for training, evaluation and inference; old notebook checkpoints load with `strict=True` |
| 2 | Comparison cell 47 auto-picked `models/cnn_best.pt` | Compared the **baseline** CNN (72.5 %) instead of the tuned one (81.9 %) | Tuned checkpoint copied to `models/cnn_best_tuned.pt` and used by default |
| 3 | Cell 47 took YOLO metrics from the last **validation** epoch, CNN metrics from **test** | Apples-to-oranges comparison | Both read from test-split evaluations |
| 4 | Webcam cell used `IMG_SIZE` from `globals()` with a fallback of 128 | After a kernel restart the CNN would get 128 × 128 inputs instead of 224 × 224 | Image size stored inside the checkpoint |
| 5 | Checkpoints were raw `state_dict`s | Class names, image size and dropout had to be remembered by hand | Checkpoints are self-describing dicts; raw ones are still accepted |
| 6 | Hard-coded `G:\Code\smart-waste-detection` in several cells | Only ran on one machine | `--project-dir`, default = repository folder |
| 7 | Cell 8 overwrote `DATASET_PATH` with a non-existent folder | Later re-runs printed "folder not found" | Single `Config.dataset_dir` |
| 8 | API key typed at a prompt | Fine, but not scriptable | `ROBOFLOW_API_KEY` env var / `.env`, prompt as fallback |
| 9 | Baseline YOLO weights and tuned weights left in run folders | Hard to find the model to deploy | `models/yolo_baseline_best.pt`, `models/yolo_best.pt` |
| 10 | Whole-frame CNN webcam only | Unreliable with several objects or background | `--mode yolo` and `--mode hybrid` added |

## Behaviour kept identical

Seed 42, crop generation rules, 224 × 224 input, augmentations, ImageNet normalisation, WasteCNN
layers, AdamW + cross-entropy, early stopping on validation loss, the three CNN and three YOLO tuning
configurations, YOLO image size 640 and batch 16, and selection by validation loss (CNN) or validation
mAP@50-95 (YOLO).

One small difference: the notebook's `train_cnn` default weight decay was 1e-3; in `main.py` the
`train-cnn` default is the tuned value 1e-4. The `all` command still runs the baseline with 1e-3 so the
original experiment is reproduced.

## Not verified by execution

`main.py` was checked for syntax, imports and CLI wiring, and its `WasteCNN` was checked against the
notebook's parameter count (423,046) and `state_dict` keys. It has not yet been run end-to-end on a GPU;
please run `python main.py all` once before relying on it and open an issue if anything breaks.
