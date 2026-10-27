#!/usr/bin/env python3
"""
Smart Waste Detection — CNN vs YOLO11
=====================================

Single entry point for the whole project, converted from the original
`waste_detection.ipynb` notebook into reusable functions and a command-line
interface.

Pipeline
--------
1. download      Download the Roboflow dataset (YOLOv11 format).
2. crops         Cut every ground-truth box into an ImageFolder dataset for the CNN.
3. train-cnn     Train the from-scratch WasteCNN with early stopping.
4. tune-cnn      Grid of CNN hyperparameters; keeps the best checkpoint.
5. eval-cnn      Test-set accuracy, classification report and confusion matrix.
6. train-yolo    Fine-tune a COCO-pretrained YOLO11 model (baseline).
7. tune-yolo     Grid of YOLO hyperparameters; keeps the best run.
8. eval-yolo     Test-set precision, recall, mAP@50 and mAP@50-95.
9. compare       CNN vs YOLO comparison table and chart (both on the test split).
10. predict      Run YOLO (optionally + CNN) on an image or folder.
11. webcam       Real-time inference: --mode yolo | cnn | hybrid.
    all          Runs steps 1-9 in order.

Examples
--------
    python main.py all
    python main.py tune-cnn --epochs 50
    python main.py eval-yolo --weights models/yolo_best.pt
    python main.py predict --source path/to/image.jpg
    python main.py webcam --mode hybrid

Author: Tanjim Samad Swapnil
License: MIT
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Matplotlib must not try to open windows when running headless (e.g. on a server).
import matplotlib

if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOFLOW_WORKSPACE = "object-detection-cfmul"
ROBOFLOW_PROJECT = "waste-detection-0momv"
ROBOFLOW_VERSION = 1

# ImageFolder sorts class folders alphabetically (upper case before lower case),
# which gives exactly this order. Used as a fallback when loading old checkpoints.
DEFAULT_CLASS_NAMES = ["Garbage_Bag", "Glass", "Paper_Bag", "Pet_Bottle", "Plastic_Bag", "can"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All paths and hyperparameters in one place (defaults = notebook values)."""

    project_dir: Path = Path(__file__).resolve().parent
    seed: int = 42
    num_workers: int = 0          # 0 is the safe choice on Windows

    # CNN
    img_size: int = 224
    batch_size: int = 32
    cnn_epochs: int = 50
    cnn_patience: int = 8
    cnn_lr: float = 1e-3
    cnn_dropout: float = 0.2
    cnn_weight_decay: float = 1e-4
    cnn_tuning_grid: list = field(default_factory=lambda: [
        {"lr": 1e-3, "dropout": 0.2, "weight_decay": 1e-4},
        {"lr": 1e-3, "dropout": 0.5, "weight_decay": 1e-4},
        {"lr": 3e-4, "dropout": 0.4, "weight_decay": 1e-4},
    ])

    # YOLO
    yolo_model: str = "yolo11m.pt"
    yolo_tuning_model: str = "yolo11n.pt"
    yolo_epochs: int = 40
    yolo_tuning_epochs: int = 25
    yolo_img_size: int = 640
    yolo_batch: int = 16
    yolo_patience: int = 8
    yolo_tuning_patience: int = 5
    yolo_lr0: float = 1e-3
    yolo_weight_decay: float = 1e-4
    yolo_tuning_grid: list = field(default_factory=lambda: [
        {"lr0": 1e-3, "weight_decay": 5e-4},
        {"lr0": 3e-4, "weight_decay": 5e-4},
        {"lr0": 1e-3, "weight_decay": 1e-3},
    ])
    conf_threshold: float = 0.25

    # Derived paths ---------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "waste_dataset"

    @property
    def data_yaml(self) -> Path:
        return self.dataset_dir / "data.yaml"

    @property
    def crop_dir(self) -> Path:
        return self.data_dir / "cnn_crops"

    @property
    def models_dir(self) -> Path:
        return self.project_dir / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.project_dir / "outputs"

    @property
    def yolo_runs_dir(self) -> Path:
        return self.outputs_dir / "yolo_runs"

    def make_dirs(self) -> None:
        for folder in (self.data_dir, self.models_dir, self.outputs_dir):
            folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    msg = f"Device: {device}"
    if device.type == "cuda":
        msg += f" ({torch.cuda.get_device_name(0)})"
    print(msg)
    return device


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print("Saved:", path)


def load_class_names(cfg: Config) -> list[str]:
    """Read class names from data.yaml (list or {id: name} dict)."""
    import yaml

    if not cfg.data_yaml.exists():
        return list(DEFAULT_CLASS_NAMES)
    with open(cfg.data_yaml, "r", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def require(path: Path, hint: str) -> Path:
    if not path.exists():
        sys.exit(f"[error] Not found: {path}\n        {hint}")
    return path


# ---------------------------------------------------------------------------
# 1. Dataset download and inspection
# ---------------------------------------------------------------------------

def download_dataset(cfg: Config) -> Path:
    """Download the Roboflow dataset in YOLOv11 format.

    The API key is read from the ROBOFLOW_API_KEY environment variable
    (or a .env file); if absent you are prompted for it. It is never stored.
    """
    if cfg.data_yaml.exists():
        print("Dataset already present:", cfg.dataset_dir)
        return cfg.dataset_dir

    try:
        from dotenv import load_dotenv

        load_dotenv(cfg.project_dir / ".env")
    except ImportError:
        pass

    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        from getpass import getpass

        api_key = getpass("Enter your Roboflow API key: ")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov11", location=str(cfg.dataset_dir))
    print("Dataset downloaded to:", dataset.location)
    return Path(dataset.location)


def inspect_dataset(cfg: Config) -> pd.DataFrame:
    """Print image / label-file counts per split."""
    rows = []
    for split in SPLITS:
        images = list_images(cfg.dataset_dir / split / "images")
        labels = list((cfg.dataset_dir / split / "labels").glob("*.txt"))
        rows.append({"split": split, "images": len(images), "label_files": len(labels)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def plot_annotated_samples(cfg: Config, n: int = 4) -> Path:
    """Draw ground-truth boxes on a few random training images."""
    import matplotlib.patches as patches
    from PIL import Image

    class_names = load_class_names(cfg)
    image_paths = list_images(cfg.dataset_dir / "train" / "images")
    samples = random.sample(image_paths, min(n, len(image_paths)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, image_path in zip(axes.flatten(), samples):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        ax.imshow(image)
        ax.set_title(image_path.name, fontsize=8)
        ax.axis("off")
        label_path = cfg.dataset_dir / "train" / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            cls_id, x, y, bw, bh = map(float, values[:5])
            x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
            ax.add_patch(patches.Rectangle((x1, y1), bw * w, bh * h,
                                           linewidth=2, edgecolor="lime", facecolor="none"))
            ax.text(x1, max(0, y1 - 5), class_names[int(cls_id)],
                    color="black", backgroundcolor="lime", fontsize=8)
    plt.tight_layout()
    out = cfg.outputs_dir / "dataset_annotated_samples.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)
    return out


# ---------------------------------------------------------------------------
# 2. Object crops for the CNN
# ---------------------------------------------------------------------------

def create_crops(cfg: Config) -> pd.DataFrame:
    """Crop every YOLO box into data/cnn_crops/<split>/<class>/<image>_<i>.jpg."""
    import cv2
    from tqdm.auto import tqdm

    require(cfg.data_yaml, "Run `python main.py download` first.")
    class_names = load_class_names(cfg)

    if cfg.crop_dir.exists():
        shutil.rmtree(cfg.crop_dir)

    counts: dict[tuple[str, str], int] = {}
    for split in SPLITS:
        image_dir = cfg.dataset_dir / split / "images"
        label_dir = cfg.dataset_dir / split / "labels"
        if not image_dir.exists():
            continue
        for name in class_names:
            (cfg.crop_dir / split / name).mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(list_images(image_dir), desc=f"Cropping {split}"):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]

            for idx, line in enumerate(label_path.read_text().splitlines()):
                values = line.split()
                if len(values) < 5:
                    continue
                cls_id, x, y, bw, bh = map(float, values[:5])
                cls_id = int(cls_id)
                if not 0 <= cls_id < len(class_names):
                    continue
                x1, y1 = max(0, int((x - bw / 2) * w)), max(0, int((y - bh / 2) * h))
                x2, y2 = min(w, int((x + bw / 2) * w)), min(h, int((y + bh / 2) * h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                name = class_names[cls_id]
                cv2.imwrite(str(cfg.crop_dir / split / name / f"{image_path.stem}_{idx}.jpg"), crop)
                counts[(split, name)] = counts.get((split, name), 0) + 1

    table = pd.Series(counts).unstack(fill_value=0)
    print("Object crops created:\n", table)

    table.T.plot(kind="bar", figsize=(12, 5))
    plt.title("Object crop distribution")
    plt.ylabel("Number of crops")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(cfg.outputs_dir / "crop_distribution.png", dpi=150)
    plt.close()
    table.to_csv(cfg.outputs_dir / "crop_distribution.csv")
    return table


# ---------------------------------------------------------------------------
# 3. CNN model, data and training
# ---------------------------------------------------------------------------

def build_wastecnn(num_classes: int, dropout: float = 0.2):
    """Return the WasteCNN used for every result in this repository.

    4 conv blocks (32-64-128-256 filters, 3x3, BatchNorm, ReLU), max-pooling
    after the first three, global average pooling, and a 256-128-N MLP head
    with dropout. 423,046 trainable parameters for 6 classes.
    """
    import torch.nn as nn

    class WasteCNN(nn.Module):
        def __init__(self, num_classes: int, dropout: float = 0.2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return WasteCNN(num_classes, dropout)


def get_transforms(img_size: int):
    from torchvision import transforms

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_cnn_loaders(cfg: Config):
    from torch.utils.data import DataLoader
    from torchvision import datasets

    require(cfg.crop_dir / "train", "Run `python main.py crops` first.")
    train_tf, eval_tf = get_transforms(cfg.img_size)
    ds = {
        "train": datasets.ImageFolder(cfg.crop_dir / "train", transform=train_tf),
        "valid": datasets.ImageFolder(cfg.crop_dir / "valid", transform=eval_tf),
        "test": datasets.ImageFolder(cfg.crop_dir / "test", transform=eval_tf),
    }
    loaders = {
        split: DataLoader(d, batch_size=cfg.batch_size, shuffle=(split == "train"),
                          num_workers=cfg.num_workers)
        for split, d in ds.items()
    }
    print("CNN classes:", ds["train"].classes)
    print({k: len(v) for k, v in ds.items()}, "crops")
    return loaders, ds["train"].classes


def save_cnn_checkpoint(model, path: Path, class_names: list[str], img_size: int,
                        dropout: float, extra: dict | None = None) -> None:
    """Save weights *with* the metadata needed to rebuild the model later."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": img_size,
        "dropout": dropout,
        "architecture": "WasteCNN",
        **(extra or {}),
    }, path)


def load_cnn(path: Path, device, class_names: list[str] | None = None):
    """Load a WasteCNN checkpoint.

    Accepts both the self-describing checkpoints written by this script and the
    raw state_dicts saved by the original notebook (e.g. models/cnn_best.pt).
    Returns (model, class_names, img_size).
    """
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        class_names = ckpt.get("class_names", class_names)
        img_size = ckpt.get("img_size", 224)
        dropout = ckpt.get("dropout", 0.2)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state, img_size, dropout = ckpt["state_dict"], 224, 0.2
    else:  # raw state_dict from the notebook
        state, img_size, dropout = ckpt, 224, 0.2

    class_names = class_names or list(DEFAULT_CLASS_NAMES)
    model = build_wastecnn(len(class_names), dropout).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, class_names, img_size


def train_cnn(model, loaders, device, epochs: int, lr: float, weight_decay: float,
              patience: int, save_fn) -> tuple[object, pd.DataFrame]:
    """Train with AdamW + cross-entropy; early-stop on validation loss.

    `save_fn(model)` is called every time validation loss improves.
    The best weights are restored before returning.
    """
    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val, best_weights, bad_epochs = float("inf"), copy.deepcopy(model.state_dict()), 0

    def run_epoch(loader, train: bool):
        model.train(train)
        loss_sum, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for images, labels in tqdm(loader, leave=False, disable=not train):
                images, labels = images.to(device), labels.to(device)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return loss_sum / total, correct / total

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(loaders["train"], train=True)
        va_loss, va_acc = run_epoch(loaders["valid"], train=False)
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)
        print(f"Epoch {epoch:02d} | Train loss: {tr_loss:.4f} | Val loss: {va_loss:.4f} | "
              f"Train acc: {tr_acc:.3f} | Val acc: {va_acc:.3f}")

        if va_loss < best_val:
            best_val, bad_epochs = va_loss, 0
            best_weights = copy.deepcopy(model.state_dict())
            save_fn(model)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history)


def plot_cnn_history(history: pd.DataFrame, out: Path, title: str = "CNN") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(history) + 1)
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(epochs, history[f"train_{metric}"], label=f"Train {metric}")
        ax.plot(epochs, history[f"val_{metric}"], label=f"Validation {metric}")
        ax.set_title(f"{title} {metric}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)


def cmd_train_cnn(cfg: Config, lr=None, dropout=None, weight_decay=None,
                  save_path: Path | None = None, tag: str = "cnn") -> tuple[object, pd.DataFrame]:
    """Train a single CNN configuration."""
    set_seed(cfg.seed)
    device = get_device()
    loaders, class_names = get_cnn_loaders(cfg)
    lr = cfg.cnn_lr if lr is None else lr
    dropout = cfg.cnn_dropout if dropout is None else dropout
    weight_decay = cfg.cnn_weight_decay if weight_decay is None else weight_decay
    save_path = save_path or cfg.models_dir / "cnn_best.pt"

    model = build_wastecnn(len(class_names), dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"WasteCNN | trainable parameters: {n_params:,} | lr={lr} dropout={dropout} wd={weight_decay}")

    model, history = train_cnn(
        model, loaders, device, epochs=cfg.cnn_epochs, lr=lr, weight_decay=weight_decay,
        patience=cfg.cnn_patience,
        save_fn=lambda m: save_cnn_checkpoint(
            m, save_path, class_names, cfg.img_size, dropout,
            {"lr": lr, "weight_decay": weight_decay}),
    )
    history.to_csv(cfg.outputs_dir / f"{tag}_history.csv", index_label="epoch")
    plot_cnn_history(history, cfg.outputs_dir / f"{tag}_training_curves.png")
    return model, history


def cmd_tune_cnn(cfg: Config) -> pd.DataFrame:
    """Train every config in cfg.cnn_tuning_grid; copy the best to models/cnn_best_tuned.pt."""
    results = []
    for i, c in enumerate(cfg.cnn_tuning_grid, start=1):
        print(f"\n=== CNN configuration {i}: {c} ===")
        path = cfg.models_dir / f"cnn_tuning_{i}.pt"
        _, history = cmd_train_cnn(cfg, lr=c["lr"], dropout=c["dropout"],
                                   weight_decay=c["weight_decay"], save_path=path,
                                   tag=f"cnn_tuning_{i}")
        best = int(history["val_loss"].idxmin())
        results.append({**c, "config": i, "checkpoint": str(path),
                        "best_val_loss": history.loc[best, "val_loss"],
                        "best_val_accuracy": history.loc[best, "val_accuracy"],
                        "best_epoch": best + 1, "epochs_run": len(history)})

    df = pd.DataFrame(results).sort_values("best_val_loss")
    df.to_csv(cfg.outputs_dir / "cnn_hyperparameter_tuning.csv", index=False)
    print("\n", df.to_string(index=False))

    best_path = Path(df.iloc[0]["checkpoint"])
    shutil.copy(best_path, cfg.models_dir / "cnn_best_tuned.pt")
    print("Best CNN checkpoint →", cfg.models_dir / "cnn_best_tuned.pt")
    return df


def cmd_eval_cnn(cfg: Config, checkpoint: Path | None = None) -> dict:
    """Evaluate a CNN checkpoint on the test crops."""
    import seaborn as sns
    import torch
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 precision_recall_fscore_support)

    device = get_device()
    checkpoint = checkpoint or _default_cnn_checkpoint(cfg)
    require(checkpoint, "Train a CNN first (`train-cnn` or `tune-cnn`).")
    loaders, folder_classes = get_cnn_loaders(cfg)
    model, class_names, _ = load_cnn(checkpoint, device, folder_classes)
    print("Evaluating:", checkpoint)

    y_true, y_pred = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in loaders["test"]:
            y_pred.extend(model(images.to(device)).argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    elapsed = time.perf_counter() - start

    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   zero_division=0, digits=4)
    print(report)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {"checkpoint": str(checkpoint), "accuracy": accuracy_score(y_true, y_pred),
               "weighted_precision": p, "weighted_recall": r, "weighted_f1": f1,
               "num_test_crops": len(y_true), "eval_seconds": round(elapsed, 2)}
    save_json(metrics, cfg.outputs_dir / "cnn_test_metrics.json")
    (cfg.outputs_dir / "cnn_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("CNN test confusion matrix")
    plt.tight_layout()
    plt.savefig(cfg.outputs_dir / "cnn_confusion_matrix.png", dpi=150)
    plt.close()
    return metrics


def _default_cnn_checkpoint(cfg: Config) -> Path:
    for name in ("cnn_best_tuned.pt", "cnn_tuning_1.pt", "cnn_best.pt"):
        if (cfg.models_dir / name).exists():
            return cfg.models_dir / name
    return cfg.models_dir / "cnn_best_tuned.pt"


# ---------------------------------------------------------------------------
# 4. YOLO training, tuning and evaluation
# ---------------------------------------------------------------------------

def _yolo_history(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = df.columns.str.strip()
    return df


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    lookup = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def train_yolo(cfg: Config, model_name: str, run_name: str, epochs: int, patience: int,
               lr0: float, weight_decay: float) -> Path:
    from ultralytics import YOLO

    require(cfg.data_yaml, "Run `python main.py download` first.")
    model = YOLO(model_name)
    result = model.train(
        data=str(cfg.data_yaml), epochs=epochs, imgsz=cfg.yolo_img_size, batch=cfg.yolo_batch,
        patience=patience, optimizer="AdamW", lr0=lr0, weight_decay=weight_decay,
        workers=cfg.num_workers, seed=cfg.seed, deterministic=True, plots=True,
        project=str(cfg.yolo_runs_dir), name=run_name, exist_ok=True,
    )
    return Path(result.save_dir)


def plot_yolo_losses(run_dir: Path, out: Path) -> None:
    hist = _yolo_history(run_dir)
    cols = [c for c in hist.columns if "loss" in c.lower()]
    if not cols:
        return
    x = hist["epoch"] if "epoch" in hist else hist.index + 1
    plt.figure(figsize=(12, 5))
    for c in cols:
        plt.plot(x, hist[c], label=c)
    plt.title("YOLO training and validation losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)


def cmd_train_yolo(cfg: Config) -> Path:
    run_dir = train_yolo(cfg, cfg.yolo_model, "baseline", cfg.yolo_epochs, cfg.yolo_patience,
                         cfg.yolo_lr0, cfg.yolo_weight_decay)
    plot_yolo_losses(run_dir, cfg.outputs_dir / "yolo_baseline_loss_curves.png")
    shutil.copy(run_dir / "weights" / "best.pt", cfg.models_dir / "yolo_baseline_best.pt")
    print("Baseline YOLO weights →", cfg.models_dir / "yolo_baseline_best.pt")
    return run_dir


def cmd_tune_yolo(cfg: Config) -> pd.DataFrame:
    """Train each config in cfg.yolo_tuning_grid; pick the best by val mAP@50-95."""
    results = []
    for i, c in enumerate(cfg.yolo_tuning_grid, start=1):
        print(f"\n=== YOLO configuration {i}: {c} ===")
        run_dir = train_yolo(cfg, cfg.yolo_tuning_model, f"tuning_{i}", cfg.yolo_tuning_epochs,
                             cfg.yolo_tuning_patience, c["lr0"], c["weight_decay"])
        hist = _yolo_history(run_dir)
        col = _find_col(hist, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        if col is not None:
            best_idx = hist[col].idxmax()
            best_map, best_epoch = float(hist.loc[best_idx, col]), int(hist.loc[best_idx, "epoch"])
        else:
            best_map, best_epoch = float("nan"), -1
        results.append({**c, "config": i, "model": cfg.yolo_tuning_model,
                        "best_val_map50_95": best_map, "best_epoch": best_epoch,
                        "run_dir": str(run_dir)})

    df = pd.DataFrame(results).sort_values("best_val_map50_95", ascending=False)
    df.to_csv(cfg.outputs_dir / "yolo_hyperparameter_tuning.csv", index=False)
    print("\n", df.to_string(index=False))

    best_weights = Path(df.iloc[0]["run_dir"]) / "weights" / "best.pt"
    shutil.copy(best_weights, cfg.models_dir / "yolo_best.pt")
    print("Best YOLO weights →", cfg.models_dir / "yolo_best.pt")
    return df


def _default_yolo_weights(cfg: Config) -> Path:
    for name in ("yolo_best.pt", "yolo_baseline_best.pt"):
        if (cfg.models_dir / name).exists():
            return cfg.models_dir / name
    return cfg.models_dir / "yolo_best.pt"


def cmd_eval_yolo(cfg: Config, weights: Path | None = None) -> dict:
    """Evaluate YOLO weights on the held-out *test* split."""
    from ultralytics import YOLO

    weights = weights or _default_yolo_weights(cfg)
    require(weights, "Train YOLO first (`train-yolo` or `tune-yolo`).")
    print("Evaluating:", weights)
    metrics = YOLO(str(weights)).val(
        data=str(cfg.data_yaml), split="test", imgsz=cfg.yolo_img_size, batch=cfg.yolo_batch,
        workers=cfg.num_workers, plots=True, project=str(cfg.outputs_dir / "yolo_evaluation"),
        name="test", exist_ok=True,
    )
    p, r = float(metrics.box.mp), float(metrics.box.mr)
    results = {"weights": str(weights), "precision": p, "recall": r,
               "f1": 2 * p * r / (p + r) if p + r else 0.0,
               "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
               "per_class_mAP50_95": dict(zip(metrics.names.values(), map(float, metrics.box.maps)))}
    print(pd.Series({k: v for k, v in results.items() if k != "per_class_mAP50_95"}))
    save_json(results, cfg.outputs_dir / "yolo_test_metrics.json")
    return results


# ---------------------------------------------------------------------------
# 5. Comparison
# ---------------------------------------------------------------------------

def cmd_compare(cfg: Config) -> pd.DataFrame:
    """Compare the best CNN and the best YOLO on the test split.

    Both numbers come from the test split (the original notebook compared CNN test
    scores with YOLO's last *validation* epoch; that is fixed here).
    """
    cnn_path, yolo_path = cfg.outputs_dir / "cnn_test_metrics.json", cfg.outputs_dir / "yolo_test_metrics.json"
    cnn = json.loads(cnn_path.read_text()) if cnn_path.exists() else cmd_eval_cnn(cfg)
    yolo = json.loads(yolo_path.read_text()) if yolo_path.exists() else cmd_eval_yolo(cfg)

    metrics = ["Precision", "Recall", "F1-score", "mAP@50", "mAP@50-95"]
    cnn_vals = [cnn["weighted_precision"], cnn["weighted_recall"], cnn["weighted_f1"], np.nan, np.nan]
    yolo_vals = [yolo["precision"], yolo["recall"], yolo["f1"], yolo["mAP50"], yolo["mAP50_95"]]
    df = pd.DataFrame({"Metric": metrics, "CNN": cnn_vals, "YOLO": yolo_vals})
    df.to_csv(cfg.outputs_dir / "cnn_vs_yolo_metrics.csv", index=False)
    print(df.round(4).to_string(index=False))

    x, w = np.arange(len(metrics)), 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, vals, label in ((-w / 2, cnn_vals, "CNN (test crops)"), (w / 2, yolo_vals, "YOLO (test images)")):
        bars = ax.bar(x + offset, vals, w, label=label)
        for bar in bars:
            v = bar.get_height()
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("CNN vs YOLO performance (test split)")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.text(0.01, -0.13, "mAP is N/A for the CNN: it classifies pre-cropped objects and predicts no boxes.",
            transform=ax.transAxes, fontsize=9)
    plt.tight_layout()
    plt.savefig(cfg.outputs_dir / "cnn_vs_yolo_comparison.png", dpi=150)
    plt.close()
    return df


# ---------------------------------------------------------------------------
# 6. Inference (images, folders, webcam)
# ---------------------------------------------------------------------------

class CNNClassifier:
    """Thin wrapper that classifies BGR (OpenCV) image arrays with WasteCNN."""

    def __init__(self, checkpoint: Path, device):
        import torch
        from torchvision import transforms

        self.torch = torch
        self.device = device
        self.model, self.class_names, img_size = load_cnn(checkpoint, device)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, bgr: np.ndarray) -> tuple[str, float]:
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(dim=0)
        return self.class_names[idx.item()], conf.item()


def annotate_frame(frame: np.ndarray, mode: str, yolo=None, cnn: CNNClassifier | None = None,
                   conf: float = 0.25, imgsz: int = 640) -> np.ndarray:
    """Draw predictions on a BGR frame for mode 'yolo', 'cnn' or 'hybrid'."""
    import cv2

    if mode == "cnn":
        label, p = cnn(frame)
        cv2.putText(frame, f"{label}: {p * 100:.1f}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    result = yolo.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    if mode == "yolo":
        return result.plot()

    # hybrid: YOLO localises, the CNN re-classifies each crop
    h, w = frame.shape[:2]
    for box in result.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = np.clip(box, 0, [w, h, w, h])
        if x2 <= x1 or y2 <= y1:
            continue
        label, p = cnn(frame[y1:y2, x1:x2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{label} {p:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 0), 2, cv2.LINE_AA)
    return frame


def _load_models(cfg: Config, mode: str, weights: Path | None, checkpoint: Path | None):
    yolo = cnn = None
    if mode in ("yolo", "hybrid"):
        from ultralytics import YOLO

        weights = weights or _default_yolo_weights(cfg)
        yolo = YOLO(str(require(weights, "Train YOLO or pass --weights.")))
    if mode in ("cnn", "hybrid"):
        checkpoint = checkpoint or _default_cnn_checkpoint(cfg)
        cnn = CNNClassifier(require(checkpoint, "Train the CNN or pass --checkpoint."), get_device())
    return yolo, cnn


def cmd_predict(cfg: Config, source: Path, mode: str = "yolo", weights: Path | None = None,
                checkpoint: Path | None = None) -> None:
    """Run inference on one image or every image in a folder."""
    import cv2

    yolo, cnn = _load_models(cfg, mode, weights, checkpoint)
    paths = list_images(source) if source.is_dir() else [source]
    out_dir = cfg.outputs_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print("Skipping unreadable file:", path)
            continue
        annotated = annotate_frame(frame, mode, yolo, cnn, cfg.conf_threshold, cfg.yolo_img_size)
        out = out_dir / f"{path.stem}_{mode}.jpg"
        cv2.imwrite(str(out), annotated)
        print("Saved:", out)


def cmd_webcam(cfg: Config, mode: str = "yolo", camera: int = 0, weights: Path | None = None,
               checkpoint: Path | None = None) -> None:
    """Real-time inference from a webcam. Press Q to quit.

    Needs the GUI build of OpenCV (`opencv-python`, not `opencv-python-headless`).
    """
    import cv2

    yolo, cnn = _load_models(cfg, mode, weights, checkpoint)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit(f"[error] Could not open camera {camera}. Try --camera 1.")
    print(f"Webcam inference ({mode}). Press Q to quit.")
    window = f"Smart Waste Detection — {mode}"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame.")
                break
            t0 = time.perf_counter()
            frame = annotate_frame(frame, mode, yolo, cnn, cfg.conf_threshold, cfg.yolo_img_size)
            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            cv2.putText(frame, f"{fps:.1f} FPS", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 7. Full pipeline and CLI
# ---------------------------------------------------------------------------

def cmd_all(cfg: Config) -> None:
    download_dataset(cfg)
    inspect_dataset(cfg)
    plot_annotated_samples(cfg)
    create_crops(cfg)
    # Baseline run exactly as in the notebook (dropout 0.27, wd 1e-3, patience 5)
    patience, cfg.cnn_patience = cfg.cnn_patience, 5
    cmd_train_cnn(cfg, lr=1e-3, dropout=0.27, weight_decay=1e-3,
                  save_path=cfg.models_dir / "cnn_baseline.pt", tag="cnn_baseline")
    cfg.cnn_patience = patience
    cmd_tune_cnn(cfg)
    cmd_eval_cnn(cfg)
    cmd_train_yolo(cfg)
    cmd_tune_yolo(cfg)
    cmd_eval_yolo(cfg)
    cmd_compare(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smart waste detection: from-scratch CNN vs YOLO11 on a 6-class Roboflow dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-dir", type=Path, default=Config.project_dir,
                        help="Repository root (data/, models/ and outputs/ live here).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader / YOLO workers.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download the Roboflow dataset.")
    sub.add_parser("inspect", help="Print split sizes and save annotated samples.")
    sub.add_parser("crops", help="Create the CNN crop dataset from YOLO labels.")

    p = sub.add_parser("train-cnn", help="Train one CNN configuration.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)

    p = sub.add_parser("tune-cnn", help="Grid-search CNN hyperparameters.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)

    p = sub.add_parser("eval-cnn", help="Evaluate a CNN checkpoint on the test crops.")
    p.add_argument("--checkpoint", type=Path)

    p = sub.add_parser("train-yolo", help="Train the baseline YOLO model.")
    p.add_argument("--model", default="yolo11m.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)

    p = sub.add_parser("tune-yolo", help="Grid-search YOLO hyperparameters.")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=25)

    p = sub.add_parser("eval-yolo", help="Evaluate YOLO weights on the test split.")
    p.add_argument("--weights", type=Path)

    sub.add_parser("compare", help="CNN vs YOLO comparison on the test split.")

    for name, helptext in (("predict", "Predict on an image or folder."),
                           ("webcam", "Real-time webcam inference.")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--mode", choices=("yolo", "cnn", "hybrid"), default="yolo")
        p.add_argument("--weights", type=Path, help="YOLO .pt weights.")
        p.add_argument("--checkpoint", type=Path, help="CNN .pt checkpoint.")
        p.add_argument("--conf", type=float, default=0.25)
        if name == "predict":
            p.add_argument("--source", type=Path, required=True)
        else:
            p.add_argument("--camera", type=int, default=0)

    sub.add_parser("all", help="Run the full pipeline (download → compare).")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config(project_dir=args.project_dir.resolve(), seed=args.seed, num_workers=args.workers)
    cfg.make_dirs()

    cmd = args.command
    if cmd == "download":
        download_dataset(cfg)
    elif cmd == "inspect":
        inspect_dataset(cfg)
        plot_annotated_samples(cfg)
    elif cmd == "crops":
        create_crops(cfg)
    elif cmd == "train-cnn":
        cfg.cnn_epochs, cfg.cnn_patience, cfg.batch_size = args.epochs, args.patience, args.batch_size
        cmd_train_cnn(cfg, args.lr, args.dropout, args.weight_decay)
    elif cmd == "tune-cnn":
        cfg.cnn_epochs, cfg.cnn_patience = args.epochs, args.patience
        cmd_tune_cnn(cfg)
    elif cmd == "eval-cnn":
        cmd_eval_cnn(cfg, args.checkpoint)
    elif cmd == "train-yolo":
        cfg.yolo_model, cfg.yolo_epochs, cfg.yolo_batch = args.model, args.epochs, args.batch
        cmd_train_yolo(cfg)
    elif cmd == "tune-yolo":
        cfg.yolo_tuning_model, cfg.yolo_tuning_epochs = args.model, args.epochs
        cmd_tune_yolo(cfg)
    elif cmd == "eval-yolo":
        cmd_eval_yolo(cfg, args.weights)
    elif cmd == "compare":
        cmd_compare(cfg)
    elif cmd == "predict":
        cfg.conf_threshold = args.conf
        cmd_predict(cfg, args.source, args.mode, args.weights, args.checkpoint)
    elif cmd == "webcam":
        cfg.conf_threshold = args.conf
        cmd_webcam(cfg, args.mode, args.camera, args.weights, args.checkpoint)
    elif cmd == "all":
        cmd_all(cfg)


if __name__ == "__main__":
    main()
