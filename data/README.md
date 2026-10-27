# data/

This folder is filled by `main.py` and is git-ignored (except this file).

```
data/
├── waste_dataset/          ← `python main.py download`  (Roboflow, YOLOv11 format)
│   ├── data.yaml
│   ├── train/{images,labels}/   3,337 images
│   ├── valid/{images,labels}/     959 images
│   └── test/{images,labels}/      477 images
└── cnn_crops/              ← `python main.py crops`
    ├── train/<class>/*.jpg        806 crops per class (4,836 total)
    ├── valid/<class>/*.jpg        230 crops per class (1,380 total)
    └── test/<class>/*.jpg         115 crops per class (690 total)
```

Dataset: [Waste Detection v1](https://universe.roboflow.com/object-detection-cfmul/waste-detection-0momv/dataset/1)
by *object-detection-cfmul* on Roboflow Universe, **CC BY 4.0**. See [`docs/DATASET.md`](../docs/DATASET.md).
