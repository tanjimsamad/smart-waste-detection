# Dataset

## Source

| | |
|---|---|
| Name | Waste Detection, version 1 |
| Publisher | *object-detection-cfmul* on Roboflow Universe |
| URL | https://universe.roboflow.com/object-detection-cfmul/waste-detection-0momv/dataset/1 |
| License | **CC BY 4.0** — free to use and redistribute with attribution |
| Format used | `yolov11` export (YOLO txt labels + `data.yaml`) |
| Download size | ≈ 213 MB zip, 9,551 files after extraction |

The dataset is **not** stored in this repository. `python main.py download` fetches it with your own
Roboflow API key (read from `ROBOFLOW_API_KEY` in `.env`, or prompted).

## Classes

| ID | Class | Description |
|---:|---|---|
| 0 | `Garbage_Bag` | Dark, tied bin bags |
| 1 | `Glass` | Glass bottles and jars |
| 2 | `Paper_Bag` | Brown or printed paper shopping bags |
| 3 | `Pet_Bottle` | Clear PET drink bottles |
| 4 | `Plastic_Bag` | Thin carrier bags, often crumpled |
| 5 | `can` | Aluminium drink cans |

## Splits

| Split | Images | Objects | Per class | Objects / image |
|---|---:|---:|---:|---:|
| train | 3,337 | 4,836 | 806 | 1.45 |
| valid | 959 | 1,380 | 230 | 1.44 |
| test | 477 | 690 | 115 | 1.45 |
| **total** | **4,773** | **6,906** | **1,151** | |

Image counts come from the Ultralytics scan logs; object counts from the crop step. Every class has
exactly the same number of objects in every split (a 70 / 20 / 10 object split), so the dataset is
balanced. Most images contain one or two objects.

## Label format

One `.txt` file per image, one line per object, normalised to [0, 1]:

```
<class_id> <x_center> <y_center> <width> <height>
```

One label in the training set contains a polygon instead of a box; Ultralytics converts it to a box
automatically (a warning appears in the training log).

## CNN crop dataset

`python main.py crops` turns each box into a separate image:

1. Convert the normalised box to pixel corners, clipped to the image.
2. Skip empty or degenerate boxes.
3. Save `data/cnn_crops/<split>/<class>/<image-stem>_<line-index>.jpg`.

The folder layout is what `torchvision.datasets.ImageFolder` expects. Crops keep their original
aspect ratio on disk; they are resized to 224 × 224 only when loaded.

## Attribution

If you use this repository, please credit the dataset authors as required by CC BY 4.0:

> Waste Detection Dataset (v1), object-detection-cfmul, Roboflow Universe.
> https://universe.roboflow.com/object-detection-cfmul/waste-detection-0momv
