"""Regenerate the architecture diagrams and flowcharts in docs/diagrams/.

Requires the Graphviz system package (`dot`) and `pip install graphviz`.
Run from the repository root:  python scripts/make_diagrams.py
"""
from pathlib import Path
import graphviz

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "Helvetica"
C = dict(data="#E8F1FB", conv="#DCEFE2", pool="#FFF1D6", head="#FBE3E3",
         neck="#EDE4F7", misc="#F2F2F2", edge="#555555")


def new(name, rankdir="TB", **graph_attr):
    g = graphviz.Digraph(name, format="png")
    g.attr(rankdir=rankdir, dpi="160", fontname=FONT, bgcolor="white", **graph_attr)
    g.attr("node", shape="box", style="rounded,filled", fontname=FONT, fontsize="11",
           color="#666666", penwidth="1.1")
    g.attr("edge", color=C["edge"], fontname=FONT, fontsize="9", arrowsize="0.7")
    return g


def save(g, name):
    g.render(OUT / name, cleanup=True)
    print("wrote", OUT / f"{name}.png")


# 1. End-to-end pipeline -------------------------------------------------------
g = new("pipeline", nodesep="0.35", ranksep="0.45")
g.node("rf", "Roboflow Universe\nwaste-detection-0momv v1\n(YOLOv11 format, 6 classes)", fillcolor=C["data"])
g.node("split", "train 3,337 img / 4,836 obj\nvalid   959 img / 1,380 obj\ntest    477 img /   690 obj", fillcolor=C["data"])
g.node("crop", "Crop every ground-truth box\n→ ImageFolder (split/class/*.jpg)", fillcolor=C["misc"])
g.node("aug", "Resize 224×224, flip, rotate ±10°,\ncolor jitter, ImageNet normalize", fillcolor=C["misc"])
g.node("cnn", "WasteCNN (from scratch)\n4 conv blocks + MLP head\n423,046 params", fillcolor=C["conv"])
g.node("ctune", "CNN tuning\n3 configs (lr, dropout, wd)\nearly stopping on val loss", fillcolor=C["pool"])
g.node("yolo", "YOLO11 (COCO-pretrained)\nbaseline: yolo11m, 40 ep\nAdamW, imgsz 640", fillcolor=C["neck"])
g.node("ytune", "YOLO tuning\n3 × yolo11n configs, 25 ep\nselect by val mAP@50-95", fillcolor=C["pool"])
g.node("ceval", "CNN test evaluation\naccuracy, P/R/F1,\nconfusion matrix", fillcolor=C["head"])
g.node("yeval", "YOLO test evaluation\nP, R, mAP@50, mAP@50-95,\nconfusion matrix", fillcolor=C["head"])
g.node("cmp", "CNN vs YOLO comparison\n(test split)", fillcolor=C["head"])
g.node("inf", "Inference: images / folders /\nwebcam (YOLO, CNN, or hybrid)", fillcolor=C["data"])
g.edges([("rf", "split"), ("split", "crop"), ("crop", "aug"), ("aug", "cnn"), ("cnn", "ctune"),
         ("ctune", "ceval"), ("yolo", "ytune"), ("ytune", "yeval"),
         ("ceval", "cmp"), ("yeval", "cmp"), ("cmp", "inf")])
g.edge("split", "yolo", xlabel=" full images\n + boxes")
save(g, "pipeline_overview")

# 2. WasteCNN architecture -----------------------------------------------------
g = new("cnn", rankdir="LR", nodesep="0.25", ranksep="0.32")
g.node("in", "Input crop\n3 × 224 × 224", fillcolor=C["data"])
blocks = [("b1", "Block 1\nConv3×3 3→32 (p=1)\nBatchNorm · ReLU\nMaxPool 2×2", "32 × 112 × 112"),
          ("b2", "Block 2\nConv3×3 32→64\nBatchNorm · ReLU\nMaxPool 2×2", "64 × 56 × 56"),
          ("b3", "Block 3\nConv3×3 64→128\nBatchNorm · ReLU\nMaxPool 2×2", "128 × 28 × 28"),
          ("b4", "Block 4\nConv3×3 128→256\nBatchNorm · ReLU\n(no pooling)", "256 × 28 × 28")]
prev, prev_shape = "in", ""
for name, label, shape in blocks:
    g.node(name, label, fillcolor=C["conv"])
    g.edge(prev, name, label=f" {prev_shape}")
    prev, prev_shape = name, shape
g.node("gap", "Global Average\nPool (1×1)", fillcolor=C["pool"])
g.node("fc", "Flatten → 256\nDropout(p)\nLinear 256→128 · ReLU\nDropout(p)\nLinear 128→6", fillcolor=C["head"])
g.node("out", "Logits (6)\nsoftmax → class", fillcolor=C["data"])
g.edge("b4", "gap", label="256×28×28")
g.edge("gap", "fc", label="256×1×1")
g.edge("fc", "out")
save(g, "cnn_architecture")

# 3. YOLO11 architecture -------------------------------------------------------
g = new("yolo", rankdir="TB", nodesep="0.3", ranksep="0.32", newrank="true")
with g.subgraph(name="cluster_bb") as s:
    s.attr(label="Backbone", style="rounded,dashed", color="#888888", fontname=FONT)
    s.attr("node", fillcolor=C["conv"])
    s.node("i", "Input image 3×640×640", fillcolor=C["data"])
    s.node("c0", "0  Conv 3×3 s2  (P1/2)")
    s.node("c1", "1  Conv 3×3 s2  (P2/4)")
    s.node("c2", "2  C3k2")
    s.node("c3", "3  Conv 3×3 s2  (P3/8)")
    s.node("c4", "4  C3k2  → P3 skip")
    s.node("c5", "5  Conv 3×3 s2  (P4/16)")
    s.node("c6", "6  C3k2  → P4 skip")
    s.node("c7", "7  Conv 3×3 s2  (P5/32)")
    s.node("c8", "8  C3k2")
    s.node("c9", "9  SPPF (5×5 max-pool ×3)", fillcolor=C["pool"])
    s.node("c10", "10 C2PSA (position-sensitive\nattention)", fillcolor=C["pool"])
    for a, b in zip(["i", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"],
                    ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]):
        s.edge(a, b)
with g.subgraph(name="cluster_neck") as s:
    s.attr(label="Neck (FPN + PAN)", style="rounded,dashed", color="#888888", fontname=FONT)
    s.attr("node", fillcolor=C["neck"])
    s.node("u1", "11 Upsample ×2"); s.node("k1", "12 Concat(P4) → 13 C3k2")
    s.node("u2", "14 Upsample ×2"); s.node("k2", "15 Concat(P3) → 16 C3k2\nP3 out (stride 8, small objects)")
    s.node("d1", "17 Conv 3×3 s2"); s.node("k3", "18 Concat(13) → 19 C3k2\nP4 out (stride 16)")
    s.node("d2", "20 Conv 3×3 s2"); s.node("k4", "21 Concat(10) → 22 C3k2\nP5 out (stride 32, large objects)")
    for a, b in [("u1", "k1"), ("k1", "u2"), ("u2", "k2"), ("k2", "d1"), ("d1", "k3"), ("k3", "d2"), ("d2", "k4")]:
        s.edge(a, b)
g.node("det", "23 Detect head (anchor-free, decoupled)\nper scale: box branch → 4×16 DFL bins  |  cls branch → 6 class scores\n"
       "loss = 7.5·CIoU + 0.5·BCE + 1.5·DFL   →   NMS (IoU 0.7)", fillcolor=C["head"])
g.node("o", "Boxes (x1,y1,x2,y2) + class + confidence", fillcolor=C["data"])
g.edge("c10", "u1")
g.edge("c6", "k1", style="dashed", color="#3B7DD8")
g.edge("c4", "k2", style="dashed", color="#3B7DD8")
g.edge("k1", "k3", style="dashed", color="#3B7DD8")
g.edge("c10", "k4", style="dashed", color="#3B7DD8")
for k in ["k2", "k3", "k4"]:
    g.edge(k, "det")
g.edge("det", "o")
save(g, "yolo11_architecture")

# 4. CNN training loop with early stopping --------------------------------------
g = new("train", nodesep="0.3", ranksep="0.35")
g.node("s", "Start: init WasteCNN, AdamW(lr, wd),\nCrossEntropyLoss, best_val = ∞", shape="box", fillcolor=C["data"])
g.node("tr", "Train one epoch on augmented crops\n(forward → loss → backward → step)", fillcolor=C["conv"])
g.node("va", "Evaluate on validation crops\n(val loss, val accuracy)", fillcolor=C["conv"])
g.node("q1", "val_loss < best_val ?", shape="diamond", style="filled", fillcolor=C["pool"])
g.node("sv", "Save checkpoint\nreset patience counter", fillcolor=C["misc"])
g.node("inc", "patience counter += 1", fillcolor=C["misc"])
g.node("q2", "counter ≥ patience\nor epoch = max ?", shape="diamond", style="filled", fillcolor=C["pool"])
g.node("e", "Restore best weights\nreturn model + history", fillcolor=C["head"])
g.edges([("s", "tr"), ("tr", "va"), ("va", "q1")])
g.edge("q1", "sv", label=" yes"); g.edge("q1", "inc", label=" no")
g.edge("sv", "q2"); g.edge("inc", "q2")
g.edge("q2", "tr", label=" no  ", constraint="false"); g.edge("q2", "e", label=" yes")
save(g, "cnn_training_flowchart")

# 5. Inference modes -----------------------------------------------------------
g = new("infer", rankdir="LR", nodesep="0.3", ranksep="0.4")
g.node("f", "Frame / image", fillcolor=C["data"])
g.node("y", "YOLO11 detector\n(conf ≥ 0.25)", fillcolor=C["neck"])
g.node("b", "Boxes + YOLO labels", fillcolor=C["head"])
g.node("cr", "Crop each box", fillcolor=C["misc"])
g.node("c", "WasteCNN\n(224×224 crop)", fillcolor=C["conv"])
g.node("h", "Boxes + CNN labels\n(hybrid mode)", fillcolor=C["head"])
g.node("w", "Whole-frame CNN label\n(cnn mode — only reliable when\none object fills the frame)", fillcolor=C["head"])
g.edge("f", "y", label=" yolo / hybrid"); g.edge("y", "b"); g.edge("b", "cr", label=" hybrid")
g.edge("cr", "c"); g.edge("c", "h"); g.edge("f", "w", label=" cnn")
save(g, "inference_modes")
