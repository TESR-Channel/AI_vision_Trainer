"""TESR Offline Trainer - Step 3: Train YOLO (detection or classification).

The task is detected automatically from your dataset layout:
  dataset/data.yaml + images/ + labels/   -> object detection  (yolov8n)
    labels with 8 corner coords          -> rotated boxes     (yolov8n-obb)
  dataset/train/<class>/ + val/<class>/   -> classification    (yolov8n-cls)

Usage:
    python 3_train.py                     # auto-detect task, sensible defaults
    python 3_train.py --epochs 100
    python 3_train.py --model yolov8s.pt  # bigger = more accurate, slower

Runs on GPU automatically if available (NVIDIA/CUDA), otherwise CPU
(slower but works - go get coffee). Augmentation is built into YOLO, so
~100 photos per class already trains well. The best weights are copied
to ./best.pt when done.
"""
import argparse
import shutil
from pathlib import Path


def _detect_or_obb(yaml_path):
    """Peek at the first label line: 5 tokens = detect, 9 = OBB (rotated)."""
    for split in ("train", "val"):
        d = yaml_path.parent / "labels" / split
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.txt")):
            tokens = f.read_text(encoding="utf-8").split("\n")[0].split()
            if tokens:
                return "obb" if len(tokens) >= 9 else "detect"
    return "detect"


def resolve_task(data_arg):
    """Return (task, train_target). Accepts a dataset folder or a data.yaml."""
    p = Path(data_arg).resolve()
    if p.suffix in (".yaml", ".yml"):
        if not p.exists():
            raise SystemExit("%s not found" % p)
        return _detect_or_obb(p), p
    if (p / "data.yaml").exists():
        return _detect_or_obb(p / "data.yaml"), p / "data.yaml"
    if (p / "train").is_dir():
        return "classify", p
    raise SystemExit(
        "No dataset found at %s - run 2_autolabel.py (detection), "
        "2_autolabel.py --task classify (classification), or unzip a dataset "
        "exported from the web trainer here" % p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="dataset folder (or a data.yaml)")
    ap.add_argument("--model", default=None,
                    help="default: yolov8n.pt (detect) / yolov8n-cls.pt (classify)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="default: 640 (detect) / 224 (classify)")
    args = ap.parse_args()

    task, target = resolve_task(args.data)
    default_model = {"detect": "yolov8n.pt", "obb": "yolov8n-obb.pt",
                     "classify": "yolov8n-cls.pt"}[task]
    model_name = args.model or default_model
    imgsz = args.imgsz or (224 if task == "classify" else 640)
    print("Task: %s  |  model: %s  |  imgsz: %d" % (task, model_name, imgsz))

    if task in ("detect", "obb"):
        # Make data.yaml portable: force `path:` to the yaml's own folder, so datasets
        # exported from the web trainer (path: .) train correctly from anywhere.
        lines = [l for l in target.read_text(encoding="utf-8").splitlines()
                 if not l.startswith("path:")]
        lines.insert(0, "path: %s" % target.parent.as_posix())
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from ultralytics import YOLO   # first run downloads the base model (~6 MB)

    model = YOLO(model_name)
    results = model.train(data=str(target), epochs=args.epochs, imgsz=imgsz)
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, "best.pt")
    print("\nDone. Best weights copied to ./best.pt")
    print("Next:  python 4_run.py")


if __name__ == "__main__":
    main()
