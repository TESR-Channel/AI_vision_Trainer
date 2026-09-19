"""TESR Offline Trainer - Step 3: Train YOLO on the dataset.

Works with BOTH dataset layouts and picks the right model automatically:
  detection      dataset/data.yaml + labels/*.txt (5 numbers)  -> yolov8n
  rotated (OBB)  dataset/data.yaml + labels/*.txt (9 numbers)  -> yolov8n-obb
  classification dataset/train/<class>/*.jpg folders           -> yolov8n-cls

Usage:
    python 3_train.py                 # defaults: 80 epochs, imgsz 640
    python 3_train.py --epochs 40

Output: best.pt (copied next to this script) - use it with 4_run.py.
"""
import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).parent


def _detect_or_obb(p):
    """Detection dataset: 5-number labels -> 'detect'; 9-number -> 'obb'."""
    for txt in sorted((p / "labels" / "train").glob("*.txt")) or sorted(p.glob("labels/*.txt")):
        for line in txt.read_text().splitlines():
            n = len(line.split())
            if n == 5:
                return "detect"
            if n == 9:
                return "obb"
    return "detect"


def resolve_task(p):
    """Return (task, data_arg) from the dataset folder layout."""
    if (p / "data.yaml").exists():
        task = _detect_or_obb(p)
        return task, str(p / "data.yaml")
    if (p / "train").is_dir() and any(d.is_dir() for d in (p / "train").iterdir()):
        return "classify", str(p)
    raise SystemExit(
        "No dataset found at %s - press 'Download Dataset (YOLO)' on the web "
        "trainer and unzip the dataset folder here, next to 3_train.py" % p)


def fix_yaml_paths(p):
    """Make data.yaml paths absolute so training works from any cwd."""
    y = p / "data.yaml"
    if not y.exists():
        return
    lines = []
    for line in y.read_text(encoding="utf-8").splitlines():
        if line.startswith("path:"):
            line = "path: %s" % p.resolve()
        lines.append(line)
    y.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "dataset"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default=None,
                    help="override the auto model choice, e.g. yolov8s.pt")
    args = ap.parse_args()

    data_dir = Path(args.data)
    task, data_arg = resolve_task(data_dir)
    fix_yaml_paths(data_dir)

    model_name = args.model or {
        "detect": "yolov8n.pt",
        "obb": "yolov8n-obb.pt",
        "classify": "yolov8n-cls.pt",
    }[task]
    print("Task: %s  |  model: %s  |  data: %s" % (task, model_name, data_arg))

    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.train(data=data_arg, epochs=args.epochs, imgsz=args.imgsz)

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, HERE / "best.pt")
        print("\nDone. best.pt copied to %s" % (HERE / "best.pt"))
        print("Test it:   python 4_run.py --conf 0.25")
        print("Deploy it: python 5_export.py --target pi   (run on THIS computer)")
    else:
        print("Training finished but best.pt was not found at %s" % best)


if __name__ == "__main__":
    main()
