"""TESR YOLO Trainer - Step 3: Train YOLO on the dataset.

Works with BOTH dataset layouts and picks the right model automatically:
  detection      dataset/data.yaml + labels/*.txt (5 numbers)  -> yolov8n
  rotated (OBB)  dataset/data.yaml + labels/*.txt (9 numbers)  -> yolov8n-obb
  classification dataset/train/<class>/*.jpg folders           -> yolov8n-cls

Usage:
    python train.py                 # defaults: 80 epochs, imgsz 640
    python train.py --epochs 40

Output: best.pt (copied next to this script) - use it with run.py.
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
        "trainer and unzip the dataset folder here, next to train.py" % p)


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


def export_onnx(weights, imgsz, dest):
    """best.pt -> dest (.onnx) at the training size, the format TESR ROI Studio reads."""
    from ultralytics import YOLO
    try:
        out = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=12)
    except Exception as e:                       # very new torch without opset 12 -> its default
        print("opset 12 export failed (%s) - trying the default opset" % e)
        out = YOLO(str(weights)).export(format="onnx", imgsz=imgsz)
    shutil.move(str(out), str(dest))            # one model file only: NAME.onnx (no leftover best.onnx)
    print("\n==> %s is ready: TESR ROI Studio -> + Add a trained model -> pick this file" % dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "dataset"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default=None,
                    help="override the auto model choice, e.g. yolov8s.pt")
    ap.add_argument("--batch", type=int, default=16,
                    help="lower to 8 if the GPU runs out of memory")
    ap.add_argument("--device", default=None,
                    help="0 = first GPU, cpu = force CPU (default: auto)")
    ap.add_argument("--onnx", default=None, metavar="NAME",
                    help="after training also export NAME.onnx next to this script "
                         "(ready for TESR ROI Studio: + Add a trained model)")
    ap.add_argument("--degrees", type=float, default=None,
                    help="random turn while training; rotated-box (OBB) datasets default to 90 "
                         "so a few slanted photos teach every slant (0 = off)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    task, data_arg = resolve_task(data_dir)
    fix_yaml_paths(data_dir)

    model_name = args.model or {
        "detect": "yolov8n.pt",
        "obb": "yolov8n-obb.pt",
        "classify": "yolov8n-cls.pt",
    }[task]
    # one-click kit: keep the downloaded starting weights out of sight, inside venv/
    if not args.model and (HERE / "venv").is_dir():
        model_name = str(HERE / "venv" / model_name)
    print("Task: %s  |  model: %s  |  data: %s" % (task, model_name, data_arg))

    from ultralytics import YOLO

    model = YOLO(model_name)
    extra = {}
    deg = args.degrees if args.degrees is not None else (90.0 if task == "obb" else None)
    if deg:
        extra["degrees"] = deg                  # rotated boxes turn with the picture
        print("Random turn while training: +/-%g degrees" % deg)
    results = model.train(data=data_arg, epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, device=args.device, **extra)

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, HERE / "best.pt")
        print("\nDone. best.pt copied to %s" % (HERE / "best.pt"))
        if args.onnx:
            export_onnx(HERE / "best.pt", args.imgsz, HERE / (args.onnx + ".onnx"))
        print("Test it:   python run.py --camera usb --source 0")
        print("Deploy it: python export.py --target pi   (run on THIS computer)")
    else:
        print("Training finished but best.pt was not found at %s" % best)


if __name__ == "__main__":
    main()
