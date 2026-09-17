"""TESR Offline Trainer - Step 3: Train a real YOLO detector.

Usage:
    python 3_train.py                     # defaults: yolov8n, 60 epochs, 640px
    python 3_train.py --epochs 100
    python 3_train.py --model yolov8s.pt  # bigger = more accurate, slower

Runs on GPU automatically if available (NVIDIA/CUDA), otherwise CPU
(slower but works - go get coffee). Augmentation (flip, scale, mosaic,
lighting) is built into YOLO, so ~100 photos per class already trains well.
The best weights are copied to ./best.pt when done.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit("%s not found - run 2_autolabel.py first" % args.data)

    from ultralytics import YOLO   # first run downloads the base model (~6 MB)

    model = YOLO(args.model)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, "best.pt")
    print("\nDone. Best weights copied to ./best.pt")
    print("Next:  python 4_run.py")


if __name__ == "__main__":
    main()
