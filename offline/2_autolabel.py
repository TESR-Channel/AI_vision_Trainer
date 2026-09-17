"""TESR Offline Trainer - Step 2: AUTO-label + build the YOLO dataset.

No more drawing boxes by hand. For photos taken on a PLAIN background,
this script finds the object automatically (Otsu threshold -> largest
contour) and writes YOLO labels + data.yaml, split into train/val.

Usage:
    python 2_autolabel.py                # label everything in dataset/raw
    python 2_autolabel.py --review       # preview each box (ENTER=ok, s=skip, q=quit)

Photos where no clean object is found go to dataset/needs_review/ -
re-shoot those on a plainer background, or label them manually later.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def find_box(bgr, margin=0.03):
    """Return (cx, cy, w, h) normalized, or None if no clean object found."""
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Object should NOT dominate the image border - pick the polarity where it doesn't
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.005 * W * H or area > 0.90 * W * H:
        return None
    x, y, w, h = cv2.boundingRect(c)
    mx, my = int(margin * W), int(margin * H)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(W, x + w + mx), min(H, y + h + my)
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/raw")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=float, default=0.15, help="validation share")
    ap.add_argument("--review", action="store_true", help="preview every box")
    args = ap.parse_args()

    raw = Path(args.raw)
    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit("No class folders in %s - run 1_capture.py first" % raw)
    print("Classes:", ", ".join("%d=%s" % (i, c) for i, c in enumerate(classes)))

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    review_dir = out / "needs_review"

    every = max(2, round(1 / args.val)) if args.val > 0 else 0   # every Nth image -> val
    done, failed = 0, 0
    for ci, cls in enumerate(classes):
        imgs = sorted((raw / cls).glob("*.jpg")) + sorted((raw / cls).glob("*.png"))
        kept = 0
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            box = find_box(bgr)
            if box is None:
                review_dir.mkdir(exist_ok=True)
                shutil.copy2(p, review_dir / p.name)
                failed += 1
                continue
            if args.review:
                H, W = bgr.shape[:2]
                cx, cy, w, h = box
                view = bgr.copy()
                cv2.rectangle(view, (int((cx - w / 2) * W), int((cy - h / 2) * H)),
                              (int((cx + w / 2) * W), int((cy + h / 2) * H)), (76, 168, 201), 2)
                cv2.imshow("Review (ENTER=ok, s=skip, q=quit)", view)
                k = cv2.waitKey(0) & 0xFF
                if k == ord("q"):
                    cv2.destroyAllWindows()
                    print("Stopped by user.")
                    break
                if k == ord("s"):
                    review_dir.mkdir(exist_ok=True)
                    shutil.copy2(p, review_dir / p.name)
                    failed += 1
                    continue
            split = "val" if (every and kept % every == 0) else "train"
            kept += 1
            shutil.copy2(p, out / "images" / split / p.name)
            (out / "labels" / split / (p.stem + ".txt")).write_text(
                "%d %.6f %.6f %.6f %.6f\n" % (ci, *box), encoding="utf-8")
            done += 1
    if args.review:
        cv2.destroyAllWindows()

    (out / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n%s" % (
            out.resolve().as_posix(),
            "".join("  %d: %s\n" % (i, c) for i, c in enumerate(classes))),
        encoding="utf-8")
    print("Auto-labeled %d photos (%d need review -> %s)" % (done, failed, review_dir))
    print("Dataset ready: %s" % (out / "data.yaml"))
    print("Next:  python 3_train.py")


if __name__ == "__main__":
    main()
