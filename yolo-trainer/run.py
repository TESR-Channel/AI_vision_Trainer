"""TESR YOLO Trainer - Step 4: Run your trained YOLO model.

Works with every task - it reads the task from best.pt automatically:
  detection      -> boxes + crosshair + CENTER (x, y)px, multi-object
  OBB            -> ROTATED boxes that follow tilted objects + center
  classification -> class name + confidence for the whole frame

A status line on screen always tells you what is happening - including the
best candidate BELOW your confidence threshold, so "no answer" is never silent.

Usage:
    python run.py                        # live webcam, press q to quit
    python run.py --source photo.jpg     # single image
    python run.py --conf 0.25            # lower threshold (small datasets)
    python run.py --headless             # no window (SSH / edge device) - Ctrl+C to stop

Detection prints "name center=(x, y)px conf=..." - the numbers a robot arm,
conveyor PLC, or MQTT pipeline needs.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

COLOR = (76, 168, 201)[::-1]
GRAY = (130, 130, 130)


def open_camera(idx):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def _label(frame, text, x, y, color):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def annotate(frame, result, task, conf_min):
    """Draw predictions + a status line; return the console lines to print."""
    out = []
    if task == "classify":
        p = result.probs
        name = result.names[int(p.top1)]
        conf = float(p.top1conf)
        shown = name if conf >= conf_min else "none"
        cv2.putText(frame, "%s %.0f%%" % (shown, conf * 100), (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    GRAY if shown == "none" else COLOR, 2)
        out.append("%s conf=%.2f (top: %s)" % (shown, conf, name))
        return out

    best_name, best_conf = None, 0.0

    if task == "obb" and result.obb is not None:
        ob = result.obb
        for i in range(len(ob)):
            conf = float(ob.conf[i])
            name = result.names[int(ob.cls[i])]
            if conf > best_conf:
                best_name, best_conf = name, conf
            if conf < conf_min:
                continue
            pts = np.asarray(ob.xyxyxyxy[i].cpu()).astype(int).reshape(-1, 2)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.polylines(frame, [pts], True, COLOR, 2)
            cv2.drawMarker(frame, (cx, cy), COLOR, cv2.MARKER_CROSS, 22, 2)
            _label(frame, "%s %.0f%%" % (name, conf * 100),
                   max(2, int(pts[:, 0].min()) + 4), max(20, int(pts[:, 1].min()) - 8), COLOR)
            out.append("%s center=(%d, %d)px conf=%.2f" % (name, cx, cy, conf))
    elif task == "detect":
        for b in result.boxes:
            conf = float(b.conf[0])
            name = result.names[int(b.cls[0])]
            if conf > best_conf:
                best_name, best_conf = name, conf
            if conf < conf_min:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
            cv2.drawMarker(frame, (cx, cy), COLOR, cv2.MARKER_CROSS, 22, 2)
            _label(frame, "%s %.0f%%" % (name, conf * 100),
                   max(2, x1 + 4), max(20, y1 - 8), COLOR)
            _label(frame, "(%d, %d)px" % (cx, cy), max(2, x1 + 4), y2 + 22, COLOR)
            out.append("%s center=(%d, %d)px conf=%.2f" % (name, cx, cy, conf))

    # Always-on status line - never leaves you guessing
    if out:
        status = "%d object(s) >= %.2f" % (len(out), conf_min)
    elif best_name:
        status = "no object >= %.2f  (best: %s %.0f%% - try --conf %.2f)" % (
            conf_min, best_name, best_conf * 100, max(0.1, round(best_conf - 0.05, 2)))
    else:
        status = "no object found at all - check lighting / add more training photos"
    _label(frame, status, 10, 24, COLOR if out else GRAY)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--source", default="0", help="image path, or camera index")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--headless", action="store_true",
                    help="no display window - print results only (SSH / edge). "
                         "Image source saves result_<name>.jpg instead")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    task = getattr(model, "task", "detect")
    if task not in ("classify", "obb"):
        task = "detect"
    print("Task: %s  |  classes: %s" % (task, ", ".join(model.names.values())))

    # run inference with a LOW floor so the status line can show near-misses;
    # args.conf only decides what is drawn/printed as a real detection
    floor = min(0.1, args.conf)

    if not args.source.isdigit():
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit("Could not read image: " + args.source)
        lines = annotate(frame, model(frame, conf=floor, verbose=False)[0], task, args.conf)
        for ln in lines:
            print(ln)
        if not lines:
            print("none - nothing above --conf %.2f (see the status line on the image)"
                  % args.conf)
        if args.headless:
            out_path = "result_" + Path(args.source).name
            cv2.imwrite(out_path, frame)
            print("Annotated image saved: %s" % out_path)
        else:
            cv2.imshow("TESR YOLO (press any key)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cap = open_camera(int(args.source))
    if cap is None:
        raise SystemExit("Could not open camera %s - try --source 0 or --source 1 "
                         "(index 2+ is often an IR/virtual camera), and close any "
                         "app using the camera" % args.source)
    if args.headless:
        print("Headless mode - Ctrl+C to stop")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for ln in annotate(frame, model(frame, conf=floor, verbose=False)[0],
                               task, args.conf):
                print(ln)
            if not args.headless:
                cv2.imshow("TESR YOLO - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
