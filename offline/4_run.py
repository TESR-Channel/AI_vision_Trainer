"""TESR Offline Trainer - Step 4: Run your trained YOLO model.

Works with both tasks - it reads the task from best.pt automatically:
  detection      -> boxes + crosshair + CENTER (x, y)px, multi-object
  classification -> class name + confidence for the whole frame

Usage:
    python 4_run.py                        # live webcam, press q to quit
    python 4_run.py --source photo.jpg     # single image
    python 4_run.py --conf 0.6             # stricter confidence

Detection prints "name center=(x, y)px conf=.." - the numbers a robot arm,
conveyor PLC, or MQTT pipeline needs.
"""
import argparse

import cv2

COLOR = (76, 168, 201)[::-1]


def open_camera(idx):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def annotate(frame, result, task, conf_min):
    """Draw predictions on the frame; return the lines to print."""
    out = []
    if task == "classify":
        p = result.probs
        name = result.names[int(p.top1)]
        conf = float(p.top1conf)
        if conf < conf_min:
            name = "none"
        cv2.putText(frame, "%s %.0f%%" % (name, conf * 100), (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (120, 120, 120) if name == "none" else COLOR, 2)
        out.append("%s conf=%.2f" % (name, conf))
        return out
    for b in result.boxes:
        conf = float(b.conf[0])
        if conf < conf_min:
            continue
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        name = result.names[int(b.cls[0])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        cv2.drawMarker(frame, (cx, cy), COLOR, cv2.MARKER_CROSS, 22, 2)
        cv2.putText(frame, "%s %.0f%%" % (name, conf * 100),
                    (max(2, x1 + 4), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
        cv2.putText(frame, "(%d, %d)px" % (cx, cy), (max(2, x1 + 4), y2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)
        out.append("%s center=(%d, %d)px conf=%.2f" % (name, cx, cy, conf))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--source", default="0", help="image path, or camera index")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    task = "classify" if getattr(model, "task", "") == "classify" else "detect"
    print("Task: %s  |  classes: %s" % (task, ", ".join(model.names.values())))

    if not args.source.isdigit():
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit("Could not read image: " + args.source)
        lines = annotate(frame, model(frame, verbose=False)[0], task, args.conf)
        for ln in lines:
            print(ln)
        if not lines:
            print("none - no trained object found")
        cv2.imshow("TESR YOLO (press any key)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = open_camera(int(args.source))
    if cap is None:
        raise SystemExit("Could not open camera %s - try --source 1, and close "
                         "any app using the camera" % args.source)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for ln in annotate(frame, model(frame, verbose=False)[0], task, args.conf):
                print(ln)
            cv2.imshow("TESR YOLO - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
