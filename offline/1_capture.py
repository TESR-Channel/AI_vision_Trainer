"""TESR Offline Trainer - Step 1: Capture photos per class.

Usage:
    python 1_capture.py --name jetson
    python 1_capture.py --name pi --auto 4      # auto-save 4 photos/sec

Keys:  SPACE = save photo    a = toggle auto mode    q = quit
Tip:   Use a PLAIN background (one solid color) so Step 2 can draw the
       boxes for you automatically. Move the object around the frame -
       corners, edges, near, far - and vary the angle.
"""
import argparse
import time
from pathlib import Path

import cv2


def open_camera(idx):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="class name, e.g. jetson")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--auto", type=float, default=0,
                    help="photos per second in auto mode (0 = manual SPACE only)")
    ap.add_argument("--out", default="dataset/raw", help="output folder")
    args = ap.parse_args()

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.jpg")))

    cap = open_camera(args.camera)
    if cap is None:
        raise SystemExit("Could not open camera %d - try --camera 1, and close "
                         "any app using the camera (browser tab, Zoom...)" % args.camera)

    auto = args.auto > 0
    interval = 1.0 / args.auto if args.auto > 0 else 0.25
    last = 0.0
    print("SPACE = save | a = toggle auto | q = quit  ->  %s" % out)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            view = frame.copy()
            status = "AUTO %.1f/s" % (1.0 / interval) if auto else "MANUAL (SPACE)"
            cv2.putText(view, "%s | %s | %d photos" % (args.name, status, n),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (76, 168, 201), 2)
            cv2.imshow("TESR Capture - " + args.name, view)
            key = cv2.waitKey(1) & 0xFF
            save = key == ord(" ")
            if auto and time.time() - last >= interval:
                save = True
            if save:
                n += 1
                cv2.imwrite(str(out / ("%s_%04d.jpg" % (args.name, n))), frame)
                last = time.time()
            if key == ord("a"):
                auto = not auto
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Saved %d photos of '%s' in %s" % (n, args.name, out))
    print("Next:  python 2_autolabel.py")


if __name__ == "__main__":
    main()
