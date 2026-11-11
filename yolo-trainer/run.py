r"""TESR YOLO Trainer - Step 4: Run your trained YOLO model.

Works with every task - it reads the task from best.pt automatically:
  detection      -> boxes + crosshair + CENTER (x, y)px, multi-object
  OBB            -> ROTATED boxes that follow tilted objects + center
  classification -> class name + confidence for the whole frame

A status line on screen always tells you what is happening - including the
best candidate BELOW your confidence threshold, so "no answer" is never silent.
A live FPS counter shows top-right of the window (printed periodically in
--headless) - measure real speed on the real device.

Usage - the same words as every TESR export (predict.py, tesr_vision_runner.py):
    venv\Scripts\activate                              (Mac / Linux: source venv/bin/activate)
    python run.py --camera usb --source 0              # USB webcam 0, press q to quit
    python run.py --camera usb --source 2              # another webcam
    python run.py --camera csi --source 0              # Pi / Jetson ribbon camera
    python run.py --camera file --source clip.mp4      # a video file (or a photo: --source photo.jpg)
    python run.py --camera ip --source rtsp://...      # an IP camera stream
    python run.py --camera usb --source 0 --conf 0.25  # lower threshold (small datasets)
    python run.py --camera usb --source 0 --headless   # no window (SSH / edge device) - Ctrl+C to stop
The model is picked for you: best.pt, else the *_yolo.onnx next to this script (--weights to choose).

Detection prints "name center=(x, y)px conf=..." - the numbers a robot arm,
conveyor PLC, or MQTT pipeline needs.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

COLOR = (76, 168, 201)[::-1]
GRAY = (130, 130, 130)


HERE = Path(__file__).resolve().parent


def pick_weights(arg):
    """--weights left out: best.pt, else the NAME_yolo.onnx the one-click kit wrote."""
    if arg:
        return arg
    for cand in [HERE / "best.pt"] + sorted(HERE.glob("*_yolo.onnx")) + sorted(HERE.glob("*.onnx")):
        if cand.is_file():
            return str(cand)
    raise SystemExit("No model here - train first (TRAIN_WINDOWS.bat / train_mac_linux.sh), "
                     "or give one: --weights path/to/best.pt")


def is_jetson():
    return Path("/etc/nv_tegra_release").exists()


class PiCam(object):
    """Pi ribbon camera through Picamera2, with the cv2.VideoCapture read()/release() calls."""

    def __init__(self, index):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise SystemExit("CSI camera on the Pi needs Picamera2:  sudo apt install -y python3-picamera2\n"
                             "and a venv made with --system-site-packages")
        self.cam = Picamera2(index)
        self.cam.configure(self.cam.create_preview_configuration(
            main={"format": "RGB888", "size": (1280, 720)}))      # RGB888 = BGR order for OpenCV
        self.cam.start()
        time.sleep(0.5)

    def read(self):
        return True, self.cam.capture_array()

    def release(self):
        self.cam.stop()


def open_source(kind, source):
    """Return something with read()/release() for --camera usb | csi | file | ip."""
    if kind == "csi":
        idx = int(source) if source.isdigit() else 0
        if not is_jetson():
            return PiCam(idx)
        pipe = ("nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM),width=1280,height=720,"
                "framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
                "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1" % idx)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise SystemExit("Could not open the Jetson CSI camera - check the ribbon cable; "
                             "OpenCV needs GStreamer (use the system python3-opencv)")
        return cap
    if kind in ("file", "ip"):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit("Could not open %s: %s" % ("stream" if kind == "ip" else "file", source))
        return cap
    cap = open_camera(int(source))
    if cap is None:
        raise SystemExit("Could not open camera %s - try --source 0, 1 or 2 "
                         "and close any app using the camera" % source)
    return cap


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
    ap.add_argument("--weights", default=None,
                    help="model file (default: best.pt, else the *_yolo.onnx in this folder)")
    ap.add_argument("--camera", choices=["usb", "csi", "file", "ip"], default="usb",
                    help="usb webcam, csi ribbon camera, file (photo / video), ip stream - "
                         "the same words as predict.py and tesr_vision_runner.py")
    ap.add_argument("--source", default="0",
                    help="camera index for usb / csi (0, 1, 2), a path for file, a URL for ip")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--headless", action="store_true",
                    help="no display window - print results only (SSH / edge). "
                         "Image source saves result_<name>.jpg instead")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = pick_weights(args.weights)
    print("Model: %s" % weights)
    model = YOLO(weights)
    task = getattr(model, "task", "detect")
    if task not in ("classify", "obb"):
        task = "detect"
    print("Task: %s  |  classes: %s" % (task, ", ".join(model.names.values())))

    # run inference with a LOW floor so the status line can show near-misses;
    # args.conf only decides what is drawn/printed as a real detection
    floor = min(0.1, args.conf)

    is_photo = args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    if is_photo or (not args.source.isdigit() and args.camera == "usb" and Path(args.source).is_file()
                    and not args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))):
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

    kind = args.camera
    if kind == "usb" and not args.source.isdigit():          # a path or URL given with the default
        kind = "ip" if "://" in args.source else "file"
    cap = open_source(kind, args.source)
    if args.headless:
        print("Headless mode - Ctrl+C to stop")
    fps, t_prev, n_frames = 0.0, time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for ln in annotate(frame, model(frame, conf=floor, verbose=False)[0],
                               task, args.conf):
                print(ln)
            now = time.time()                      # end-to-end FPS (camera + AI + draw)
            if now > t_prev:
                inst = 1.0 / (now - t_prev)
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            t_prev = now
            n_frames += 1
            if args.headless:
                if n_frames % 30 == 0:
                    print("FPS: %.1f" % fps)
            else:
                _label(frame, "%.1f FPS" % fps, frame.shape[1] - 130, 24, COLOR)
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
