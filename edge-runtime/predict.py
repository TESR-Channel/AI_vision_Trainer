"""TESR AI Vision Web Trainer - run your exported model.onnx on any edge device.

One script for PC, Raspberry Pi 5 and NVIDIA Jetson. Needs only:
    onnxruntime  numpy  opencv        (no TensorFlow, no PyTorch)

    python predict.py                         # camera from config.json (usb / csi)
    python predict.py --source photo.jpg      # one image
    python predict.py --source clip.mp4       # a video file
    python predict.py --camera csi            # ribbon camera (Pi: Picamera2, Jetson: GStreamer)
    python predict.py --mode multi            # detection: many objects per frame (beta)
    python predict.py --headless              # no window (SSH) - prints results only

The model knows its own task (config.json):
  classification -> one class name for the whole frame
  detection      -> box + CENTER (x, y) in pixels; modes:
                    single  one object, regression box (fast, default)
                    precise one object, 25-crop grid scan (slower, tighter)
                    multi   many objects from the model's YOLO-style output0 (beta)

"none" = nothing you trained is in view: confidence below --min-conf OR the
frame does not look like your training photos (prototypes in config.json).
Written for Python 3.6+ so it also runs on Jetson Nano (JetPack 4).
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = (76, 168, 201)                 # BGR of TESR gold #c9a84c
GREY = (120, 120, 120)


def load_config():
    path = os.path.join(HERE, "config.json")
    if not os.path.exists(path):
        raise SystemExit("config.json not found next to predict.py - unzip the whole export folder")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_jetson():
    return os.path.exists("/etc/nv_tegra_release") or "tegra" in platform.release().lower()


def is_raspberry_pi():
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except Exception:
        return False


# ----------------------------------------------------------------- model ---
class Model(object):
    def __init__(self, cfg, use_gpu=True, trt=False):
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit("onnxruntime is not installed. Run the installer in this folder "
                             "(see README_DEPLOY.md), or:  pip install onnxruntime")
        path = os.path.join(HERE, cfg.get("model", "model.onnx"))
        if not os.path.exists(path):
            raise SystemExit("Model file not found: " + path)
        avail = ort.get_available_providers()
        wanted = []
        if trt and "TensorrtExecutionProvider" in avail:
            wanted.append(("TensorrtExecutionProvider", {
                "trt_fp16_enable": True, "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.join(HERE, "trt_cache")}))
        if use_gpu and "CUDAExecutionProvider" in avail:
            wanted.append("CUDAExecutionProvider")
        wanted.append("CPUExecutionProvider")
        self.sess = ort.InferenceSession(path, providers=wanted)
        self.provider = self.sess.get_providers()[0]
        self.outputs = [o.name for o in self.sess.get_outputs()]
        self.size = int(cfg.get("input", 224))
        self.names = cfg["classes"]
        self.task = cfg.get("task", "classification")
        self.bg = set(cfg.get("background_classes", []))
        p = cfg.get("prototypes")
        self.proto = None
        if p:
            C = np.asarray(p["centroids"], dtype=np.float32)
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
            self.proto = {"C": C, "thr": float(p["threshold"])}

    def blob(self, bgr):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    def run(self, bgr, tta=False):
        x = self.blob(bgr)
        out = dict(zip(self.outputs, self.sess.run(None, {"images": x})))
        if tta and "box" in out:
            # average the box with the box of the mirrored frame: same object, independent
            # noise -> about a third less frame-to-frame jitter (measured)
            xf = np.ascontiguousarray(x[:, :, :, ::-1])
            bf = dict(zip(self.outputs, self.sess.run(None, {"images": xf})))["box"].copy()
            bf[0, 0] = 1.0 - bf[0, 0]
            out["box"] = (out["box"] + bf) / 2.0
        return out

    def similarity(self, feat):
        if self.proto is None:
            return None
        f = feat / (np.linalg.norm(feat) + 1e-9)
        return float(np.max(self.proto["C"] @ f))


# ------------------------------------------------------------- deciding ---
def decide(model, out, min_conf, proto_th):
    probs = out["probs"][0]
    i = int(np.argmax(probs))
    conf = float(probs[i])
    sim = model.similarity(out["feat"][0])
    thr = proto_th if proto_th is not None else (model.proto["thr"] if model.proto else None)
    name = model.names[i]
    if conf < min_conf or (sim is not None and thr is not None and sim < thr):
        name = "none"
    return name, conf, probs


def clamp_region(r):
    cx, cy, w, h = r
    w = min(max(w, 0.05), 1.0)
    h = min(max(h, 0.05), 1.0)
    return (min(max(cx, w / 2), 1 - w / 2), min(max(cy, h / 2), 1 - h / 2), w, h)


def crop(bgr, r):
    H, W = bgr.shape[:2]
    cx, cy, w, h = r
    x1 = max(0, int((cx - w / 2) * W)); x2 = min(W, max(x1 + 2, int((cx + w / 2) * W)))
    y1 = max(0, int((cy - h / 2) * H)); y2 = min(H, max(y1 + 2, int((cy + h / 2) * H)))
    return bgr[y1:y2, x1:x2]


def grid_locate(model, bgr, ci):
    """Precise mode: 25 crops (9 coarse + 10 refine + 2 aspect), keep the best."""
    def score(r):
        return float(model.run(crop(bgr, r))["probs"][0][ci])
    best, bs = None, -1.0
    for cy in (0.25, 0.5, 0.75):
        for cx in (0.25, 0.5, 0.75):
            r = clamp_region((cx, cy, 0.5, 0.5)); s = score(r)
            if s > bs:
                bs, best = s, r
    for size in (0.32, 0.5):
        for dx, dy in ((0, 0), (0.12, 0), (-0.12, 0), (0, 0.12), (0, -0.12)):
            r = clamp_region((best[0] + dx, best[1] + dy, size, size)); s = score(r)
            if s > bs:
                bs, best = s, r
    for r0 in ((best[0], best[1], best[2] * 1.3, best[3] * 0.77),
               (best[0], best[1], best[2] * 0.77, best[3] * 1.3)):
        r = clamp_region(r0); s = score(r)
        if s > bs:
            bs, best = s, r
    return list(best)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def decode_multi(model, out, min_conf, iou_thr, max_det=20):
    """output0 [1,4+nc,N]: cx,cy,w,h in model pixels + per-class scores
    (background classes and the similarity gate already applied in the model)."""
    pred = out["output0"][0].T                    # [N, 4+nc]
    scores = pred[:, 4:]
    cls = np.argmax(scores, axis=1)
    conf = scores[np.arange(len(cls)), cls]
    keep = np.where(conf >= min_conf)[0]
    s = float(model.size)
    dets = []
    for k in keep[np.argsort(-conf[keep])]:
        cx, cy, w, h = pred[k, :4] / s
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        c = int(cls[k])
        if any(d["cls"] == c and iou(d["xyxy"], box) > iou_thr for d in dets):
            continue
        dets.append({"cls": c, "name": model.names[c], "conf": float(conf[k]),
                     "xyxy": box, "box": [float(cx), float(cy), float(w), float(h)]})
        if len(dets) >= max_det:
            break
    return dets


class Stabilizer(object):
    """Steadies the single-object result between frames.
    - box: exponential smoothing (alpha 1 = raw, lower = steadier, a little more lag)
    - class: switches only after it is seen on 2 frames in a row
    - "none": the last box is held for a few frames (a hand passing by, motion blur)"""

    def __init__(self, alpha=0.4, hold=6):
        self.alpha, self.hold = alpha, hold
        self.name, self.conf, self.box = None, 0.0, None
        self.cand, self.cand_n, self.miss = None, 0, 0

    def update(self, name, conf, box):
        if name == "none" or box is None:
            self.miss += 1
            if self.box is not None and self.miss <= self.hold:
                return self.name, self.conf, self.box
            self.name, self.box, self.cand, self.cand_n = None, None, None, 0
            return "none", conf, None
        self.miss = 0
        if self.name is not None and name != self.name:
            self.cand_n = self.cand_n + 1 if name == self.cand else 1
            self.cand = name
            if self.cand_n < 2:                      # not confirmed yet: keep the current answer
                return self.name, self.conf, self.box
            self.box = None                          # confirmed switch: start fresh
        self.cand, self.cand_n = None, 0
        if self.box is None or self.name != name:
            self.box = list(box)
        else:
            a = self.alpha
            self.box = [p + a * (n - p) for p, n in zip(self.box, box)]
        self.name, self.conf = name, conf
        return self.name, self.conf, self.box


# --------------------------------------------------------------- drawing ---
def draw_box(bgr, name, conf, box, idx=None):
    import cv2
    H, W = bgr.shape[:2]
    cx, cy, bw, bh = box
    px, py = int(cx * W), int(cy * H)
    x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
    x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), GOLD, 2)
    cv2.drawMarker(bgr, (px, py), GOLD, cv2.MARKER_CROSS, 22, 2)
    label = "%s %d%%" % (name, round(conf * 100))
    if idx is not None:
        label = "#%d %s" % (idx, label)
    cv2.putText(bgr, label, (max(2, x1 + 4), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GOLD, 2)
    return px, py


def put(bgr, text, y, color=GOLD, scale=0.7):
    import cv2
    cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# ---------------------------------------------------------------- camera ---
class Camera(object):
    """USB webcam, Pi CSI (Picamera2) or Jetson CSI (GStreamer) behind one read()."""

    def __init__(self, kind, index, width, height):
        import cv2
        self.picam = None
        self.cap = None
        if kind == "csi" and not is_jetson():
            try:
                from picamera2 import Picamera2
            except ImportError:
                raise SystemExit(
                    "CSI camera needs Picamera2:  sudo apt install -y python3-picamera2\n"
                    "and a venv made with --system-site-packages (install_pi.sh does this).")
            self.picam = Picamera2(index)
            self.picam.configure(self.picam.create_preview_configuration(
                main={"format": "RGB888", "size": (width, height)}))   # RGB888 = BGR order for OpenCV
            self.picam.start()
            time.sleep(0.5)
            return
        if kind == "csi":                               # Jetson ribbon camera
            pipe = ("nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM),width=1280,height=720,"
                    "framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw,width=%d,height=%d,"
                    "format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
                    % (index, width, height))
            self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                raise SystemExit(
                    "Could not open the Jetson CSI camera.\n"
                    "- check the ribbon cable, then:  gst-launch-1.0 nvarguscamerasrc ! fakesink\n"
                    "- OpenCV must have GStreamer: python3 -c \"import cv2;print(cv2.getBuildInformation())\" | grep GStreamer\n"
                    "  (pip's opencv-python has NO GStreamer - use the system python3-opencv)")
            return
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else \
                   [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(index, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ok, _ = cap.read()
                if ok:
                    self.cap = cap
                    return
            cap.release()
        raise SystemExit(
            "Could not open USB camera %d.\n"
            "- try --source 1 (or 2); list cameras on Linux:  v4l2-ctl --list-devices\n"
            "- close other apps using it (browser Live Test tab, Zoom, Teams)\n"
            "- ribbon camera? use --camera csi" % index)

    def read(self):
        if self.picam is not None:
            return True, self.picam.capture_array()
        return self.cap.read()

    def release(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


# ------------------------------------------------------------------ main ---
def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Run a TESR web-trained ONNX model")
    ap.add_argument("--source", default=None, help="camera index (0,1..) or image/video path")
    ap.add_argument("--camera", choices=["usb", "csi"], default=None,
                    help="camera type (default from config.json: %s)" % cfg.get("camera", "usb"))
    ap.add_argument("--mode", choices=["auto", "classify", "single", "precise", "multi"], default="auto",
                    help="auto = config.json (%s)" % cfg.get("detect_mode", "single"))
    ap.add_argument("--min-conf", type=float, default=None,
                    help="confidence gate (default %.2f)" % cfg.get("min_conf", 0.65))
    ap.add_argument("--iou", type=float, default=0.45, help="multi mode: NMS overlap")
    ap.add_argument("--proto-th", type=float, default=None, help="override the similarity gate")
    ap.add_argument("--no-proto", action="store_true", help="disable the similarity gate")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--headless", action="store_true", help="no window; print results only")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU provider exists")
    ap.add_argument("--trt", action="store_true", help="Jetson: use the onnxruntime TensorRT provider if installed")
    ap.add_argument("--smooth", type=float, default=0.4,
                    help="box smoothing 0..1 (1 = raw/no smoothing, lower = steadier but slight lag)")
    ap.add_argument("--hold", type=int, default=6, help="frames to keep the last box when the object is briefly lost")
    ap.add_argument("--tta", choices=["auto", "on", "off"], default="auto",
                    help="also run the mirrored frame and average the box (steadier, 2x compute). "
                         "auto = on, except on Raspberry Pi")
    args = ap.parse_args()

    import cv2

    model = Model(cfg, use_gpu=not args.cpu, trt=args.trt)
    if args.no_proto:
        model.proto = None
    min_conf = args.min_conf if args.min_conf is not None else float(cfg.get("min_conf", 0.65))
    mode = args.mode if args.mode != "auto" else cfg.get("detect_mode", "single")
    if model.task != "detection":
        mode = "classify"
    if mode == "multi" and "output0" not in model.outputs:
        raise SystemExit("--mode multi needs a detection export (output0 missing)")

    tta = args.tta == "on" or (args.tta == "auto" and not is_raspberry_pi() and cfg.get("target") != "pi5")
    tta = tta and mode in ("single", "precise")
    stab = Stabilizer(alpha=min(1.0, max(0.05, args.smooth)), hold=max(0, args.hold))
    print("TESR model | task: %s | mode: %s | runtime: %s" % (model.task, mode, model.provider))
    if mode in ("single", "precise"):
        print("steadying: smooth %.2f, hold %d frames, mirror-average %s"
              % (stab.alpha, stab.hold, "on" if tta else "off"))
    print("classes: " + ", ".join(model.names))
    if model.proto is not None and mode != "multi":
        print('"none" gates: confidence < %.2f OR similarity < %.2f'
              % (min_conf, args.proto_th if args.proto_th is not None else model.proto["thr"]))

    def run_frame(bgr):
        out = model.run(bgr, tta=tta)
        if mode == "multi":
            dets = decode_multi(model, out, min_conf, args.iou)
            H, W = bgr.shape[:2]
            res = []
            for k, d in enumerate(dets, 1):
                px, py = draw_box(bgr, d["name"], d["conf"], d["box"], k)
                res.append((d["name"], round(d["conf"], 3), (px, py)))
            put(bgr, "%d object(s)" % len(dets), H - 14, GOLD, 0.6)
            return res
        name, conf, _ = decide(model, out, min_conf, args.proto_th)
        if mode == "classify":
            put(bgr, "%s %d%%" % (name, round(conf * 100)) if name != "none" else "none", 34,
                GREY if name == "none" else GOLD, 1.0)
            return [(name, round(conf, 3), None)]
        box = None
        if name != "none":
            if mode == "precise":
                box = grid_locate(model, bgr, model.names.index(name))
            else:
                box = [float(v) for v in out["box"][0]]
        name, conf, box = stab.update(name, conf, box)
        if box is None:
            put(bgr, "none", 34, GREY, 1.0)
            return [("none", round(conf, 3), None)]
        px, py = draw_box(bgr, name, conf, box)
        put(bgr, "center (%d, %d)px" % (px, py), bgr.shape[0] - 14, GOLD, 0.6)
        return [(name, round(conf, 3), (px, py))]

    def show(bgr, title):
        if args.headless:
            return False
        cv2.imshow(title, bgr)
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    src = args.source if args.source is not None else str(cfg.get("camera_index", 0))
    if not src.isdigit():
        if not os.path.exists(src):
            raise SystemExit("File not found: " + src)
        if src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            bgr = cv2.imread(src)
            if bgr is None:
                raise SystemExit("Could not read image: " + src)
            print(run_frame(bgr))
            if not args.headless:
                cv2.imshow("TESR model (press any key)", bgr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return
        cam = cv2.VideoCapture(src)
        reader = cam.read
        release = cam.release
    else:
        kind = args.camera or cfg.get("camera", "usb")
        cam = Camera(kind, int(src), args.width, args.height)
        reader = cam.read
        release = cam.release

    fps, t_last, n = 0.0, time.time(), 0
    try:
        while True:
            ok, bgr = reader()
            if not ok or bgr is None:
                break
            if bgr.ndim == 3 and bgr.shape[2] == 4:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
            res = run_frame(bgr)
            n += 1
            now = time.time()
            if now - t_last >= 1.0:
                fps = n / (now - t_last)
                n, t_last = 0, now
                if args.headless:
                    print("%.1f FPS  %s" % (fps, res))
            put(bgr, "%.1f FPS" % fps, bgr.shape[0] - 40, GOLD, 0.55)
            if show(bgr, "TESR model - press q to quit"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        release()                          # always free the camera
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
