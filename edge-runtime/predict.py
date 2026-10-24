"""TESR AI Vision Web Trainer - run your exported model.onnx on any edge device.

One script for PC, Raspberry Pi 5 and NVIDIA Jetson. Needs only:
    onnxruntime  numpy  opencv        (no TensorFlow, no PyTorch)

    python predict.py                         # camera from config.json (usb / csi)
    python predict.py --source photo.jpg      # one image
    python predict.py --source clip.mp4       # a video file
    python predict.py --camera csi            # ribbon camera (Pi: Picamera2, Jetson: GStreamer)
    python predict.py --mode single           # detection: only the main object
    python predict.py --headless              # no window (SSH) - prints results only
    python predict.py --trt                   # Jetson: the engine build_trt.sh made

The model knows its own task (config.json):
  classification -> one class name for the whole frame
  detection      -> box + CENTER (x, y) in pixels; modes:
                    multi   every trained object in view, each tracked + smoothed (default)
                    single  the one main object
                    precise one object, 25-crop grid scan (slower)

multi mode lists only the objects it finds ("0 object(s)", headless "[]" if none);
--min-conf is its per-object threshold (default multi_conf in config.json) -
lower finds more, higher is stricter. The similarity gate is not used in multi.
single / precise / classification print "none" = nothing you trained is in
view: confidence below --min-conf (default 0.65) OR the frame does not look
like your training photos (prototypes in config.json; --no-proto turns it off).
Written for Python 3.6+ so it also runs on Jetson Nano (JetPack 4) - which is
more limited than Jetson Orin Nano (performance and support for new software).
"""
import argparse
import json
import math
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


# --------------------------------------------------- TensorRT (Jetson) ---
ENGINES = ["model_fp16.engine", "model_fp32.engine"]     # written by build_trt.sh


def _cudart():
    """The CUDA runtime that ships with JetPack, loaded with ctypes (no pycuda needed)."""
    import ctypes
    import ctypes.util
    import glob
    names = [ctypes.util.find_library("cudart")] + sorted(
        glob.glob("/usr/local/cuda*/lib64/libcudart.so*") + glob.glob("/usr/lib/aarch64-linux-gnu/libcudart.so*"))
    for n in names:
        if not n:
            continue
        try:
            lib = ctypes.CDLL(n)
        except OSError:
            continue
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        return lib
    raise RuntimeError("CUDA runtime (libcudart) not found")


class TrtSession(object):
    """Runs a .engine from build_trt.sh; same run(None, feed) call as onnxruntime."""

    def __init__(self, path):
        import ctypes
        import tensorrt as trt
        self.ct, self.cuda = ctypes, _cudart()
        with open(path, "rb") as f:
            engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("TensorRT could not load it (built with another TensorRT version "
                               "or on another board?) - run  bash build_trt.sh  again on this Jetson")
        self.engine, self.ctx = engine, engine.create_execution_context()
        new_api = hasattr(engine, "num_io_tensors")             # TensorRT 8.5+ / 10
        n = engine.num_io_tensors if new_api else engine.num_bindings
        self.io = []                                             # (name, is_input, host array, device ptr)
        for i in range(n):
            if new_api:
                name = engine.get_tensor_name(i)
                is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape, dt = engine.get_tensor_shape(name), engine.get_tensor_dtype(name)
            else:                                                # TensorRT 8.2 (Jetson Nano, JetPack 4)
                name, is_in = engine.get_binding_name(i), engine.binding_is_input(i)
                shape, dt = engine.get_binding_shape(i), engine.get_binding_dtype(i)
            host = np.zeros(tuple(shape), dtype=trt.nptype(dt))
            ptr = ctypes.c_void_p()
            if self.cuda.cudaMalloc(ctypes.byref(ptr), host.nbytes) != 0:
                raise RuntimeError("cudaMalloc failed")
            if new_api:
                self.ctx.set_tensor_address(name, ptr.value)
            self.io.append((name, is_in, host, ptr))
        self.new_api = new_api
        self.output_names = [io[0] for io in self.io if not io[1]]

    def run(self, _names, feed):
        cp = self.cuda.cudaMemcpy
        for name, is_in, host, ptr in self.io:
            if is_in:
                host[...] = feed[name]
                cp(ptr, host.ctypes.data, host.nbytes, 1)        # host -> GPU
        if self.new_api:
            self.ctx.execute_async_v3(0)
            self.cuda.cudaDeviceSynchronize()
        else:
            self.ctx.execute_v2([io[3].value for io in self.io])
        out = []
        for name, is_in, host, ptr in self.io:
            if not is_in:
                cp(host.ctypes.data, ptr, host.nbytes, 2)        # GPU -> host
                out.append(host.copy())
        return out


def open_engine():
    """--trt: the engine build_trt.sh made, or None after printing why not."""
    found = [e for e in ENGINES if os.path.exists(os.path.join(HERE, e))]
    if not found:
        print("--trt: no %s here - build it on this Jetson first:  bash build_trt.sh" % " / ".join(ENGINES))
        return None
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("--trt: found %s, but Python cannot import tensorrt - on the Jetson:\n"
              "       sudo apt install -y python3-libnvinfer   (the venv must use --system-site-packages,\n"
              "       install_jetson.sh does this)" % found[0])
        return None
    try:
        sess = TrtSession(os.path.join(HERE, found[0]))
    except Exception as e:
        print("--trt: could not use %s: %s" % (found[0], e))
        return None
    sess.provider = "TensorRT engine %s" % found[0]
    return sess


# ----------------------------------------------------------------- model ---
class Model(object):
    def __init__(self, cfg, use_gpu=True, trt=False):
        path = os.path.join(HERE, cfg.get("model", "model.onnx"))
        self.sess = open_engine() if trt else None
        if self.sess is not None:
            self.provider, self.outputs = self.sess.provider, self.sess.output_names
        else:
            try:
                import onnxruntime as ort
            except ImportError:
                raise SystemExit("onnxruntime is not installed. Run the installer in this folder "
                                 "(see README_DEPLOY.md), or:  pip install onnxruntime")
            if not os.path.exists(path):
                raise SystemExit("Model file not found: " + path)
            avail = ort.get_available_providers()
            wanted = []
            if trt and "TensorrtExecutionProvider" in avail:
                # onnxruntime-gpu with TensorRT builds its own engine (trt_cache/, first start is slow)
                wanted.append(("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True, "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": os.path.join(HERE, "trt_cache")}))
            if use_gpu and "CUDAExecutionProvider" in avail:
                wanted.append("CUDAExecutionProvider")
            wanted.append("CPUExecutionProvider")
            if trt:
                w = wanted[0] if isinstance(wanted[0], str) else wanted[0][0]
                print("--trt: continuing with onnxruntime " + w
                      + (" (it builds its own engine in trt_cache/ - first start takes minutes)"
                         if w == "TensorrtExecutionProvider" else " - no TensorRT"))
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
        if tta and ("box" in out or "output0" in out):
            # also run the mirrored frame: same objects, independent noise -> steadier boxes
            xf = np.ascontiguousarray(x[:, :, :, ::-1])
            of = dict(zip(self.outputs, self.sess.run(None, {"images": xf})))
            if "box" in out:
                bf = of["box"].copy()
                bf[0, 0] = 1.0 - bf[0, 0]
                out["box"] = (out["box"] + bf) / 2.0
            if "output0" in of:
                out["output0_flip"] = of["output0"]
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
    if conf < min_conf or (sim is not None and thr is not None and sim < thr) or name in model.bg:
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
    """output0 [1,4+nc,N]: cx,cy,w,h in model pixels + per-class scores.
    Neighbouring cells that see the same object are MERGED into one score-weighted box
    (steadier than picking one cell), and the mirrored frame's detections are fused in
    when available (out["output0_flip"])."""
    s = float(model.size)

    def peaks(o):
        # keep a cell only where its class score is the local maximum of its 3x3
        # neighbourhood (one response per object; weak echoes next to it are dropped)
        n = o.shape[1]
        g = int(round(math.sqrt(n)))
        if g * g != n:
            return o.T
        sc = o[4:].reshape(-1, g, g)
        pad = np.pad(sc, ((0, 0), (1, 1), (1, 1)), constant_values=-1.0)
        nb = np.max([pad[:, 1 + dy:1 + dy + g, 1 + dx:1 + dx + g]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
        sc = np.where(sc >= nb, sc, 0.0)
        o = o.copy()
        o[4:] = sc.reshape(-1, n)
        return o.T

    preds = [peaks(out["output0"][0])]
    if "output0_flip" in out:
        pf = peaks(out["output0_flip"][0]).copy()
        pf[:, 0] = s - pf[:, 0]
        preds.append(pf)
    pred = np.concatenate(preds, 0)                     # [N, 4+nc]
    scores = pred[:, 4:]
    cls = np.argmax(scores, axis=1)
    conf = scores[np.arange(len(cls)), cls]
    bg = set(i for i, n in enumerate(model.names) if n in model.bg)
    keep = [k for k in np.where(conf >= min_conf)[0] if int(cls[k]) not in bg]
    keep.sort(key=lambda k: -conf[k])
    groups = []
    for k in keep:
        cx, cy, w, h = pred[k, :4] / s
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        c = int(cls[k])
        for g in groups:
            o = iou(g["xyxy"], box)
            inside = g["xyxy"][0] <= cx <= g["xyxy"][2] and g["xyxy"][1] <= cy <= g["xyxy"][3]
            # center inside a stronger box = same object (any class); else same-class overlap
            if inside or (g["cls"] == c and o > iou_thr) or o > 0.5:
                g["m"].append((float(conf[k]), [cx, cy, w, h]))
                break
        else:
            if len(groups) < max_det:
                groups.append({"cls": c, "conf": float(conf[k]), "xyxy": box, "m": [(float(conf[k]), [cx, cy, w, h])]})
    dets = []
    for g in groups:
        wts = np.array([m[0] for m in g["m"]]); bx = np.array([m[1] for m in g["m"]])
        b = (bx * wts[:, None]).sum(0) / wts.sum()
        dets.append({"cls": g["cls"], "name": model.names[g["cls"]], "conf": g["conf"],
                     "xyxy": [b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2],
                     "box": [float(v) for v in b]})
    return dets


STEADY_PRESETS = {
    # One Euro filter: strong smoothing while the object is still, almost no lag when it
    # moves. min_cutoff = smoothing at rest (lower = steadier), beta = how fast it follows
    # motion, deadband = ignore changes smaller than this (fraction of the frame).
    "high":   {"min_cutoff": 0.05, "beta": 4.0, "d_cutoff": 1.0, "deadband": 0.006},
    "normal": {"min_cutoff": 0.10, "beta": 6.0, "d_cutoff": 1.0, "deadband": 0.004},
    "low":    {"min_cutoff": 0.50, "beta": 10.0, "d_cutoff": 1.0, "deadband": 0.002},
}


class Stabilizer(object):
    """Steadies the single-object result between frames (tuned on real footage).
    - box: One Euro filter + small deadband, time-based so it behaves the same at any FPS
    - class: switches only after it is seen on 2 frames in a row; confidence is smoothed
    - "none": the last box is held for a few frames (a hand passing by, motion blur)"""

    def __init__(self, preset="normal", hold=6):
        self.off = preset == "off"
        p = STEADY_PRESETS.get(preset, STEADY_PRESETS["normal"])
        self.mc, self.beta, self.dc, self.db = p["min_cutoff"], p["beta"], p["d_cutoff"], p["deadband"]
        self.hold = hold
        self.reset()
        self.cand, self.cand_n, self.miss = None, 0, 0
        self.name, self.conf = None, 0.0

    def reset(self):
        self.x = self.dx = self.shown = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def _filter(self, box, now):
        z = np.asarray(box, dtype=np.float64)
        if self.off:
            return list(z)
        if self.x is None:
            self.x, self.dx, self.shown, self.t = z.copy(), np.zeros(4), z.copy(), now
            return list(z)
        dt = min(max(now - self.t, 1e-3), 0.5)
        self.t = now
        self.dx = self.dx + self._alpha(self.dc, dt) * ((z - self.x) / dt - self.dx)
        cut = self.mc + self.beta * np.abs(self.dx)
        a = 2 * math.pi * cut * dt
        self.x = self.x + (a / (a + 1)) * (z - self.x)
        moved = np.abs(self.x - self.shown) > np.array([self.db, self.db, self.db * 1.5, self.db * 1.5])
        self.shown[moved] = self.x[moved]
        return list(self.shown)

    def update(self, name, conf, box, now=None):
        now = time.time() if now is None else now
        if name == "none" or box is None:
            self.miss += 1
            if self.shown is not None and self.miss <= self.hold:
                return self.name, self.conf, list(self.shown)
            self.reset()
            self.name, self.cand, self.cand_n = None, None, 0
            return "none", conf, None
        self.miss = 0
        if self.name is not None and name != self.name:
            self.cand_n = self.cand_n + 1 if name == self.cand else 1
            self.cand = name
            if self.cand_n < 2:                      # not confirmed yet: keep the current answer
                return self.name, self.conf, list(self.shown)
            self.reset()                             # confirmed switch: start fresh
        self.cand, self.cand_n = None, 0
        self.conf = conf if (self.name != name or self.off) else self.conf + 0.3 * (conf - self.conf)
        self.name = name
        return self.name, self.conf, self._filter(box, now)


class MultiSteadier(object):
    """Many objects at once: each detection is matched to a track (by overlap / distance),
    and every track has its own Stabilizer - same smoothing, hold and class
    hysteresis as single-object mode. Tracks keep an id (#1, #2 ...) while visible."""

    def __init__(self, preset="high", hold=6, confirm=3):
        self.preset, self.hold, self.confirm = preset, hold, confirm
        self.tracks = []            # dicts: id, st (Stabilizer), box (last raw xywh), hits

    def _new_id(self):
        used = set(t["id"] for t in self.tracks)
        i = 1
        while i in used:
            i += 1
        return i                    # lowest free number: objects stay 1, 2, 3 ...

    @staticmethod
    def _cost(a, b):
        ax = [a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2]
        bx = [b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2]
        o = iou(ax, bx)
        d = math.hypot(a[0] - b[0], a[1] - b[1]) / max(1e-6, (a[2] + a[3] + b[2] + b[3]) / 4)
        return o, d

    def update(self, dets, now=None, start_conf=0.0):
        """dets may include weak detections: they can only CONTINUE an existing track;
        starting a new object needs conf >= start_conf (hysteresis -> stable ids)."""
        now = time.time() if now is None else now
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                o, dist = self._cost(t["box"], d["box"])
                if o > 0.15 or dist < 0.6:
                    pairs.append((-(o - 0.2 * dist), ti, di))
        pairs.sort()
        used_t, used_d, out = set(), set(), []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            if d["conf"] < start_conf:
                continue        # weak: keeps the object from being re-created, but never moves/shows a box
            used_t.add(ti)
            t["box"] = d["box"]
            t["hits"] += 1
            name, conf, box = t["st"].update(d["name"], d["conf"], d["box"], now)
            if box is not None and t["hits"] >= self.confirm:
                out.append({"id": t["id"], "name": name, "conf": conf, "box": box})
        alive = []
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                if t["hits"] < self.confirm:
                    continue                                              # never confirmed: drop
                name, conf, box = t["st"].update("none", 0.0, None, now)   # hold for a few frames
                if box is None:
                    continue
                out.append({"id": t["id"], "name": name, "conf": conf, "box": box})
            alive.append(t)
        self.tracks = alive
        for di, d in enumerate(dets):
            if di in used_d or d["conf"] < start_conf:
                continue
            st = Stabilizer(self.preset, self.hold)
            name, conf, box = st.update(d["name"], d["conf"], d["box"], now)
            t = {"id": self._new_id(), "st": st, "box": d["box"], "hits": 1}
            self.tracks.append(t)
            if self.confirm <= 1:
                out.append({"id": t["id"], "name": name, "conf": conf, "box": box})
        out.sort(key=lambda r: r["id"])
        return out


# --------------------------------------------------------------- drawing ---
def draw_box(bgr, name, conf, box, idx=None):
    """Box + center cross + one label:  #1 Orange 100% (128, 262)
    The label sits just above the box (inside it when there is no room above) and is
    always kept fully inside the image, on a dark strip so it stays readable."""
    import cv2
    H, W = bgr.shape[:2]
    cx, cy, bw, bh = box
    px, py = int(cx * W), int(cy * H)
    x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
    x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), GOLD, 2)
    cv2.drawMarker(bgr, (px, py), GOLD, cv2.MARKER_CROSS, 18, 2)
    label = "%s %d%% (%d, %d)" % (name, round(conf * 100), px, py)
    if idx is not None:
        label = "#%d %s" % (idx, label)
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, tht), base = cv2.getTextSize(label, font, scale, th)
    pad = 4
    lw, lh = tw + 2 * pad, tht + base + 2 * pad
    lx = min(max(0, x1), max(0, W - lw))                      # stay inside the image
    ly = y1 - lh if y1 - lh >= 0 else max(0, min(y1, H - lh))  # above the box, else inside it
    cv2.rectangle(bgr, (lx, ly), (lx + lw, ly + lh), (12, 17, 20), -1)
    cv2.putText(bgr, label, (lx + pad, ly + pad + tht), font, scale, GOLD, th, cv2.LINE_AA)
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
    ap.add_argument("--record", default=None,
                    help="also save the RAW camera frames (no boxes drawn) to this video, e.g. clip.mp4")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU provider exists")
    ap.add_argument("--trt", action="store_true",
                    help="Jetson: run model_fp16.engine (or model_fp32.engine) built by build_trt.sh; "
                         "without it, says why and runs on onnxruntime")
    ap.add_argument("--steady", choices=["auto", "high", "normal", "low", "off"], default="auto",
                    help="box steadiness: high = steadiest, low = follows fastest, off = raw "
                         "(auto = config.json, default high)")
    ap.add_argument("--hold", type=int, default=None, help="frames to keep the last box when the object is briefly lost (default 6)")
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
    # multi-object detector threshold (per object); --min-conf overrides it
    multi_conf = args.min_conf if args.min_conf is not None else float(cfg.get("multi_conf", 0.4))
    if model.task != "detection":
        mode = "classify"
    if mode == "multi" and "output0" not in model.outputs:
        raise SystemExit("--mode multi needs a detection export (output0 missing)")

    tta = args.tta == "on" or (args.tta == "auto" and not is_raspberry_pi() and cfg.get("target") != "pi5")
    tta = tta and mode in ("single", "precise", "multi")
    steady = args.steady if args.steady != "auto" else (cfg.get("steady") or {}).get("preset", "high")
    hold = args.hold if args.hold is not None else int((cfg.get("steady") or {}).get("hold", 6))
    stab = Stabilizer(steady, hold=max(0, hold))
    mstab = MultiSteadier(steady, hold=max(0, hold) + 4)
    print("TESR model | task: %s | mode: %s | runtime: %s" % (model.task, mode, model.provider))
    if mode in ("single", "precise", "multi"):
        print("steady: %s | hold %d frames | mirror-average %s" % (steady, stab.hold, "on" if tta else "off"))
    print("classes: " + ", ".join(model.names))
    if model.proto is not None and mode != "multi":
        print('"none" gates: confidence < %.2f OR similarity < %.2f'
              % (min_conf, args.proto_th if args.proto_th is not None else model.proto["thr"]))

    def run_frame(bgr):
        out = model.run(bgr, tta=tta)
        if mode == "multi":
            # weak detections (>= half the threshold) may only keep an existing object alive
            dets = decode_multi(model, out, multi_conf * 0.5, args.iou)
            H, W = bgr.shape[:2]
            res = []
            for d in mstab.update(dets, start_conf=multi_conf):
                px, py = draw_box(bgr, d["name"], d["conf"], d["box"], d["id"])
                res.append((d["id"], d["name"], round(d["conf"], 3), (px, py)))
            put(bgr, "%d object(s)" % len(res), H - 14, GOLD, 0.6)
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
            mstab.confirm = 1                 # a single photo: show detections straight away
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
    writer = None
    try:
        while True:
            ok, bgr = reader()
            if not ok or bgr is None:
                break
            if bgr.ndim == 3 and bgr.shape[2] == 4:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                             (bgr.shape[1], bgr.shape[0]))
                    print("recording raw frames to " + args.record)
                writer.write(bgr)                   # before anything is drawn
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
        if writer is not None:
            writer.release()
        release()                          # always free the camera
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
