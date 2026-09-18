"""TESR Offline Trainer - Step 5: Export best.pt for Edge devices.

Pick your target device - the right format is chosen for you:

    python 5_export.py --target pi        # Raspberry Pi  -> NCNN (fastest on Pi CPU)
    python 5_export.py --target jetson    # Jetson        -> TensorRT engine (FP16)
    python 5_export.py --target onnx      # anything else -> ONNX (portable)
    python 5_export.py --sample-only      # no export - just write the sample code for best.pt

Every export also writes SAMPLE CODE next to your model:
    sample_predict.py   predict(frame) -> list of dicts (name, conf, center, box)
                        works for detection, OBB and classification automatically
    SAMPLE_README.md    bilingual (EN/TH) - use it in your own project + MQTT/Node-RED example

IMPORTANT for Jetson: a TensorRT .engine is compiled FOR the exact GPU that
builds it. Copy best.pt to the Jetson first, then run this script THERE.
NCNN and ONNX exports are portable - export on your PC and copy the result.

Speed tip for Raspberry Pi:  --imgsz 320  (smaller input = more FPS,
slightly lower accuracy - test on your real scene).

The exported model runs with the SAME runner:
    python 4_run.py --weights <exported model> [--headless]
"""
import argparse
from pathlib import Path

TARGETS = {
    "pi":     ("ncnn",   "NCNN - fastest on Raspberry Pi CPU (output is a folder)"),
    "jetson": ("engine", "TensorRT FP16 - must be exported ON the Jetson itself"),
    "onnx":   ("onnx",   "ONNX - portable, runs anywhere onnxruntime does"),
    "tflite": ("tflite", "TensorFlow Lite - for TFLite/Coral pipelines"),
}

SAMPLE_CODE = '''"""TESR AI Vision - minimal sample: use your trained model in YOUR OWN code.

Works with every model this toolkit produces (best.pt, best.engine,
best_ncnn_model, .onnx) and every task - detection, rotated boxes (OBB)
and classification - detected automatically from the model itself.

Run the demo:
    python sample_predict.py                 # webcam
    python sample_predict.py photo.jpg       # single image

Use in your own project (this is the point):
    from sample_predict import predict
    results = predict(frame_bgr, conf=0.5)   # frame from cv2 (BGR)

What predict() returns - plain Python you can use directly:
    detection:      [{"name": "pi", "conf": 0.87, "center": (322, 240),
                      "box": (250, 180, 394, 300)}, ...]
    OBB (rotated):  [{"name": "pi", "conf": 0.87, "center": (322, 240),
                      "corners": [(x, y), (x, y), (x, y), (x, y)]}, ...]
    classification: [{"name": "pi", "conf": 0.93}]        # whole frame, one item
"""
import sys

import cv2
from ultralytics import YOLO

WEIGHTS = "__WEIGHTS__"   # written by 5_export.py - change to any other model file

model = YOLO(WEIGHTS)
TASK = model.task if getattr(model, "task", None) in ("classify", "obb") else "detect"


def predict(frame_bgr, conf=0.5):
    """One frame (cv2 BGR image) in -> list of result dicts out."""
    r = model(frame_bgr, conf=conf, verbose=False)[0]
    out = []
    if TASK == "classify":
        p = r.probs
        out.append({"name": r.names[int(p.top1)], "conf": float(p.top1conf)})
    elif TASK == "obb":
        if r.obb is not None:
            for i in range(len(r.obb)):
                pts = [(float(x), float(y))
                       for x, y in r.obb.xyxyxyxy[i].reshape(-1, 2).tolist()]
                cx = sum(p[0] for p in pts) / 4.0
                cy = sum(p[1] for p in pts) / 4.0
                out.append({"name": r.names[int(r.obb.cls[i])],
                            "conf": float(r.obb.conf[i]),
                            "center": (int(cx), int(cy)),
                            "corners": pts})
    else:
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append({"name": r.names[int(b.cls[0])],
                        "conf": float(b.conf[0]),
                        "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        "box": (int(x1), int(y1), int(x2), int(y2))})
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    print("Task: %s  |  classes: %s" % (TASK, ", ".join(model.names.values())))

    if not src.isdigit():                       # single image
        frame = cv2.imread(src)
        if frame is None:
            raise SystemExit("Could not read image: " + src)
        for d in predict(frame):
            print(d)
        raise SystemExit(0)

    cap = cv2.VideoCapture(int(src))            # webcam - Ctrl+C to stop
    if not cap.isOpened():
        raise SystemExit("Could not open camera " + src)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for d in predict(frame):
                print(d)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
'''

SAMPLE_README = '''# Use your trained model in your own code — sample_predict.py

*(English first · ภาษาไทยด้านล่าง)*

This file was generated by `5_export.py` and is wired to the model: `__WEIGHTS__`
It supports **every task automatically**: Object Detection / Rotated Boxes (OBB) / Classification.

## 1. Try the demo first

```bash
python sample_predict.py              # webcam — prints results every frame, Ctrl+C to stop
python sample_predict.py photo.jpg    # single image
```

## 2. Use it in your own project (the whole point)

Copy `sample_predict.py` + the model file into your project, then:

```python
import cv2
from sample_predict import predict

frame = cv2.imread("photo.jpg")           # or a frame from your camera
for d in predict(frame, conf=0.5):
    print(d["name"], d["conf"])
    if "center" in d:                      # detect / OBB only
        x, y = d["center"]                 # pixel coords — feed a robot arm / PLC directly
```

Results are plain dicts:

| Task | Structure per item |
|---|---|
| Detection | `{"name", "conf", "center": (x, y), "box": (x1, y1, x2, y2)}` |
| OBB | `{"name", "conf", "center": (x, y), "corners": [(x, y) × 4]}` |
| Classification | `{"name", "conf"}` (whole frame, one item) |

## 3. Publish results to MQTT → Node-RED / PLC

```python
# pip install paho-mqtt
import json
import cv2
import paho.mqtt.client as mqtt
from sample_predict import predict

cli = mqtt.Client()
cli.connect("localhost", 1883)            # change to your MQTT broker's IP

cap = cv2.VideoCapture(0)
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for d in predict(frame, conf=0.5):
            cli.publish("tesr/vision", json.dumps(d))
except KeyboardInterrupt:
    pass
finally:
    cap.release()
```

On the Node-RED side: an **mqtt in** node subscribed to `tesr/vision` + a **json**
node gives you a ready-to-use message in your flow.

## Switching models

Edit the `WEIGHTS = "..."` line in `sample_predict.py` to point at any other file
(`best.pt`, `best.engine`, `best_ncnn_model`, `.onnx` — same code for all of them).

---

# ใช้โมเดลของคุณในโค้ดตัวเอง — sample_predict.py

ไฟล์นี้ถูกสร้างโดย `5_export.py` และผูกกับโมเดล: `__WEIGHTS__`
รองรับ **ทุกโหมดอัตโนมัติ**: Object Detection / Rotated Boxes (OBB) / Classification

## 1. ลองรันดูก่อน

```bash
python sample_predict.py              # webcam — พิมพ์ผลทุกเฟรม, Ctrl+C หยุด
python sample_predict.py photo.jpg    # รูปเดี่ยว
```

## 2. เอาไปใช้ในโปรเจกต์ของคุณ (จุดประสงค์หลัก)

ก๊อป `sample_predict.py` + ไฟล์โมเดลไปไว้ในโปรเจกต์ แล้ว:

```python
import cv2
from sample_predict import predict

frame = cv2.imread("photo.jpg")           # หรือเฟรมจากกล้อง
for d in predict(frame, conf=0.5):
    print(d["name"], d["conf"])
    if "center" in d:                      # detect / OBB เท่านั้น
        x, y = d["center"]                 # พิกัด pixel — ส่งให้แขนกล/PLC ได้เลย
```

ผลลัพธ์เป็น dict ธรรมดา:

| โหมด | โครงสร้างต่อชิ้น |
|---|---|
| Detection | `{"name", "conf", "center": (x, y), "box": (x1, y1, x2, y2)}` |
| OBB | `{"name", "conf", "center": (x, y), "corners": [(x, y) × 4]}` |
| Classification | `{"name", "conf"}` (ทั้งเฟรม 1 รายการ) |

## 3. ส่งผลเข้า MQTT → Node-RED / PLC

โค้ดเดียวกับตัวอย่างภาษาอังกฤษด้านบน — เปลี่ยน `"localhost"` เป็น IP ของ
MQTT broker ของคุณ ฝั่ง Node-RED ใส่ node **mqtt in** subscribe topic
`tesr/vision` + **json** node ก็ได้ข้อความพร้อมใช้ต่อใน flow ทันที

## เปลี่ยนโมเดล

แก้บรรทัด `WEIGHTS = "..."` ใน `sample_predict.py` ให้ชี้ไฟล์ใหม่ได้เลย
(`best.pt`, `best.engine`, `best_ncnn_model`, `.onnx` — โค้ดเดียวกันหมด)
'''


def write_samples(weights_path):
    Path("sample_predict.py").write_text(
        SAMPLE_CODE.replace("__WEIGHTS__", str(weights_path)), encoding="utf-8")
    Path("SAMPLE_README.md").write_text(
        SAMPLE_README.replace("__WEIGHTS__", str(weights_path)), encoding="utf-8")
    print("Sample code written:")
    print("    sample_predict.py   (predict(frame) -> list of dicts; demo included)")
    print("    SAMPLE_README.md    (EN/TH - use in your own code + MQTT/Node-RED example)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--target", choices=sorted(TARGETS), default="pi",
                    help="pi (NCNN) / jetson (TensorRT) / onnx / tflite")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="export input size, e.g. 320 for more FPS on Pi (default: model's own)")
    ap.add_argument("--fp32", action="store_true",
                    help="disable FP16 half precision (engine/tflite)")
    ap.add_argument("--sample-only", action="store_true",
                    help="skip the export - just write sample code for --weights as-is")
    args = ap.parse_args()

    if args.sample_only:
        write_samples(args.weights)
        print("Run the demo:  python sample_predict.py")
        return

    fmt, note = TARGETS[args.target]
    print("Target: %s  |  format: %s\n%s" % (args.target, fmt, note))
    if args.target == "jetson":
        print("\nNOTE: run this ON the Jetson - an .engine built on a PC will NOT load there.\n")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    kw = {"format": fmt}
    if args.imgsz:
        kw["imgsz"] = args.imgsz
    if fmt in ("engine", "tflite") and not args.fp32:
        kw["half"] = True   # FP16 - big speedup on Jetson, same commands afterwards
    out = model.export(**kw)

    print("\nExported: %s" % out)
    write_samples(out)
    print("Run it exactly like best.pt:")
    print("    python 4_run.py --weights %s" % out)
    print("    python 4_run.py --weights %s --headless   # over SSH, no display" % out)
    print("    python sample_predict.py                  # minimal integration demo")
    if args.target == "pi" and not args.imgsz:
        print("Too slow on the Pi? Re-export with:  python 5_export.py --target pi --imgsz 320")


if __name__ == "__main__":
    main()
