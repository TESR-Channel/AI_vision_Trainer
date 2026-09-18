"""TESR Offline Trainer - Step 5: Export best.pt for Edge devices.

Pick your target device - the right format is chosen for you:

    python 5_export.py --target pi        # Raspberry Pi  -> NCNN (fastest on Pi CPU)
    python 5_export.py --target jetson    # Jetson        -> TensorRT engine (FP16)
    python 5_export.py --target onnx      # anything else -> ONNX (portable)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--target", choices=sorted(TARGETS), default="pi",
                    help="pi (NCNN) / jetson (TensorRT) / onnx / tflite")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="export input size, e.g. 320 for more FPS on Pi (default: model's own)")
    ap.add_argument("--fp32", action="store_true",
                    help="disable FP16 half precision (engine/tflite)")
    args = ap.parse_args()

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
    print("Run it exactly like best.pt:")
    print("    python 4_run.py --weights %s" % out)
    print("    python 4_run.py --weights %s --headless   # over SSH, no display" % out)
    if args.target == "pi" and not args.imgsz:
        print("Too slow on the Pi? Re-export with:  python 5_export.py --target pi --imgsz 320")


if __name__ == "__main__":
    main()
