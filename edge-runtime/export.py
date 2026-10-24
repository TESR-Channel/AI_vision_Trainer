"""TESR web model - package it for a device (run this ON YOUR COMPUTER).

    python export.py --target pi                  # Raspberry Pi 5
    python export.py --target jetson              # Jetson Orin Nano / NX / AGX
    python export.py --target jetson_nano         # old Jetson Nano (JetPack 4, experimental; more limited than Orin Nano)

The export picks the BOARD. Which camera to open is decided on the device,
when you run predict.py there:  python predict.py --camera usb --source 0
(optional --camera usb|csi here only sets the default written into config.json)

Creates  exports/tesr-model-<target>/  and  exports/tesr-model-<target>.zip
containing only what the device needs (model + predict.py + its installer).
Copy that folder/zip to the board, then:  bash install_pi.sh  (or install_jetson.sh)
and  python predict.py

Same model.onnx on every device - nothing is retrained or converted here.
Jetson speed-up (TensorRT) is built ON the Jetson with  bash build_trt.sh
"""
import argparse
import json
import os
import shutil
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "device_kit")

TARGETS = {
    "pi":          {"name": "pi5", "label": "Raspberry Pi 5",
                    "files": ["install_pi.sh"],
                    "install": "bash install_pi.sh"},
    "jetson":      {"name": "jetson", "label": "NVIDIA Jetson Orin Nano / NX / AGX",
                    "files": ["install_jetson.sh", "build_trt.sh"],
                    "install": "bash install_jetson.sh"},
    "jetson_nano": {"name": "jetson_nano", "label": "Jetson Nano (JetPack 4, experimental)",
                    "files": ["install_jetson.sh", "build_trt.sh"],
                    "install": "bash install_jetson.sh"},
    "pc":          {"name": "pc", "label": "PC (Windows / Linux / macOS)",
                    "files": ["install_windows.bat", "install_linux.sh", "requirements.txt"],
                    "install": "install_windows.bat  (Linux: bash install_linux.sh)"},
}
COMMON = ["model.onnx", "classes.txt", "predict.py", "README_DEPLOY.md", "START_HERE.md"]


def find(name):
    for d in (HERE, KIT):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Package this model for a device")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS),
                    help="pi | jetson | jetson_nano | pc")
    ap.add_argument("--camera", choices=["usb", "csi"], default="usb",
                    help="camera on the device (default usb; ribbon camera = csi)")
    args = ap.parse_args()

    T = TARGETS[args.target]
    cfg_path = os.path.join(HERE, "config.json")
    if not os.path.exists(cfg_path):
        sys.exit("config.json not found - run export.py inside the unzipped model folder")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    missing = [n for n in COMMON + T["files"] if not find(n)]
    if missing:
        sys.exit("Missing files: %s\nDownload the model again from the web trainer "
                 "(this folder is from an older version)." % ", ".join(missing))

    out_dir = os.path.join(HERE, "exports", "tesr-model-" + T["name"])
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for n in COMMON + T["files"]:
        shutil.copy2(find(n), os.path.join(out_dir, n))
    cfg["target"] = "pi5" if args.target == "pi" else args.target
    cfg["camera"] = args.camera
    # keep the long prototype vectors on one line, everything else readable
    protos = cfg.pop("prototypes", None)
    text = json.dumps(dict(cfg, prototypes="__P__"), indent=2).replace('"__P__"', json.dumps(protos))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(text)

    zip_path = out_dir + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sorted(os.listdir(out_dir)):
            info = zipfile.ZipInfo("tesr-model-%s/%s" % (T["name"], n), date_time=time.localtime()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if n.endswith(".sh") else 0o644
            info.external_attr = (0o100000 | mode) << 16       # keep .sh executable on Linux
            with open(os.path.join(out_dir, n), "rb") as f:
                z.writestr(info, f.read())

    rel = os.path.relpath(out_dir, HERE)
    print("Done - %s" % T["label"])
    print("  folder : %s" % rel)
    print("  zip    : %s.zip" % rel)
    print("\nNext, ON THE DEVICE:")
    print("  1. copy the zip to the board and unzip it")
    print("  2. %s" % T["install"])
    print("  3. source venv/bin/activate && python predict.py")
    if args.target.startswith("jetson"):
        print("  4. (optional, faster) bash build_trt.sh   -> TensorRT engine built on the Jetson")


if __name__ == "__main__":
    main()
