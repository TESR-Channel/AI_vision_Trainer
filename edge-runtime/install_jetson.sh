#!/usr/bin/env bash
# ==== TESR web model - installer for NVIDIA Jetson (Orin Nano / Orin NX / AGX Orin) ====
# Uses the system OpenCV (it has GStreamer, which the CSI camera needs; pip's
# opencv-python does not) through a venv with --system-site-packages.
set -e
cd "$(dirname "$0")"
[ -f /etc/nv_tegra_release ] && echo "L4T: $(head -n1 /etc/nv_tegra_release)"
sudo apt update
sudo apt install -y python3-venv python3-pip python3-numpy
python3 -c "import cv2" 2>/dev/null || sudo apt install -y python3-opencv
python3 -m venv --system-site-packages venv
. venv/bin/activate
pip install --upgrade pip
# CPU onnxruntime works everywhere. For the GPU, either:
#   - build a TensorRT engine:  bash build_trt.sh   (course chapter 8), or
#   - install onnxruntime-gpu from the Jetson AI Lab pip index that matches
#     YOUR JetPack/CUDA version, then run:  python predict.py --trt
pip install onnxruntime || {
  echo "pip could not install onnxruntime for this Python ($(python3 -V))."
  echo "Install the Jetson build of onnxruntime for your JetPack, then re-run."
  exit 1; }
python -c "import onnxruntime as o, cv2; print('onnxruntime', o.__version__, o.get_available_providers()); import re; print('OpenCV GStreamer:', bool(re.search(r'GStreamer:\\s+YES', cv2.getBuildInformation())))"
echo
echo "Done! Run:"
echo "  source venv/bin/activate"
echo "  python predict.py --camera csi    # ribbon camera (nvarguscamerasrc)"
echo "  python predict.py --camera usb    # USB webcam"
echo "Optional GPU speed-up:  bash build_trt.sh"
