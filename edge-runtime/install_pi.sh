#!/usr/bin/env bash
# ==== TESR web model - installer for Raspberry Pi 5 (Raspberry Pi OS 64-bit) ====
# OpenCV, NumPy and Picamera2 come from apt (Picamera2 cannot be pip-installed);
# the venv sees them through --system-site-packages, pip adds only onnxruntime.
# Never "sudo pip3 install" - Pi OS blocks it (PEP 668) and it can break the system.
set -e
cd "$(dirname "$0")"
if [ "$(uname -m)" != "aarch64" ]; then
  echo "WARNING: this is not a 64-bit OS ($(uname -m)). onnxruntime needs 64-bit Raspberry Pi OS."
fi
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-numpy python3-picamera2
python3 -m venv --system-site-packages venv
. venv/bin/activate
pip install --upgrade pip
pip install onnxruntime
python -c "import onnxruntime, cv2, numpy; print('onnxruntime', onnxruntime.__version__, '| OpenCV', cv2.__version__)"
echo
echo "Done! Run:"
echo "  source venv/bin/activate"
echo "  python predict.py                 # camera type from config.json"
echo "  python predict.py --camera csi    # ribbon camera"
echo "  python predict.py --camera usb    # USB webcam"
