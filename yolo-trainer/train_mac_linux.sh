#!/usr/bin/env bash
# ==== TESR YOLO Trainer - one command to train (macOS / Linux / Jetson) ====
#   bash train_mac_linux.sh
# First run: makes its own Python environment (venv) and installs everything (internet needed).
# Next runs: trains right away. Change the 3 settings below if you like.
set -e
cd "$(dirname "$0")"
NAME="${NAME:-__NAME__}"
IMGSZ="${IMGSZ:-__IMGSZ__}"
EPOCHS="${EPOCHS:-__EPOCHS__}"
echo "=================================================="
echo "  TESR YOLO Trainer __VERSION__  -  $NAME"
echo "  image size $IMGSZ  -  $EPOCHS epochs"
echo "=================================================="

JETSON=""; [ -f /etc/nv_tegra_release ] && JETSON=1
PY=""
for v in python3.12 python3.11 python3.13 python3.10 python3.14 python3; do
  if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import sys; sys.exit(0 if (3,8) <= sys.version_info[:2] <= (3,14) else 1)' 2>/dev/null; then
    PY="$v"; break
  fi
done
if [ -z "$PY" ]; then echo "Python 3.10 - 3.14 not found. Install it (python.org / apt) and run again."; exit 1; fi
echo "Python: $PY"

if [ ! -x venv/bin/python ]; then
  echo; echo "[1/3] Making the Python environment venv ..."
  if [ -n "$JETSON" ]; then
    # Jetson: reuse JetPack's CUDA PyTorch from the system (see README, Jetson section)
    "$PY" -m venv --system-site-packages venv || { echo "Run: sudo apt install python3-venv  - then again"; exit 1; }
  else
    "$PY" -m venv venv || { echo "Run: sudo apt install python3-venv  - then again"; exit 1; }
  fi
fi
VPY=venv/bin/python
if [ ! -f venv/tesr_ok.txt ]; then
  echo; echo "[2/3] Installing YOLO - first time only, takes a few minutes ..."
  "$VPY" -m pip install --upgrade pip
  "$VPY" -m pip install -r requirements.txt
  echo ok > venv/tesr_ok.txt
fi
if [ -n "$JETSON" ] && ! "$VPY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "NOTE: PyTorch here has no CUDA - training runs on the Jetson CPU (slow)."
  echo "      For the GPU install JetPack's PyTorch: README.md, Jetson section."
fi
BATCH=16; [ -n "$JETSON" ] && BATCH=8

echo; echo "[3/3] Training $NAME ..."
"$VPY" train.py --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$BATCH" --onnx "$NAME"
echo
echo "=================================================="
echo "  Done!  $NAME.onnx  is in $(pwd)"
echo "  ROI Studio: + Add a trained model - pick $NAME.onnx"
echo "  Test with the camera:"
echo "    source venv/bin/activate"
echo "    python run.py --camera usb --source 0      (Pi / Jetson ribbon camera: --camera csi)"
echo "  1 zip = 1 model. Two AIs (find + count)? Download the other project too."
echo "=================================================="
