#!/usr/bin/env bash
# ==== TESR web model - installer for a Linux / macOS PC ====
# Raspberry Pi -> install_pi.sh    Jetson -> install_jetson.sh
set -e
cd "$(dirname "$0")"
python3 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Done! Run:  source venv/bin/activate && python predict.py"
