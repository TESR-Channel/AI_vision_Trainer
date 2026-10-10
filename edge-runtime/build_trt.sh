#!/usr/bin/env bash
# ==== ONNX -> TensorRT engine on the Jetson (run ON the Jetson) ====
# trtexec ships with JetPack. The engine is tied to THIS device + TensorRT
# version, so always build it on the board that will run it.
set -e
cd "$(dirname "$0")"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -x "$TRTEXEC" ] || TRTEXEC=$(command -v trtexec || true)
if [ -z "$TRTEXEC" ]; then
  echo "trtexec not found - install TensorRT:  sudo apt install -y tensorrt"; exit 1
fi
echo "== FP32 baseline =="
$TRTEXEC --onnx=model.onnx --saveEngine=model_fp32.engine 2>&1 | grep -E "Throughput|mean =|PASSED|FAILED|ERROR" || true
echo "== FP16 (recommended on Jetson) =="
$TRTEXEC --onnx=model.onnx --saveEngine=model_fp16.engine --fp16 2>&1 | grep -E "Throughput|mean =|PASSED|FAILED|ERROR" || true
ls -lh model_*.engine 2>/dev/null || true
echo
echo "Compare the two 'Throughput' lines above - those are YOUR measured numbers."
echo "Run the engine from Python with onnxruntime-gpu + TensorRT provider:  python predict.py --trt"
