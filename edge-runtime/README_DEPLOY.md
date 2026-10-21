# Deploy your TESR web-trained model

This folder is everything the device needs. No TensorFlow, no PyTorch -
only `onnxruntime` + `numpy` + `opencv`.

| File | What it is |
|---|---|
| `model.onnx` | Your model: MobileNetV2 backbone + your trained head(s), input `images [1,3,224,224]` RGB 0..1 |
| `config.json` | Classes, task, camera type, detection mode, "none" gate |
| `predict.py` | Runs the model on a camera, image or video |
| `install_*.sh / .bat` | One-command installers per device |
| `build_trt.sh` | Jetson only: ONNX -> TensorRT engine (FP32 vs FP16 benchmark) |

## 1 - Copy the folder to the device

```bash
# from your PC (replace user@device-ip)
scp -r tesr-model-<target> user@device-ip:~/
```
(or a USB stick). Then on the device: `cd ~/tesr-model-<target>`

## 2 - Install once

| Device | Command |
|---|---|
| Windows PC | double-click `install_windows.bat` |
| Linux / macOS PC | `bash install_linux.sh` |
| Raspberry Pi 5 (64-bit Pi OS) | `bash install_pi.sh` |
| Jetson Orin Nano / NX / AGX | `bash install_jetson.sh` |
| Jetson Nano (JetPack 4, Python 3.6) | `bash install_jetson.sh` - **experimental**: if pip has no onnxruntime for Python 3.6, install the Jetson build of onnxruntime for JetPack 4.6, or use `bash build_trt.sh` |

> Raspberry Pi: never `sudo pip3 install` - Pi OS blocks it (PEP 668).
> The installer makes a venv with `--system-site-packages` so the apt
> OpenCV + Picamera2 are visible inside it.

## 3 - Run

```bash
source venv/bin/activate          # Windows: venv\Scripts\activate
python predict.py                 # camera type + mode from config.json
python predict.py --camera csi    # ribbon camera (Pi: Picamera2 / Jetson: nvarguscamerasrc)
python predict.py --camera usb --source 1
python predict.py --source photo.jpg
python predict.py --headless      # over SSH: prints results + FPS, no window
```

Detection modes (`--mode`):

| Mode | What you get | Notes |
|---|---|---|
| `single` | 1 object: box + center (x,y) px | fastest - one model run per frame |
| `precise` | 1 object, tighter box | 25 crops per frame - slower |
| `multi` | many objects, each with box + center | **beta** - reads the YOLO-style `output0`; boxes are coarse (32-px grid at 224) |

For production-grade multi-object boxes, train the YOLO dataset you can
download from the web trainer with `yolo-trainer/train.py`.

## 4 - Speed

FPS depends on the device, camera and mode - `predict.py` prints the
**measured** FPS; always measure on the real device.

Jetson GPU (course chapter 8 - ONNX -> TensorRT):
```bash
bash build_trt.sh     # builds model_fp32.engine + model_fp16.engine and prints Throughput for each
```

## Troubleshooting

- **Everything is "none"** -> `--min-conf 0.5`, or `--proto-th 0.6`, or `--no-proto`; collect more varied photos
- **Box shakes** -> it is steadied by default (smoothing + mirror-average + hold); steadier: `--smooth 0.25`; less lag for a fast robot: `--smooth 0.7`; raw: `--smooth 1 --tta off --hold 0`
- **Multi mode finds nothing** -> `--min-conf 0.4`; train a `background` class of the empty scene
- **Multi mode shows the same object twice** -> raise overlap merging: `--iou 0.3`
- **CSI on Jetson fails** -> `gst-launch-1.0 nvarguscamerasrc ! fakesink` must work first; use the system OpenCV (has GStreamer), not pip's
- **CSI on Pi fails** -> `rpicam-hello` must work first; `sudo apt install -y python3-picamera2`
- **USB camera will not open** -> `v4l2-ctl --list-devices`, try `--source 1`; close other apps using it

## Use the same model in other tools

- **TESR ROI Studio** - Add a trained model -> `model.onnx`, type = detection
  (or classification), size = 224, classes from `classes.txt`
- **Any YOLO decoder** - `output0` has the Ultralytics layout `[1, 4+nc, N]`
  (cx,cy,w,h in 224-px space + class scores); apply conf filter + NMS
- **TensorRT / OpenVINO / onnxruntime** - standard ONNX opset 13, static shapes
