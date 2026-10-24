# Deploy your TESR web-trained model

One flow: **web (train) -> computer (test + export) -> Raspberry Pi / Jetson (run)**.
No TensorFlow, no PyTorch - only `onnxruntime` + `numpy` + `opencv`.

| File | What it is |
|---|---|
| `model.onnx` | Your model: MobileNetV2 backbone + your trained head(s), input `images [1,3,224,224]` RGB 0..1 |
| `config.json` | Classes, task, camera type, detection mode, "none" gate, box steadiness |
| `predict.py` | Runs the model on a camera, image or video |
| `export.py` | Packages this model for a device: `python export.py --target pi\|jetson\|jetson_nano` |
| `install_windows.bat / install_linux.sh` | Installers for the computer |
| `device_kit/` | Device installers used by `export.py` (`install_pi.sh`, `install_jetson.sh`, `build_trt.sh`) |
| `project/` | Your photos + labels - the web page's **Load Project** opens this zip |

## 1 - Export for the device (on your computer)

```bash
python export.py --target pi                  # Raspberry Pi 5
python export.py --target jetson              # Jetson Orin Nano / NX / AGX
python export.py --target jetson_nano         # Jetson Nano (JetPack 4, experimental)
```
Result: `exports/tesr-model-<target>.zip` - copy it to the board (USB stick, or
`scp exports/tesr-model-pi5.zip user@device-ip:~/`) and unzip it there.
The model file is the same on every device; export only picks the right
installer. Optional: `--camera csi` (default `usb`) writes the default camera
into `config.json`; on the device `python predict.py --camera usb|csi` always
overrides it.

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

Detection modes (`--mode`, default from `config.json`):

| Mode | What you get | Notes |
|---|---|---|
| `multi` (default) | **every** trained object in view: box + center (x,y) px + track id `#1, #2 ...` | each object is tracked and smoothed on its own |
| `single` | the one main object: box + center | slightly tighter box when only one object is ever in view |
| `precise` | one object, 25-crop grid scan | slower |

Multi-object threshold: `--min-conf 0.3` (default from `config.json` `multi_conf`).
For the most accurate boxes on crowded scenes, train the YOLO dataset from the
web trainer with `yolo-trainer/train.py`.

## 4 - Speed

FPS depends on the device, camera and mode - `predict.py` prints the
**measured** FPS; always measure on the real device.

Jetson GPU (course chapter 6 - ONNX -> TensorRT), on the Jetson itself:
```bash
bash build_trt.sh          # builds model_fp32.engine + model_fp16.engine and prints Throughput for each
python predict.py --trt    # runs model_fp16.engine (model_fp32.engine if there is no FP16 one)
```
`--trt` needs JetPack's TensorRT Python module (`sudo apt install -y python3-libnvinfer`).
The first line then says `runtime: TensorRT engine model_fp16.engine`. If there is
no engine, or it cannot be loaded (built on another board / TensorRT version),
`predict.py` prints why and carries on with onnxruntime - rebuild with `bash build_trt.sh`.

## Troubleshooting

- **Everything is "none"** -> `--min-conf 0.5`, or `--proto-th 0.6`, or `--no-proto`; collect more varied photos
- **Box steadiness** -> default `--steady high` (smoothest: One Euro filter + mirror-average + hold). Fast robot / conveyor: `--steady normal` or `--steady low` (follows faster). Raw: `--steady off --tta off --hold 0`
- **An object is missed** -> lower `--min-conf 0.2`; add photos of it in more positions (also half out of frame)
- **Boxes on hands / things you did not train** -> web page: **+ Background**, capture the empty scene + your hands moving (no boxes), train again; quick fix: `--min-conf 0.5`
- **The same object gets two boxes** -> `--iou 0.3`
- **CSI on Jetson fails** -> `gst-launch-1.0 nvarguscamerasrc ! fakesink` must work first; use the system OpenCV (has GStreamer), not pip's
- **CSI on Pi fails** -> `rpicam-hello` must work first; `sudo apt install -y python3-picamera2`
- **`onnxruntime is not installed`** -> the venv is not active: `source venv/bin/activate` (Windows: `venv\Scripts\activate`), or run the installer again
- **USB camera will not open** -> `v4l2-ctl --list-devices`, try `--source 1`; close other apps using it

## Use the same model in other tools

- **TESR ROI Studio** - Add a trained model -> `model.onnx`, type = detection
  (or classification), size = 224, classes from `classes.txt`
- **Any YOLO decoder** - `output0` has the Ultralytics layout `[1, 4+nc, N]`
  (cx,cy,w,h in 224-px space + class scores); apply conf filter + NMS
- **TensorRT / OpenVINO / onnxruntime** - standard ONNX opset 13, static shapes
