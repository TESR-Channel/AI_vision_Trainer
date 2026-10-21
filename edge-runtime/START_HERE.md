# 🚀 START HERE — เอาโมเดลไปใช้ต่อ

> ไฟล์สำคัญ: `model.onnx` (โมเดล) · `predict.py` (ตัวรัน) · `config.json` (คลาส, กล้อง, โหมด)
> อยากแก้/เพิ่มคลาส: หน้าเว็บ → **Load Project** → เลือก zip นี้ได้เลย (รูป + label อยู่ในโฟลเดอร์ `project/`)

## เลือกเคสของคุณ 👇

### ① อยากลองบนคอมก่อน (Windows / Linux / Mac)
```
Windows : ดับเบิลคลิก install_windows.bat
Linux   : bash install_linux.sh
รัน     : python predict.py
```
👉 เปิดกล้อง เห็นกรอบ + center (x,y) ทันที · กด `q` ออก

### ② เอาไปรันบน Raspberry Pi 5
```
1. ก๊อปทั้งโฟลเดอร์ไป Pi   (USB / scp)
2. bash install_pi.sh
3. source venv/bin/activate
4. python predict.py --camera usb      # กล้อง USB
   python predict.py --camera csi      # กล้องสายแพ
```
⚠️ ห้าม `sudo pip3 install` — ใช้ตัวติดตั้งนี้เท่านั้น

### ③ เอาไปรันบน Jetson (Orin Nano / NX / AGX)
```
1. ก๊อปทั้งโฟลเดอร์ไป Jetson
2. bash install_jetson.sh
3. source venv/bin/activate
4. python predict.py --camera csi      # หรือ --camera usb
```

### ④ อยากให้ Jetson เร็วขึ้น (ONNX → TensorRT)
```
bash build_trt.sh
```
👉 ได้ `model_fp16.engine` + เห็นตัวเลข **Throughput** FP32 vs FP16 ที่วัดจริงบนบอร์ด

### ⑤ ใช้ผ่าน SSH (ไม่มีจอ)
```
python predict.py --headless
```
👉 พิมพ์ผล + FPS ออกทาง terminal

### ⑥ เอาไปใส่ในโค้ด Python ของคุณเอง
วาง `predict.py` + `model.onnx` + `config.json` ไว้โฟลเดอร์เดียวกับโค้ด แล้ว:
```python
import cv2
from predict import load_config, Model, decide, decode_multi

cfg = load_config()
model = Model(cfg)                      # โหลดครั้งเดียว นอก loop
frame = cv2.imread("photo.jpg")         # หรือภาพจากกล้อง
out = model.run(frame)

name, conf, _ = decide(model, out, 0.65, None)      # ชื่อคลาส / "none"
if "box" in out:                                     # งาน detection
    cx, cy, w, h = out["box"][0]                     # 0..1 ของภาพ
    print(name, conf, "center px:", int(cx * frame.shape[1]), int(cy * frame.shape[0]))
for d in decode_multi(model, out, 0.5, 0.45) if "output0" in out else []:
    print(d["name"], d["conf"], d["box"])           # หลายวัตถุ (beta)
```
👉 เอา `name` / center ไปสั่งแขนกล, PLC, MQTT → Node-RED ต่อได้เลย

### ⑦ อยากตีกรอบ ROI / นับของในพื้นที่
เปิด **TESR ROI Studio** → **+ เพิ่มโมเดลที่เทรนแล้ว** → เลือก `model.onnx`
→ ชนิดตาม `config.json` (detection / classification) · size **224** · ใส่คลาสจาก `classes.txt`

### ⑧ อยากได้แม่นขึ้น / หลายวัตถุแบบจริงจัง
กลับไปหน้าเว็บ → **Download Dataset (YOLO)** → เทรน YOLO บนคอมหรือบน Jetson
(ดู `START_HERE.md` ในไฟล์ dataset)

## ปุ่มปรับเร็ว ๆ

| อาการ | ลอง |
|---|---|
| ขึ้น `none` ตลอด | `--min-conf 0.5` หรือ `--no-proto` |
| จับผิดของ | `--min-conf 0.8` · เทรนคลาส `background` เพิ่ม |
| อยากได้หลายวัตถุ | `--mode multi` |
| อยากได้กรอบแม่นขึ้น (1 วัตถุ) | `--mode precise` (ช้ากว่า) |
| กรอบตามช้าไป (งานแขนกล / สายพานเร็ว) | `--steady normal` หรือ `--steady low` (ค่าเริ่มต้น `high` = นิ่งที่สุด) |
| กล้องไม่เปิด | `--source 1` · ปิดแท็บเว็บ/Zoom ที่ใช้กล้องอยู่ |

รายละเอียดเต็ม → `README_DEPLOY.md`

---

## English quick map

| I want to… | Do this |
|---|---|
| Try on my PC | `install_windows.bat` / `bash install_linux.sh` → `python predict.py` |
| Run on Raspberry Pi 5 | copy folder → `bash install_pi.sh` → `python predict.py --camera usb\|csi` |
| Run on Jetson | copy folder → `bash install_jetson.sh` → `python predict.py --camera csi\|usb` |
| Faster on Jetson | `bash build_trt.sh` (ONNX → TensorRT FP16) |
| No screen (SSH) | `python predict.py --headless` |
| Use in my own code | `from predict import load_config, Model, decide` (example above) |
| Draw ROIs / count | TESR ROI Studio → Add a trained model → `model.onnx`, size 224 |
| More accuracy / real multi-object | web page → Download Dataset (YOLO) → train YOLO |
