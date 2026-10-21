# 🚀 START HERE

```
เว็บ (เทรน)  →  คอมพิวเตอร์ (ทดสอบ + export)  →  Raspberry Pi / Jetson (ใช้งาน)
```

## ขั้นที่ 1 — บนคอมพิวเตอร์: ติดตั้ง + ทดสอบ
```
Windows : ดับเบิลคลิก install_windows.bat
Linux   : bash install_linux.sh

venv\Scripts\activate          (Linux: source venv/bin/activate)
python predict.py              ← เปิดกล้อง เห็นกรอบ + center (x,y) · กด q ออก
```

## ขั้นที่ 2 — บนคอมพิวเตอร์: export ไปอุปกรณ์
```
python export.py --target pi                  # Raspberry Pi 5 + กล้อง USB
python export.py --target pi --camera csi     # Raspberry Pi 5 + กล้องสายแพ
python export.py --target jetson              # Jetson Orin Nano / NX / AGX
python export.py --target jetson --camera csi
```
👉 ได้ไฟล์ `exports/tesr-model-pi5.zip` (หรือ `-jetson.zip`) — มีแค่ที่บอร์ดต้องใช้

## ขั้นที่ 3 — บนบอร์ด: ติดตั้ง + รัน
```
1. ก๊อป zip ไปที่บอร์ด แล้วแตกไฟล์      (USB stick / scp)
2. Pi     : bash install_pi.sh
   Jetson : bash install_jetson.sh
3. source venv/bin/activate
4. python predict.py                         # ไม่มีจอ (SSH): python predict.py --headless
```
Jetson อยากเร็วขึ้น: `bash build_trt.sh` (สร้าง TensorRT บนตัว Jetson เอง · บอกค่า Throughput FP32 vs FP16 ที่วัดได้จริง)

⚠️ Pi: ห้าม `sudo pip3 install` — ใช้ `install_pi.sh` เท่านั้น

---

## ต่อยอด

**ใส่ในโค้ด Python ของคุณเอง** — วาง `predict.py` + `model.onnx` + `config.json` ไว้โฟลเดอร์เดียวกัน:
```python
import cv2
from predict import load_config, Model, decide

cfg = load_config()
model = Model(cfg)                         # โหลดครั้งเดียว นอก loop
frame = cv2.imread("photo.jpg")            # หรือภาพจากกล้อง
out = model.run(frame)
name, conf, _ = decide(model, out, 0.65, None)      # ชื่อคลาส หรือ "none"
if "box" in out:
    cx, cy, w, h = out["box"][0]                    # 0..1 ของภาพ
    print(name, conf, "center px:", int(cx * frame.shape[1]), int(cy * frame.shape[0]))
```
👉 เอา `name` / center ไปสั่งแขนกล, PLC, MQTT → Node-RED ได้เลย

**ตีกรอบ ROI / นับของ** — TESR ROI Studio → **+ เพิ่มโมเดลที่เทรนแล้ว** → `model.onnx` · size **224** · คลาสจาก `classes.txt`

**แก้ / เพิ่มคลาส** — หน้าเว็บ → **Load Project** → เลือก zip นี้ (รูป + label อยู่ใน `project/`) → เทรน → ดาวน์โหลดใหม่

**ต้องการแม่นขึ้นอีก / หลายวัตถุแบบจริงจัง** — หน้าเว็บ → **Download Dataset (YOLO)** → เทรน YOLO
(ขั้นตอนเหมือนกัน: `train.py` → `run.py` → `export.py --target pi|jetson`)

## ปรับเร็ว ๆ

| อาการ | ลอง |
|---|---|
| ขึ้น `none` ตลอด | `python predict.py --min-conf 0.5` หรือ `--no-proto` |
| จับผิดของ | `--min-conf 0.8` · เทรนคลาส `background` (ฉากว่าง) เพิ่ม |
| กรอบตามช้าไป (แขนกล / สายพานเร็ว) | `--steady normal` หรือ `--steady low` (ค่าเริ่มต้น `high` = นิ่งที่สุด) |
| อยากได้หลายวัตถุ | `--mode multi` (beta) |
| กล้องไม่เปิด | `--source 1` · ปิดแท็บเว็บ / Zoom ที่ใช้กล้องอยู่ |

รายละเอียดเต็ม → `README_DEPLOY.md`

---

## English

1. **Computer:** `install_windows.bat` (Linux: `bash install_linux.sh`) → `python predict.py`
2. **Computer:** `python export.py --target pi|jetson [--camera csi]` → `exports/tesr-model-<target>.zip`
3. **Device:** unzip → `bash install_pi.sh` / `bash install_jetson.sh` → `python predict.py` (Jetson speed-up: `bash build_trt.sh`)

Need more accuracy or real multi-object? Web page → **Download Dataset (YOLO)** → same steps with `train.py` / `run.py` / `export.py`.
