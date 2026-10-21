# 🚀 START HERE

```
เว็บ (เทรน)  →  คอมพิวเตอร์ (ทดสอบ + export)  →  Raspberry Pi / Jetson (ใช้งาน)
```

## ขั้นที่ 1 — บนคอมพิวเตอร์: ติดตั้ง + ทดสอบ
```
Windows : ดับเบิลคลิก install_windows.bat
Linux   : bash install_linux.sh

venv\Scripts\activate          (Linux: source venv/bin/activate)
python predict.py              ← เปิดกล้อง เห็นกรอบ + center (x,y) ทุกชิ้นในภาพ · กด q ออก
```

## ขั้นที่ 2 — บนคอมพิวเตอร์: export ไปอุปกรณ์
```
python export.py --target pi                  # Raspberry Pi 5
python export.py --target jetson              # Jetson Orin Nano / NX / AGX
python export.py --target jetson_nano         # Jetson Nano (JetPack 4, ทดลอง)
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
from predict import load_config, Model, decode_multi

cfg = load_config()
model = Model(cfg)                         # โหลดครั้งเดียว นอก loop
frame = cv2.imread("photo.jpg")            # หรือภาพจากกล้อง
out = model.run(frame)
for d in decode_multi(model, out, 0.3, 0.45):      # ทุกชิ้นในภาพ
    cx, cy, w, h = d["box"]                          # 0..1 ของภาพ
    print(d["name"], d["conf"], "center px:", int(cx * frame.shape[1]), int(cy * frame.shape[0]))
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
| ทักมือ / ของอื่นว่าเป็นวัตถุ | หน้าเว็บ → **+ Background** → ถ่ายฉากว่าง + มือที่ขยับ (ไม่ต้องลากกรอบ) → เทรนใหม่ |
| กรอบตามช้าไป (แขนกล / สายพานเร็ว) | `--steady normal` หรือ `--steady low` (ค่าเริ่มต้น `high` = นิ่งที่สุด) |
| เจอวัตถุไม่ครบ | `--min-conf 0.2` · ถ่ายรูปวัตถุนั้นเพิ่มหลายตำแหน่ง |
| อยากได้แค่ชิ้นหลักชิ้นเดียว | `--mode single` |
| อยากเก็บคลิปภาพจริง (ไม่มีกรอบ) ไว้ทดสอบ / ทำ ROI | `python predict.py --record clip.mp4` |
| กล้องไม่เปิด | `--source 1` · ปิดแท็บเว็บ / Zoom ที่ใช้กล้องอยู่ |

รายละเอียดเต็ม → `README_DEPLOY.md`

---

## English

1. **Computer:** `install_windows.bat` (Linux: `bash install_linux.sh`) → `python predict.py`
2. **Computer:** `python export.py --target pi|jetson|jetson_nano` → `exports/tesr-model-<target>.zip`
3. **Device:** unzip → `bash install_pi.sh` / `bash install_jetson.sh` → `python predict.py` (Jetson speed-up: `bash build_trt.sh`)

Need more accuracy or real multi-object? Web page → **Download Dataset (YOLO)** → same steps with `train.py` / `run.py` / `export.py`.
