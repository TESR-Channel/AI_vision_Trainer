# 🚀 START HERE — เทรน YOLO: `__NAME__`

> จาก TESR AI Vision Trainer __VERSION__ · โฟลเดอร์นี้มีครบ: รูป + label + ตัวเทรน + สคริปต์ติดตั้ง
> ไม่ต้องย้าย dataset ไปไหน ไม่ต้องพิมพ์คำสั่ง

## 1 · เทรน (คลิกเดียว)

| เครื่อง | ทำ |
|---|---|
| 🪟 Windows | แตก zip (**Extract All**) → ดับเบิลคลิก **`TRAIN_WINDOWS.bat`** |
| 🍎 Mac / 🐧 Linux / 🟩 Jetson | แตก zip → เปิด Terminal ในโฟลเดอร์นี้ → `bash train_mac_linux.sh` |

- ครั้งแรก: สร้าง Python environment ของตัวเอง (`.venv`) + ติดตั้ง YOLO ให้ — ต้องต่อเน็ต ใช้เวลาไม่กี่นาที
  (มีการ์ดจอ NVIDIA บน Windows จะโหลด PyTorch แบบ CUDA ~2.5 GB)
- ครั้งต่อไป: เทรนทันที
- ต้องมี **Python 3.10 – 3.14** (python.org · ตอนติดตั้งติ๊ก **Add python.exe to PATH**) — ถ้าไม่มี สคริปต์จะเปิดหน้าโหลดให้

## 2 · ได้อะไร

> **1 zip = 1 โมเดล** (ของโปรเจกต์ที่กดดาวน์โหลด) · งาน **AI 2 ชั้น** ต้องทำ 2 zip:
> โปรเจกต์ 1 (หาชิ้นงาน เช่น `ssr`) → `ssr_yolo.onnx` · โปรเจกต์ 2 (นับชิ้นเล็ก เช่น `screw`) → `screw_yolo.onnx`
> โปรเจกต์ 1 อยู่ใน `tesr-project-….zip` ที่บันทึกตอนกด ✂ → เว็บ **📂 Load Project** → **Download Dataset (YOLO)**

| ไฟล์ | ใช้ทำ |
|---|---|
| **`__NAME__.onnx`** | **TESR ROI Studio → + เพิ่มโมเดลที่เทรนแล้ว** → เลือกไฟล์นี้ |
| `runs/` | กราฟผลการเทรน (`runs/detect/train*/results.png`) |
| `best.pt` | ทดสอบกล้อง: `.venv\Scripts\python run.py` (Mac/Linux: `.venv/bin/python run.py`) |

ตั้งค่าที่เว็บเลือกให้: ขนาดภาพ **__IMGSZ__** · **__EPOCHS__** epochs — แก้ได้ที่ 3 บรรทัดบนสุดของ `.bat` / `.sh`
หน่วยความจำไม่พอ → ตั้ง `IMGSZ=320`

## 3 · เอาไปใช้ที่อื่น (ทางเลือก)

| ปลายทาง | ทำ |
|---|---|
| 🍓 Raspberry Pi 5 | บนคอมนี้: `.venv\Scripts\python export.py --target pi` → ก๊อป `best_ncnn_model/` + `run.py` ไป Pi |
| 🟩 Jetson | ก๊อป `best.pt` + `run.py` · เร็วขึ้น: บน Jetson `python3 export.py --target jetson` (TensorRT) |
| 🧩 โค้ดของคุณเอง | `export.py` สร้าง `sample_predict.py` + `SAMPLE_README.md` ให้ (มีตัวอย่าง MQTT → Node-RED) |

## เพิ่มคลาส / เพิ่มรูปทีหลัง
เว็บ: **📂 Load Project** เลือก zip นี้ (มีโปรเจกต์อยู่ใน `tesr-project/`) → เพิ่มรูป → **Download Dataset (YOLO)** ใหม่ → ดับเบิลคลิกอีกครั้ง

## ในโฟลเดอร์นี้
```
TRAIN_WINDOWS.bat  train_mac_linux.sh   ← กดตัวนี้
train.py  run.py  export.py  requirements.txt  README.md
dataset/        รูป + label (val เป็นรูปจริงเท่านั้น · รูป ✨ อยู่ใน train)
tesr-project/   ให้เว็บ Load Project กลับไปทำต่อ
```

---

## English

1. **Windows:** Extract All → double-click **`TRAIN_WINDOWS.bat`**. **Mac / Linux / Jetson:** `bash train_mac_linux.sh`.
   The first run makes its own `.venv` and installs YOLO (internet needed). Later runs train straight away. Needs Python 3.10 – 3.14.
2. 1 zip = 1 model (this project only). A two-AI QC job needs two zips: project 1 (find the part) and project 2 (count the small part).
   You get **`__NAME__.onnx`**: in TESR ROI Studio, choose **+ Add a trained model** and pick it. `best.pt` is used by `run.py` for a webcam test.
3. Settings (image size __IMGSZ__, __EPOCHS__ epochs) are the first 3 lines of the `.bat` / `.sh`.
4. Raspberry Pi: `export.py --target pi`. Jetson: `export.py --target jetson` on the Jetson. To add photos later, Load Project this zip on the web.
