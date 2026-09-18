# TESR Offline Trainer — production-grade YOLO: train in 4 steps, deploy to Edge in 1 more

เทรน AI Detection **ของจริง** บนเครื่องคุณเอง — YOLO เรียนรู้ตำแหน่งวัตถุโดยตรง
กรอบแม่น หลายวัตถุพร้อมกัน Real-time และ **ไม่ต้องนั่งลากกรอบเอง**:
ถ่ายบนพื้นเรียบ แล้วสคริปต์ตีกรอบให้อัตโนมัติ

Train a **real** YOLO detector on your own machine. Accurate boxes,
multi-object, real-time — and **no manual box drawing**: shoot on a plain
background and Step 2 labels everything automatically.

## Install (once)

Python **3.10–3.12** (TensorFlow ไม่เกี่ยวแล้ว — ตัวนี้ใช้ PyTorch ผ่าน ultralytics):

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

มี GPU NVIDIA = เทรนเร็วมาก · ไม่มีก็เทรนได้ด้วย CPU (ช้ากว่า แต่ได้ผลเท่ากัน)

## เลือกโหมดได้: Detection หรือ Classification

| โหมด | ได้อะไร | คำสั่ง Step 2 |
|---|---|---|
| **Object Detection** (default) | กรอบ + center (x, y) หลายวัตถุพร้อมกัน | `python 2_autolabel.py` |
| **Rotated Boxes (OBB)** | กรอบ**เฉียงหมุนตามวัตถุ** เช่นบอร์ดที่วางเอียง | `python 2_autolabel.py --obb` |
| **Classification** | ชื่อคลาสทั้งเฟรม (ไม่มีกรอบ — ง่ายและเร็วที่สุด) | `python 2_autolabel.py --task classify` |

`3_train.py` และ `4_run.py` **ตรวจโหมดให้อัตโนมัติ** — Detection/OBB ดูจาก `data.yaml`
+ จำนวนตัวเลขใน label (5 = yolov8n, 9 = yolov8n-obb), Classification ดูจากโฟลเดอร์
`train/<class>/` (ใช้ yolov8n-cls) — คำสั่งเทรนและรันเหมือนกันทุกโหมด ไม่ต้องจำอะไรเพิ่ม

`4_run.py` มี**บรรทัดสถานะบนจอตลอดเวลา**: เจอกี่ชิ้น หรือถ้าไม่ผ่านเกณฑ์ จะบอกว่า
candidate ที่ดีที่สุดคืออะไรกี่ % พร้อมแนะ `--conf` ที่ควรลอง — จะไม่มีการ "เงียบ" อีก

## ทางลัด: Label บนเว็บ แล้วมาเทรนที่นี่ (แนะนำ)

ใช้[หน้าเว็บ](https://tesr-channel.github.io/AI_vision_Trainer/)เก็บภาพ (+ ลากกรอบถ้าเป็น
Detection — สะดวกกว่า และไม่ต้องพึ่งพื้นหลังเรียบ) เลือก Task บนเว็บได้ทั้งสองโหมด แล้ว:

1. กดปุ่ม **⬇ Download Dataset (YOLO)** ในหน้าเว็บ (Step 3 · Collect & Label)
2. แตก zip แล้ววางโฟลเดอร์ `dataset/` ไว้ข้างสคริปต์เหล่านี้ (แทน Step 1–2 ด้านล่าง)
   — โครงสร้างใน zip ถูกต้องตามโหมดที่เลือกบนเว็บอยู่แล้ว (หมุนกรอบบนเว็บ = YOLO OBB อัตโนมัติ)
3. `python 3_train.py` → `python 4_run.py` — จบ

## The 4 steps

```bash
python 1_capture.py --name jetson --auto 4   # (ข้ามได้ถ้าใช้ dataset จากเว็บ) ถ่ายรูป ~100 ใบ/คลาส
python 1_capture.py --name pi --auto 4
python 2_autolabel.py                        # ตีกรอบอัตโนมัติ + สร้าง dataset (เติม --review ถ้าอยากตรวจทีละใบ)
python 3_train.py                            # เทรน YOLO (GPU อัตโนมัติถ้ามี)
python 4_run.py                              # กล้องสด: กรอบ + center (x, y)px — กด q ออก
```

## Tips for the auto-labeler

- **พื้นหลังสีเรียบสีเดียว** (เสื่อดำ/กระดาษขาว) — หัวใจของการตีกรอบอัตโนมัติ
- วัตถุ **1 ชิ้นต่อรูป** ตอนถ่ายเก็บ Dataset
- ขยับให้ทั่ว: มุม ขอบ ใกล้ ไกล เอียง — YOLO เสริม flip/scale/mosaic ให้เองตอนเทรน
- รูปที่ตีกรอบไม่ได้จะไปอยู่ `dataset/needs_review/` — ถ่ายใหม่บนพื้นเรียบกว่าเดิม
- ~100 รูป/คลาส ก็เทรนได้ดีแล้ว (มากกว่านั้นยิ่งดี)

## Troubleshooting

| อาการ | ทางแก้ |
|---|---|
| กล้องเปิดไม่ได้ | `--camera 1` / ปิดแอปที่ใช้กล้องอยู่ (เบราว์เซอร์, Zoom) |
| เทรนช้ามาก | ปกติของ CPU — ลด `--epochs 40` หรือใช้เครื่องที่มี GPU |
| กรอบอัตโนมัติเพี้ยน | พื้นหลังลายเกินไป — ใช้พื้นเรียบ แล้วรัน `2_autolabel.py --review` |
| รันแล้วไม่ขึ้นกรอบอะไรเลย | ดูบรรทัดสถานะบนจอ — มักเป็นเพราะ conf ต่ำ: ลอง `--conf 0.25` + เก็บรูปเพิ่ม (40–100/คลาส) ในแสงที่ใช้จริง |
| ตรวจจับพลาดตอนใช้จริง | เก็บรูปเพิ่มในสภาพแสง/ฉากที่ใช้จริง แล้วเทรนซ้ำ |

## Step 5 — Deploy ลง Edge Device (Raspberry Pi / Jetson Orin Nano)

โมเดลที่เทรนได้ (`best.pt` — จากเว็บ dataset หรือถ่ายเองก็ตาม) เอาไปรันบน Edge ได้เลย
สคริปต์ชุดเดียวกันทั้งหมด แค่ก๊อปโฟลเดอร์ `offline/` + `best.pt` ไปที่เครื่อง

### Raspberry Pi 4 / 5 (Raspberry Pi OS Bookworm 64-bit)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # ultralytics + opencv (CPU)

python 4_run.py --conf 0.25              # รันจาก best.pt ได้ทันที (ช้าหน่อย)

# เร่งความเร็ว: แปลงเป็น NCNN (เร็วสุดบน CPU ของ Pi)
python 5_export.py --target pi           # ได้โฟลเดอร์ best_ncnn_model/
python 4_run.py --weights best_ncnn_model --headless
# ยังช้า? ลดขนาด input: python 5_export.py --target pi --imgsz 320
```

- กล้อง **USB webcam ใช้ได้ทันที** (cv2.VideoCapture) — กล้อง CSI ต้องผ่าน
  picamera2/GStreamer ซึ่งยังไม่อยู่ในสคริปต์ชุดนี้
- FPS จริงขึ้นกับรุ่น Pi / ขนาด input / จำนวนคลาส — **วัดบนเครื่องจริงก่อนตัดสินใจ**

### NVIDIA Jetson Orin Nano (JetPack 6)

PyTorch บน Jetson ต้องใช้ wheel ของ NVIDIA (pip ปกติจะได้ตัว CPU) — ติดตั้งตาม
คู่มือ ultralytics สำหรับ Jetson หรือใช้ Docker image ของ ultralytics แล้ว:

```bash
# ⚠️ .engine ผูกกับ GPU ที่ build — ต้องรันคำสั่งนี้ "บน Jetson" เท่านั้น
python 5_export.py --target jetson       # TensorRT FP16 → best.engine
python 4_run.py --weights best.engine    # เร็วขึ้นมาก, คำสั่งรันเหมือนเดิม
python 4_run.py --weights best.engine --headless   # ผ่าน SSH ไม่มีจอ
```

### ใช้ทุกโหมดได้เหมือนกัน

Export/Deploy ใช้ได้ทั้ง detect / OBB / classify — `4_run.py` อ่าน task จากโมเดลเอง
และ `--headless` พิมพ์ผล (`name center=(x, y)px conf=...`) ลง console
พร้อมส่งต่อเข้า MQTT/Node-RED/PLC ได้เลย

> โมเดลจากปุ่ม **Download Model** บนเว็บ (TensorFlow) รันบน Pi ได้แต่ช้า —
> ถ้าเป้าหมายคือ Edge ให้ใช้เส้นทาง **Download Dataset → เทรน YOLO → Step 5** จะเหมาะกว่า

> เว็บเวอร์ชัน (in-browser trainer) ยังเหมาะสำหรับสอนและ POC เร็วๆ —
> แต่เมื่อต้องใช้งานจริง ให้มาทางนี้
