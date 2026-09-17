# TESR Offline Trainer — production-grade YOLO in 4 easy steps

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

## ทางลัด: Label บนเว็บ แล้วมาเทรนที่นี่ (แนะนำ)

ใช้[หน้าเว็บ](https://tesr-channel.github.io/AI_vision_Trainer/)เก็บภาพ + ลากกรอบ
(สะดวกกว่า และไม่ต้องพึ่งพื้นหลังเรียบ) แล้ว:

1. กดปุ่ม **⬇ Download Dataset (YOLO)** ในหน้าเว็บ (Step 3 · Collect & Label)
2. แตก zip แล้ววางโฟลเดอร์ `dataset/` ไว้ข้างสคริปต์เหล่านี้ (แทน Step 1–2 ด้านล่าง)
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
| ตรวจจับพลาดตอนใช้จริง | เก็บรูปเพิ่มในสภาพแสง/ฉากที่ใช้จริง แล้วเทรนซ้ำ |

## Deploy ต่อ

`best.pt` ใช้กับ ultralytics ได้ทุกที่ — Raspberry Pi, Jetson (export TensorRT:
`yolo export model=best.pt format=engine`), หรือเชื่อม MQTT/Node-RED โดยส่งค่า
center จาก `4_run.py`

> เว็บเวอร์ชัน (in-browser trainer) ยังเหมาะสำหรับสอนและ POC เร็วๆ —
> แต่เมื่อต้องใช้งานจริง ให้มาทางนี้
