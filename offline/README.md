# TESR Offline Trainer — from web dataset to a deployed edge model

**One path. No detours.** Collect and label on the [web trainer](https://tesr-channel.github.io/AI_vision_Trainer/),
train and export **on your computer**, then copy the finished model to the edge device.

```text
🌐 Web page              💻 Your computer                      📦 Edge device
collect + label   →   3_train.py → 4_run.py → 5_export.py   →   copy model → run
(Download Dataset)     train        test       package           Raspberry Pi / Jetson
```

*(ภาษาไทยด้านล่าง / Thai version below)*

## Install once (on your computer)

Python **3.10–3.12**:

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

An NVIDIA GPU trains in minutes; CPU also works, just slower — same result.

## The steps

```bash
# 1. On the web page: collect photos, draw boxes (Detection) or skip boxes
#    (Classification), then press "Download Dataset (YOLO)".
# 2. Unzip it and put the "dataset/" folder here, next to 3_train.py.

python 3_train.py                # auto-detects the task -> best.pt
python 4_run.py --conf 0.25      # test with your webcam on this computer
python 5_export.py --target pi   # package for the edge device (run HERE)
```

The task is detected automatically — no flags to remember:

| dataset/ contains | 3_train.py picks | 4_run.py shows |
|---|---|---|
| data.yaml + 5-number labels | yolov8n (detection) | boxes + center (x, y) |
| data.yaml + 9-number labels | yolov8n-obb (rotated) | tilted boxes + center |
| train/\<class\>/ folders | yolov8n-cls (classification) | class name + % |

`4_run.py` shows an **always-on status line**: how many objects passed the
threshold, or the best candidate below it with a suggested `--conf` — a silent
run never happens.

## Deploy to the edge device

**Export on this computer**, then copy the result to the device. `5_export.py`
also writes **`sample_predict.py` + `SAMPLE_README.md` (EN/TH)** next to the
model — a `predict(frame)` function returning plain dicts (name, conf, center,
box/corners) for all tasks, plus an MQTT → Node-RED example.

### Raspberry Pi 4 / 5 (Raspberry Pi OS Bookworm 64-bit) — tested working

**Step 1 — on your computer:**

```bash
python 5_export.py --target pi           # -> best_ncnn_model/ (portable)
```

**Step 2 — copy ONLY these 4 items to the Pi** (e.g. into `/home/pi/offline/`).
Nothing else — you do **not** need `best.pt`, `3_train.py`, `5_export.py` or the
dataset on the Pi:

| Copy this | Why |
|---|---|
| `best_ncnn_model/` (the whole folder) | your exported model |
| `4_run.py` | the live demo |
| `requirements.txt` | installs the dependencies |
| `sample_predict.py` *(optional)* | only if you will write your own code |

**Step 3 — on the Pi, install once** (Raspberry Pi OS blocks `pip` on the
system Python — PEP 668 — so a virtual environment is required, **never
`sudo pip3`**):

```bash
sudo apt update
sudo apt install -y python3-venv python3-full

python3 -m venv --system-site-packages ~/yolo-env
source ~/yolo-env/bin/activate       # prompt now starts with (yolo-env)

python -m pip install --upgrade pip
python -m pip install --no-cache-dir ncnn
python -m pip install --no-cache-dir -r requirements.txt
```

Quick check that you are inside the venv — `which python` must print
`/home/pi/yolo-env/bin/python`.

**Step 4 — run** (every new terminal: activate first):

```bash
source ~/yolo-env/bin/activate
cd ~/offline
python 4_run.py --weights best_ncnn_model
```

A window opens with the box, center crosshair and the status line — press `q`
to quit. Working over SSH with no screen? Add `--headless` and it prints
`name center=(x, y)px conf=...` to the console instead.

USB webcams work out of the box. Too slow? Re-export with `--imgsz 320`.
Real FPS depends on the Pi model, input size and class count — **measure on
the real device before deciding**.

### NVIDIA Jetson Orin Nano (JetPack 6)

```bash
# Copy to the Jetson: best.pt, 4_run.py, sample_predict.py
# On the Jetson (PyTorch must be NVIDIA's wheel or the ultralytics Docker image):
python 4_run.py --weights best.pt --headless    # runs on the GPU directly
```

`--headless` prints `name center=(x, y)px conf=...` to the console over SSH —
ready to pipe into MQTT / Node-RED / a PLC. Works for every task.

> **Optional speed-up (advanced):** on the Jetson itself,
> `python 5_export.py --target jetson` builds a TensorRT FP16 engine.
> A `.engine` only runs on the machine that built it — that is why the main
> path simply copies `best.pt`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `externally-managed-environment` or `ModuleNotFoundError: ncnn` on the Pi | you installed with system Python or `sudo pip3` — redo Step 3 (venv), install without sudo |
| Camera will not open | try `--source 0` / `--source 1`; close apps using it (browser Live Test tab, Zoom) |
| Training is slow | normal on CPU — lower `--epochs 40` or use a machine with an NVIDIA GPU |
| Run shows no boxes at all | read the on-screen status line — usually low conf: try `--conf 0.25` and collect more photos (40–100/class) in the real lighting |
| Wrong detections in the real scene | collect more photos in the real lighting/background, retrain |

---

# TESR Offline Trainer — จาก dataset บนเว็บ สู่โมเดลที่ deploy บน Edge

**ทางเดียว ไม่มีทางแยก** เก็บรูปและ label บน[หน้าเว็บ](https://tesr-channel.github.io/AI_vision_Trainer/)
เทรนและ export **บนคอมพิวเตอร์ของคุณ** แล้วก๊อปโมเดลที่เสร็จแล้วไปที่อุปกรณ์ Edge

```text
🌐 หน้าเว็บ                💻 คอมพิวเตอร์ของคุณ                  📦 Edge device
ถ่าย + label     →   3_train.py → 4_run.py → 5_export.py   →   ก๊อปโมเดล → รัน
(Download Dataset)     เทรน         ทดสอบ       แพ็ก               Raspberry Pi / Jetson
```

## ติดตั้งครั้งเดียว (บนคอมพิวเตอร์)

Python **3.10–3.12** → `python -m venv venv` → activate → `pip install -r requirements.txt`
มี GPU NVIDIA เทรนไม่กี่นาที · CPU ก็ได้ ช้ากว่าแต่ผลเท่ากัน

## ขั้นตอน

```bash
# 1. บนเว็บ: ถ่ายรูป + ลากกรอบ (Detection) หรือไม่ต้องลาก (Classification)
#    แล้วกด "Download Dataset (YOLO)"
# 2. แตก zip วางโฟลเดอร์ dataset/ ไว้ที่นี่ ข้าง 3_train.py

python 3_train.py                # ตรวจโหมดอัตโนมัติ -> best.pt
python 4_run.py --conf 0.25      # ทดสอบด้วย webcam บนคอมพิวเตอร์
python 5_export.py --target pi   # แพ็กสำหรับ Edge (รัน "ที่นี่" บนคอมพิวเตอร์)
```

ตัวกำหนดโหมดคือ Task ที่เลือกบนเว็บก่อนกด Download — `3_train.py` อ่านจากโครงสร้าง
dataset เอง (5 ตัวเลข = detect, 9 = OBB กรอบเอียง, โฟลเดอร์ต่อคลาส = classify)
และ `4_run.py` มีบรรทัดสถานะบนจอตลอด จะไม่มีการรันแบบ "เงียบ"

## Deploy ลง Edge

**Export บนคอมพิวเตอร์** แล้วก๊อปผลลัพธ์ไปที่เครื่อง — `5_export.py` แถม
**`sample_predict.py` + `SAMPLE_README.md` (EN/TH)** ให้ทุกครั้ง: ฟังก์ชัน
`predict(frame)` คืน dict พร้อมใช้ + ตัวอย่าง MQTT → Node-RED

- **Raspberry Pi 4/5 (ทดสอบแล้วใช้ได้จริง):**
  1. บนคอมพิวเตอร์: `python 5_export.py --target pi` → ได้ `best_ncnn_model/`
  2. **ก๊อปไป Pi แค่ 4 อย่างเท่านั้น**: โฟลเดอร์ `best_ncnn_model/` ทั้งโฟลเดอร์,
     `4_run.py`, `requirements.txt` และ `sample_predict.py` (เฉพาะถ้าจะเขียนโค้ดเอง)
     — **ไม่ต้องเอา** `best.pt`, `3_train.py`, `5_export.py` หรือ dataset ไปด้วย
  3. บน Pi (ครั้งแรกครั้งเดียว) — Raspberry Pi OS กันไม่ให้ pip ลง Python ของระบบ
     (PEP 668) ต้องใช้ venv เท่านั้น และ **ห้ามใช้ `sudo pip3` เด็ดขาด**:
     `sudo apt install -y python3-venv python3-full` →
     `python3 -m venv --system-site-packages ~/yolo-env` →
     `source ~/yolo-env/bin/activate` (หน้าจอขึ้น `(yolo-env)` นำหน้า) →
     `python -m pip install --upgrade pip` →
     `python -m pip install --no-cache-dir ncnn` →
     `python -m pip install --no-cache-dir -r requirements.txt`
     (เช็คด้วย `which python` ต้องได้ `/home/pi/yolo-env/bin/python`)
  4. รัน (เปิด Terminal ใหม่ต้อง activate ก่อนทุกครั้ง):
     `source ~/yolo-env/bin/activate` → `cd ~/offline` →
     `python 4_run.py --weights best_ncnn_model` — หน้าต่างโผล่พร้อมกรอบ +
     center (กด `q` เพื่อออก) · ใช้ผ่าน SSH ไม่มีจอ เติม `--headless`
  (กล้อง USB ใช้ได้ทันที · ช้าไปให้ re-export ด้วย `--imgsz 320` · FPS จริงวัดบนเครื่องจริง)
- **Jetson Orin Nano:** ก๊อป `best.pt` + สคริปต์ไปที่ Jetson (PyTorch ต้องเป็น wheel
  ของ NVIDIA หรือ Docker ของ ultralytics) → `python 4_run.py --weights best.pt --headless`
  รันบน GPU ได้เลย
- `--headless` พิมพ์ `name center=(x, y)px conf=...` ผ่าน SSH — ส่งต่อเข้า
  MQTT/Node-RED/PLC ได้ทันที ใช้ได้ทุกโหมด

> **ทางเลือกเร่งความเร็ว (advanced):** บนตัว Jetson เอง รัน
> `python 5_export.py --target jetson` เพื่อ build TensorRT engine —
> `.engine` ใช้ได้เฉพาะเครื่องที่ build เท่านั้น เส้นทางหลักจึงใช้แค่ `best.pt`

## Troubleshooting (ไทย)

| อาการ | ทางแก้ |
|---|---|
| Pi ขึ้น `externally-managed-environment` / หา `ncnn` ไม่เจอ | ไปติดตั้งด้วย Python ระบบหรือ `sudo pip3` — ทำข้อ 3 ใหม่ (venv) และห้ามใช้ sudo |
| กล้องเปิดไม่ได้ | `--source 0` / `--source 1`, ปิดแอปที่ใช้กล้องอยู่ (แท็บ Live Test, Zoom) |
| เทรนช้า | ปกติของ CPU — ลด `--epochs 40` หรือใช้เครื่องที่มี GPU |
| รันแล้วไม่ขึ้นกรอบ | ดูบรรทัดสถานะบนจอ — มักเป็น conf ต่ำ: `--conf 0.25` + เก็บรูปเพิ่ม (40–100/คลาส) ในแสงจริง |
| ใช้จริงแล้วตรวจพลาด | เก็บรูปเพิ่มในแสง/ฉากที่ใช้จริง แล้วเทรนซ้ำ |
