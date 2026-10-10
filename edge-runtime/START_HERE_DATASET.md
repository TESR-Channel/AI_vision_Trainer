# 🚀 START HERE — เทรน YOLO จาก dataset นี้

> ในไฟล์นี้มีรูป + label ที่คุณทำบนเว็บ พร้อมเทรน YOLO (แม่นกว่า, หลายวัตถุจริง)
> ตรวจโหมดให้อัตโนมัติ: กรอบปกติ / กรอบเอียง (OBB) / classification

## ขั้นที่ 0 — เตรียม (ครั้งเดียว)
```
1. โหลดตัวเทรน: github.com/TESR-Channel/AI_vision_Trainer → โฟลเดอร์ yolo-trainer
2. วางโฟลเดอร์ "dataset" นี้ไว้ข้าง train.py
```
```
yolo-trainer/
├── train.py  run.py  export.py
└── dataset/   ← โฟลเดอร์นี้
```

## เลือกว่าจะเทรนที่ไหน 👇

### Ⓐ เทรนบนคอม (แนะนำ — มีการ์ดจอ NVIDIA จะเร็วมาก)
```
pip install -r requirements.txt
python train.py            # ได้ best.pt
python run.py              # ทดสอบกับ webcam
```
มีการ์ดจอ NVIDIA → ติดตั้ง PyTorch แบบ CUDA ก่อน (ดู `yolo-trainer/README.md`)
ไม่มีการ์ดจอ → เทรนด้วย CPU ได้ แค่ช้ากว่า
หน่วยความจำไม่พอ → `python train.py --batch 8 --imgsz 480`

### Ⓑ เทรนบน Jetson Orin Nano เลย (ไม่ต้องมีคอมแรง)
```
1. ก๊อป yolo-trainer/ (มี dataset/ อยู่ข้างใน) ไป Jetson
2. ติดตั้ง ultralytics + PyTorch ตามหัวข้อ Jetson ใน yolo-trainer/README.md
3. sudo nvpmodel -m 0 && sudo jetson_clocks      # เร่งเต็มสปีด
4. python3 train.py --batch 8 --imgsz 480         # ได้ best.pt บน Jetson เลย
5. python3 run.py                                  # รันต่อได้ทันที
```
⚠️ ช้ากว่าเทรนบนคอมที่มีการ์ดจอ — เหมาะกับ dataset เล็ก / ไม่มีคอมแรง
⚠️ Raspberry Pi ไม่เหมาะกับการเทรน — เทรนบนคอมแล้วค่อย export ไป Pi

## เทรนเสร็จ → เอาไปใช้ที่ไหน 👇

| ปลายทาง | ทำอะไร |
|---|---|
| 💻 คอมเครื่องเดิม | `python run.py` ใช้ได้เลย |
| 🍓 Raspberry Pi 5 | บนคอม: `python export.py --target pi` → ก๊อป `best_ncnn_model/` + `run.py` ไป Pi |
| 🟩 Jetson | ก๊อป `best.pt` + `run.py` ไปรันได้เลย · อยากเร็วขึ้น: บน Jetson `python3 export.py --target jetson` (TensorRT) |
| 🎯 TESR ROI Studio | `yolo export model=best.pt format=onnx opset=12` → ROI Studio → เพิ่มโมเดล → `best.onnx` |
| 🧩 โค้ดของคุณเอง | ใช้ `sample_predict.py` + `SAMPLE_README.md` ที่ `export.py` สร้างให้ (มีตัวอย่าง MQTT → Node-RED) |

## เพิ่มคลาสใหม่ทีหลัง
```
เว็บ: Load Project → เพิ่มคลาส → ถ่าย + ติด label → Download Dataset (YOLO) ใหม่
คอม:  python train.py --model best.pt       # ต่อยอดจากโมเดลเดิม
```

## หมายเหตุ dataset
- `val/` เป็นรูปจริงเท่านั้น · รูปที่สร้างเพิ่ม (✨) อยู่ใน `train/` → ผลวัดตรงไปตรงมา
- path ใน dataset ถูกแก้ให้อัตโนมัติโดย train.py ย้ายเครื่องได้

---

## English quick map

| I want to… | Do this |
|---|---|
| Set up | put this `dataset/` folder next to `yolo-trainer/train.py` |
| Train on my PC | `pip install -r requirements.txt` → `python train.py` → `python run.py` |
| Train on Jetson Orin Nano itself | install ultralytics + PyTorch (yolo-trainer README, Jetson) → `python3 train.py --batch 8 --imgsz 480` |
| Deploy to Raspberry Pi 5 | on the PC: `python export.py --target pi` → copy `best_ncnn_model/` + `run.py` |
| Deploy to Jetson | copy `best.pt` + `run.py` · faster: `python3 export.py --target jetson` on the Jetson |
| Use in ROI Studio | `yolo export model=best.pt format=onnx opset=12` → Add a trained model |
| Add a class later | web: Load Project → add class → re-download dataset · PC: `python train.py --model best.pt` |
