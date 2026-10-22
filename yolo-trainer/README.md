# TESR YOLO Trainer — the advanced route: train YOLO yourself

This folder is the **YOLO route**, and it is optional. Collect and label on the
[web trainer](https://tesr-channel.github.io/AI_vision_Trainer/), press
**Download Dataset (YOLO)**, then train on your own computer.

```text
🌐 Web page                   💻 Your computer                  📦 Edge device
collect + label        →   train.py → run.py → export.py   →   copy model → run
(Download Dataset YOLO)      train      test     convert        Raspberry Pi / Jetson
```

## Which route am I on?

The web trainer has two exits. They are not the same thing.

| | Browser route (`edge-runtime/`) | YOLO route (this folder) |
|---|---|---|
| Button on the web page | **Download Model (.zip)** | **Download Dataset (YOLO)** |
| Where training happens | in the browser | `train.py` on your computer |
| Model file | `model.onnx` | `best.pt` |
| The device needs | `onnxruntime` only | `ultralytics` + `torch` (or NCNN) |
| Runner | `predict.py` | `run.py --weights …` |
| TESR Academy day 1–2 | **this is the route taught** | optional, for afterwards |

> ⚠️ **Both routes contain a file called `export.py`, and both accept
> `--target pi`.** They do completely different jobs: the browser one
> **packages** `model.onnx` into a zip for the board, this one **converts**
> `best.pt` into NCNN. Run the `export.py` that sits in the folder you are
> working in, and never mix the two sets of output files.

*(ภาษาไทยด้านล่าง / Thai version below)*

## 1 · Train on your computer

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# unzip "Download Dataset (YOLO)" from the web page, put dataset/ next to train.py, then:
python train.py                # auto-detects detect / rotated (OBB) / classify -> best.pt
python run.py --conf 0.25      # test with your webcam - live FPS shows top-right
```

`run.py` always shows a status line (what was found, or the best near-miss with
a suggested `--conf`) — a silent run never happens.

<details><summary><b>⚡ Faster training on an NVIDIA GPU (Windows)</b></summary>

Plain pip installs the **CPU-only** PyTorch. Switch to the CUDA build
(no CUDA Toolkit needed — a recent NVIDIA driver is enough):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"    # must print True
```

`train.py` then uses the GPU automatically — `GPU_mem` in the training log must
not be `0G`. On a 6 GB card, use `--batch 8` or `--imgsz 480` if you hit CUDA
out-of-memory. If cu126 fails to install, try `cu124`.
</details>

## 2 · Deploy to the edge device

| Device | On your computer | On the device | Fastest option |
|---|---|---|---|
| Raspberry Pi 4 / 5 | `python export.py --target pi` | copy `best_ncnn_model/` + run | re-export with `--imgsz 320` |
| Jetson Orin Nano / NX / AGX | nothing — just copy `best.pt` | runs on the GPU as-is | TensorRT (below) |
| Jetson Nano (original) | — | **this route does not run there** | use the browser route |

Every export also writes **`sample_predict.py` + `SAMPLE_README.md`** — a
`predict(frame)` function returning plain dicts, ready for your own code and
MQTT → Node-RED.

### Raspberry Pi 4 / 5 — tested working

```bash
python export.py --target pi           # on your computer -> best_ncnn_model/
```

Copy **only 4 items** to the Pi: the whole `best_ncnn_model/` folder, `run.py`,
`requirements.txt`, and `sample_predict.py` *(optional)*.

<details><summary><b>On the Pi — install once + run</b> (a venv is required, never <code>sudo pip3</code>)</summary>

```bash
sudo apt update && sudo apt install -y python3-venv python3-full
python3 -m venv --system-site-packages ~/yolo-env
source ~/yolo-env/bin/activate        # prompt shows (yolo-env); check: which python
python -m pip install --upgrade pip
python -m pip install --no-cache-dir ncnn
python -m pip install --no-cache-dir -r requirements.txt
```

Run (activate first in every new terminal):

```bash
source ~/yolo-env/bin/activate
cd ~/yolo-trainer
python run.py --weights best_ncnn_model     # add --headless over SSH
```

`externally-managed-environment` or `ModuleNotFoundError: ncnn` means the venv
was skipped or `sudo pip3` was used — redo the install above. Too slow?
Re-export with `--imgsz 320`.
</details>

### NVIDIA Jetson Orin Nano — tested working (JetPack 7)

Copy `best.pt` + `run.py` (+ `export.py` for TensorRT) to the Jetson —
**no export needed**, `best.pt` runs on the Jetson GPU directly. Check your
JetPack first: `cat /etc/nv_tegra_release` → **R39.x = JetPack 7** · R36.x = JetPack 6.

```bash
# JetPack 7 (Ubuntu 24.04 locks pip -> every pip command needs --break-system-packages):
sudo apt update && sudo apt install -y python3-pip
pip install ultralytics --break-system-packages
pip uninstall -y torch torchvision --break-system-packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --break-system-packages
pip install "onnx<2" onnxslim onnxruntime-gpu --break-system-packages    # for the TensorRT export
python3 -c "import torch; print(torch.cuda.is_available())"              # must print True

sudo nvpmodel -m 0 && sudo jetson_clocks    # full speed - run again after every boot
python3 run.py --weights best.pt            # live FPS top-right; --headless over SSH
```

**Fastest on Jetson = TensorRT.** The build takes several minutes — that is
normal and **one-time per model**: keep using `best.pt` while you are still
improving the dataset, build the engine when the model is final
(`--imgsz 320` builds and runs faster, slightly lower accuracy):

```bash
python3 export.py --target jetson      # on the Jetson -> best.engine (FP16)
python3 run.py --weights best.engine   # compare the FPS counter with best.pt
```

<details><summary><b>JetPack 6 install · error fixes</b></summary>

**JetPack 6 (R36.x — Ubuntu 22.04, Python 3.10)** — commands from the
[Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson):

```bash
sudo apt update && sudo apt install -y python3-pip
pip install -U pip
pip install ultralytics
sudo reboot

# pip's torch/torchvision cannot use the Jetson GPU - replace with NVIDIA's wheels:
pip uninstall -y torch torchvision
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl

# dependency fix for torch 2.10 on JetPack 6 ONLY (cuDSS):
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get -y install cudss
```

**Error fixes:**

- a wheel is rejected with *"not a supported wheel on this platform"* — you ran
  the block for the other JetPack; recheck `cat /etc/nv_tegra_release`
- export fails `No module named 'onnx'` (JP7) — AutoUpdate is blocked by
  PEP 668; run the `pip install "onnx<2" ...` line above
- export fails `No module named 'tensorrt'` — `sudo apt install nvidia-jetpack`
</details>

### NVIDIA Jetson Nano (the original board) — use the browser route instead

The original Jetson Nano is much more limited than a Jetson Orin Nano, and the
limit here is software, not speed: its last JetPack is **4.6.x (L4T R32)**,
which is Ubuntu 18.04 with **Python 3.6**. Current `ultralytics` needs Python
3.8 or newer, so `pip install ultralytics` fails before it starts and none of
the blocks above apply.

Two honest options:

1. **Use the browser route** — recommended, and what the course teaches. It
   needs only `onnxruntime` and has a Jetson Nano target of its own:

   ```bash
   python export.py --target jetson_nano      # the export.py in edge-runtime/, not this one
   ```

2. **Stay on YOLO**, export a portable format on your PC, and write your own
   loop on the board:

   ```bash
   python export.py --target onnx             # or --target pi for NCNN
   ```

   `run.py` will not help you here — it imports `ultralytics`, which the board
   cannot install. You would be writing an `onnxruntime` or NCNN loop yourself.

Option 2 has not been tested on a Jetson Nano. Measure it on the board before
promising anyone a frame rate.

<details><summary><b>Troubleshooting (all devices)</b></summary>

| Symptom | Fix |
|---|---|
| Camera will not open | try `--source 0` / `--source 1`; close apps using it (browser Live Test tab, Zoom) |
| Training is slow | normal on CPU — lower `--epochs 40` or use the GPU section above |
| Run shows no boxes at all | read the on-screen status line — usually low conf: try `--conf 0.25` and collect more photos (40–100/class) in the real lighting |
| Wrong detections in the real scene | collect more photos in the real lighting/background, retrain |
| Pi: `externally-managed-environment` / no `ncnn` | venv skipped or `sudo pip3` used — redo the Pi install |
</details>

---

# TESR YOLO Trainer — เส้นทางขั้นสูง: เทรน YOLO ด้วยตัวเอง

โฟลเดอร์นี้คือ **เส้นทาง YOLO** ซึ่งเป็น "ทางเลือก" ไม่ใช่ทางหลักของคอร์ส
เก็บรูป + label บน[หน้าเว็บ](https://tesr-channel.github.io/AI_vision_Trainer/)
→ กด **Download Dataset (YOLO)** → เทรนบนคอมพิวเตอร์ของคุณเอง

## ฉันอยู่เส้นทางไหน

หน้าเว็บมีทางออก 2 ทาง และไม่ใช่เรื่องเดียวกัน

| | เส้นทาง Browser (`edge-runtime/`) | เส้นทาง YOLO (โฟลเดอร์นี้) |
|---|---|---|
| ปุ่มบนหน้าเว็บ | **Download Model (.zip)** | **Download Dataset (YOLO)** |
| เทรนที่ไหน | ในเบราว์เซอร์ | `train.py` บนคอมของคุณ |
| ไฟล์โมเดล | `model.onnx` | `best.pt` |
| บนอุปกรณ์ต้องมี | `onnxruntime` เท่านั้น | `ultralytics` + `torch` (หรือ NCNN) |
| ตัวรัน | `predict.py` | `run.py --weights …` |
| TESR Academy วัน 1–2 | **สอนเส้นทางนี้** | ทางเลือก ไว้ต่อยอดหลังคอร์ส |

> ⚠️ **ทั้งสองเส้นทางมีไฟล์ชื่อ `export.py` และรับ `--target pi` เหมือนกัน**
> แต่ทำงานคนละเรื่อง: ตัวของเส้นทาง browser **แพ็ก** `model.onnx` ลง zip ให้บอร์ด
> ส่วนตัวนี้ **แปลง** `best.pt` เป็น NCNN
> ให้รัน `export.py` ตัวที่อยู่ในโฟลเดอร์ที่กำลังทำงานอยู่ และอย่าสลับไฟล์ผลลัพธ์กัน

## 1 · เทรนบนคอมพิวเตอร์

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# แตก zip จากปุ่ม "Download Dataset (YOLO)" วางโฟลเดอร์ dataset/ ข้าง train.py แล้ว:
python train.py                # ตรวจโหมดอัตโนมัติ -> best.pt
python run.py --conf 0.25      # ทดสอบด้วย webcam - FPS โชว์มุมขวาบน
```

<details><summary><b>⚡ เทรนเร็วขึ้นด้วย GPU NVIDIA (Windows)</b></summary>

pip ปกติติด PyTorch แบบ CPU เท่านั้น ต้องเปลี่ยนเป็นตัว CUDA
(ไม่ต้องลง CUDA Toolkit แยก ขอแค่ NVIDIA driver รุ่นใหม่):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"    # ต้องได้ True
```

แล้ว `train.py` ใช้ GPU อัตโนมัติ (คอลัมน์ `GPU_mem` ใน log ต้องไม่ใช่ `0G`)
· การ์ด 6 GB ถ้า out-of-memory ใช้ `--batch 8` หรือ `--imgsz 480`
</details>

## 2 · Deploy ลง Edge

| อุปกรณ์ | บนคอมพิวเตอร์ | บนอุปกรณ์ | เร็วสุด |
|---|---|---|---|
| Raspberry Pi 4 / 5 | `python export.py --target pi` | ก๊อป `best_ncnn_model/` + รัน | re-export `--imgsz 320` |
| Jetson Orin Nano / NX / AGX | ไม่ต้อง export — ก๊อป `best.pt` | รันบน GPU ได้เลย | TensorRT (ด้านล่าง) |
| Jetson Nano (รุ่นเดิม) | — | **เส้นทางนี้รันไม่ได้** | ใช้เส้นทาง browser |

### Raspberry Pi — ทดสอบแล้วใช้ได้จริง

บนคอม: `python export.py --target pi` → ก๊อปไป Pi **แค่ 4 อย่าง**:
โฟลเดอร์ `best_ncnn_model/` ทั้งโฟลเดอร์, `run.py`, `requirements.txt`, `sample_predict.py` (ถ้าจะเขียนโค้ดเอง)

<details><summary><b>บน Pi — ติดตั้งครั้งเดียว + รัน</b> (ต้องใช้ venv · ห้าม <code>sudo pip3</code> เด็ดขาด)</summary>

```bash
sudo apt update && sudo apt install -y python3-venv python3-full
python3 -m venv --system-site-packages ~/yolo-env
source ~/yolo-env/bin/activate        # หน้าจอขึ้น (yolo-env) นำหน้า
python -m pip install --upgrade pip
python -m pip install --no-cache-dir ncnn
python -m pip install --no-cache-dir -r requirements.txt
```

รัน (เปิด Terminal ใหม่ต้อง activate ก่อนทุกครั้ง):

```bash
source ~/yolo-env/bin/activate
cd ~/yolo-trainer
python run.py --weights best_ncnn_model     # ผ่าน SSH เติม --headless
```

ขึ้น `externally-managed-environment` / หา `ncnn` ไม่เจอ = ข้าม venv หรือใช้
`sudo pip3` — ทำขั้นตอนข้างบนใหม่ · ช้าไปให้ re-export ด้วย `--imgsz 320`
</details>

### Jetson Orin Nano — ทดสอบแล้วใช้ได้จริง (JetPack 7)

ก๊อป `best.pt` + `run.py` (+ `export.py` ถ้าจะทำ TensorRT) — **ไม่ต้อง export**
· เช็ครุ่นก่อน: `cat /etc/nv_tegra_release` → **R39.x = JetPack 7** · R36.x = JetPack 6 (ดูในส่วนพับ)

```bash
# JetPack 7 (Ubuntu 24.04 ล็อก pip -> ทุกคำสั่ง pip ต้องเติม --break-system-packages):
sudo apt update && sudo apt install -y python3-pip
pip install ultralytics --break-system-packages
pip uninstall -y torch torchvision --break-system-packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --break-system-packages
pip install "onnx<2" onnxslim onnxruntime-gpu --break-system-packages    # สำหรับ TensorRT export
python3 -c "import torch; print(torch.cuda.is_available())"              # ต้องได้ True

sudo nvpmodel -m 0 && sudo jetson_clocks    # เร่งเต็มสปีด (รันใหม่หลังบูตทุกครั้ง)
python3 run.py --weights best.pt            # FPS โชว์มุมขวาบน · ผ่าน SSH เติม --headless
```

**เร็วสุดบน Jetson = TensorRT** — build นานหลายนาทีเป็นเรื่องปกติ และทำ
**ครั้งเดียวต่อโมเดล**: ระหว่างยังปรับ dataset ใช้ `best.pt` ไปก่อน ค่อย build
ตอนโมเดลนิ่งแล้ว (`--imgsz 320` build เร็วขึ้น+รันเร็วขึ้น แม่นลดลงเล็กน้อย):

```bash
python3 export.py --target jetson      # บนตัว Jetson -> best.engine (FP16)
python3 run.py --weights best.engine   # เทียบ FPS กับ best.pt ได้เลย
```

### Jetson Nano (รุ่นเดิม) — ให้ใช้เส้นทาง browser แทน

Jetson Nano รุ่นเดิมมีข้อจำกัดมากกว่า Jetson Orin Nano และข้อจำกัดตรงนี้เป็นเรื่อง
**ซอฟต์แวร์ ไม่ใช่ความเร็ว**: JetPack ตัวสุดท้ายของมันคือ **4.6.x (L4T R32)**
ซึ่งเป็น Ubuntu 18.04 + **Python 3.6** แต่ `ultralytics` ปัจจุบันต้องใช้ Python 3.8
ขึ้นไป ฉะนั้น `pip install ultralytics` จะพังตั้งแต่ยังไม่เริ่ม และ block ข้างบน
ทั้งหมดใช้กับบอร์ดนี้ไม่ได้

มี 2 ทางตามจริง

1. **ใช้เส้นทาง browser** — แนะนำ และเป็นทางที่คอร์สสอน เพราะต้องมีแค่
   `onnxruntime` และมี target ของ Jetson Nano อยู่แล้ว

   ```bash
   python export.py --target jetson_nano      # ตัว export.py ใน edge-runtime/ ไม่ใช่ตัวนี้
   ```

2. **อยู่กับ YOLO ต่อ** โดย export เป็นรูปแบบพอร์ตได้บนคอม แล้วเขียน loop เองบนบอร์ด

   ```bash
   python export.py --target onnx             # หรือ --target pi เพื่อได้ NCNN
   ```

   `run.py` ช่วยไม่ได้ เพราะมันเรียก `ultralytics` ที่บอร์ดลงไม่ได้ — ต้องเขียน
   loop ด้วย `onnxruntime` หรือ NCNN เอง

ทางที่ 2 ยังไม่ได้ทดสอบบน Jetson Nano จริง ต้องวัดบนบอร์ดเองก่อนไปบอกใครว่าได้กี่ FPS

<details><summary><b>JetPack 6 · แก้ error</b></summary>

ดู block **JetPack 6 install · error fixes** ในส่วนภาษาอังกฤษด้านบน — สรุป:
JP6 ใช้ torch/torchvision wheel ของ NVIDIA + cuDSS · wheel ขึ้น "not a supported
wheel" = รัน block ผิดรุ่น JetPack · export ขึ้น `No module named 'onnx'` (JP7) =
รันบรรทัด `pip install "onnx<2" ...` ข้างบน · ขึ้น `No module named 'tensorrt'` =
`sudo apt install nvidia-jetpack`
</details>
