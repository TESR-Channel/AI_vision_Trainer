# TESR AI Vision Web Trainer

**Train your own image-recognition AI — right in your browser. Nothing to install.**

Open the page, show your objects to the camera, press Train, and watch the AI
recognize them live — with detection boxes and confidence scores. Everything runs
on **your own computer inside the browser**; your photos never leave your machine.

> 🎓 Built by [TESR — Thai Embedded Systems and Robotics](https://tesrshop.com)
> for makers, students and engineers. *Learn it. Build it. Deploy it. For real.*

---

## ✨ What it does

| | |
|---|---|
| 🧩 **Define classes** | Tell the AI what to tell apart — e.g. `remote` vs `powerbank` |
| 📷 **Collect examples** | Hold the capture button and move the object around, or upload image files — hover any photo to delete it |
| ✨ **Auto augmentation** | One click generates ×2–×5 more images (flips, rotations, lighting) right in the browser |
| 🧠 **Train in seconds** | Transfer learning runs on your GPU via WebGL/WebGPU — typically under 10 seconds |
| 🎯 **Test live** | Two modes: **Classify** the whole frame, or **Detect** — boxes drawn around objects, each labeled with *your* classes |
| 💾 **Export** | One .zip with your model **plus a ready-to-run Python sample** — use it on your PC from an image or webcam |

## 🔒 Privacy by design

```mermaid
flowchart LR
    A[📷 Your camera] --> B[🧠 AI training<br>inside your browser]
    B --> C[🎯 Live predictions<br>on your screen]
    B -.->|nothing is uploaded| X[(☁️ No server)]
```

There is **no backend**. The page is static — all computation (feature extraction,
training, inference) happens in your browser tab using [TensorFlow.js](https://www.tensorflow.org/js).
Close the tab and everything is gone, except the model you chose to download.

## 🚀 Try it

1. Open the page https://tesr-channel.github.io/AI_vision_Trainer/
2. The page checks your device first and tells you honestly whether it can train:
   - 🟢 GPU acceleration found (WebGL/WebGPU) — trains comfortably
   - 🟡 CPU only — works, keep datasets small
   - 🔴 Unsupported browser — it will tell you what to use instead
3. Add **at least 2 classes** → collect **30–50 examples each** → **Train** → point the camera and enjoy

**Tips for good results:** vary the angle, distance, background and lighting while
capturing. Add a "background / none" class so the AI knows what *nothing* looks like.

## 🖥 Requirements

- A recent **Chrome, Edge or Firefox** on a laptop/desktop (works on many phones too)
- A webcam (or use file upload instead)
- Internet connection on first load only (libraries ~10 MB, cached afterwards)

## 🧠 How it works (for the curious)

The page loads **MobileNet**, a pre-trained vision network, and uses it as a
feature extractor. Your examples are converted to compact feature vectors, and a
small neural network head is trained on top of them — that's why training takes
seconds, not hours. In **Detect** mode, **COCO-SSD** (also in-browser) proposes
bounding boxes around objects, and your freshly trained classifier names each box.

```mermaid
flowchart LR
    F[📷 Frame] --> M[MobileNet<br>feature extractor]
    M --> H[Your trained head<br>softmax classifier]
    H --> P[🎯 Class + confidence]
```

## 🐍 Use your model in Python

The **Export** button gives you one `tesr-web-model.zip` containing everything:

```
model.json + weights.bin   your trained model (TensorFlow.js format)
classes.txt                class names, one per line
predict.py                 ready-to-run sample: image file or live webcam
requirements.txt           the Python libraries it needs
README_PYTHON.md           step-by-step instructions
```

Quick start on your computer (Python 3.9–3.11):

```bash
unzip tesr-web-model.zip -d my-model && cd my-model
pip install -r requirements.txt        # just 4 packages: tensorflow, tensorflow_hub, numpy, opencv-python
python predict.py --source photo.jpg   # single image
python predict.py --source 0           # live webcam — press q to quit
```

How it works: the browser trained a small classifier head on top of **MobileNet**
features. `predict.py` rebuilds that same two-stage pipeline in Python — the
MobileNet backbone downloads automatically on first run (~10 MB, cached), and
your head weights are read directly from `weights.bin` with NumPy (no converter
library needed, so installation stays light and conflict-free).
If predictions ever look wrong, run with `--norm pm1` (switches between the two
common MobileNet pixel-scaling conventions).

## 📚 Learn more with TESR Academy

This project is part of TESR's hands-on AI education:
**AI Vision Engineering for Edge AI** — Build, Train & Deploy AI to Edge Devices.
Follow us for workshops, courses and real deployment projects.

## License

MIT © TESR — Thai Embedded Systems and Robotics
