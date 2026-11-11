/* TESR AI Vision Web Trainer - version + what's new (single source of truth).

   How to release a new version:
     1. add a new entry at the TOP of `history` (version, date, th, en)
     2. set `version` / `date` to that entry
     3. python tools/build_edge_files.py   (also rewrites CHANGELOG.md)
   Numbering  MAJOR.MINOR.PATCH
     MAJOR - old exports / workflow no longer compatible (re-export or retrain)
     MINOR - new feature, old exports keep working
     PATCH - fixes only
   The number is shown on the page, and stamped into every export
   (config.json, project.json, model.onnx metadata, START_HERE.md).    The data below is strict JSON (double quotes) so tools can read it too. */
window.TESR_APP = {
  "name": "TESR AI Vision Web Trainer",
  "version": "3.7.0",
  "date": "2026-11-12",
  "history": [
    {
      "v": "3.7.0",
      "date": "2026-11-12",
      "th": [
        "คำสั่งรันแบบเดียวกันทุกไฟล์ที่ส่งออก: venv\\Scripts\\activate แล้ว python <ตัวรัน>.py --camera usb --source 0 — predict.py (เทรนบนเว็บ) · run.py (YOLO) · tesr_vision_runner.py (ROI Studio)",
        "ชุดเทรน YOLO ใช้โฟลเดอร์ venv (เดิม .venv) เหมือนไฟล์อื่น ๆ · run.py รับ --camera usb / csi / file / ip และเลือกโมเดลให้เอง (best.pt หรือ ชื่อ_yolo.onnx)",
        "predict.py รับ --camera file / ip ด้วย · README, START_HERE, ไฟล์ติดตั้ง และข้อความบนเว็บใช้คำสั่งรูปแบบเดียวกัน"
      ],
      "en": [
        "One run command for every export: venv\\Scripts\\activate, then python <runner>.py --camera usb --source 0 - predict.py (web-trained), run.py (YOLO), tesr_vision_runner.py (ROI Studio)",
        "The YOLO kit now uses a venv folder (was .venv) like the other exports; run.py takes --camera usb / csi / file / ip and picks the model itself (best.pt or NAME_yolo.onnx)",
        "predict.py also takes --camera file / ip; READMEs, START_HERE, installers and web messages all show the same command"
      ]
    },
    {
      "v": "3.6.2",
      "date": "2026-11-12",
      "th": [
        "💾 Save Project ตั้งชื่อไฟล์ตามคลาส (เช่น tesr-project-screw.zip) แยกโปรเจกต์ 1 / 2 ได้ง่าย",
        "หลัง ✂ เริ่มโปรเจกต์ใหม่: บอกว่าไฟล์โปรเจกต์ 1 (AI #1) อยู่ใน Downloads ชื่ออะไร และต้องเก็บไว้"
      ],
      "en": [
        "💾 Save Project names the file after the class (e.g. tesr-project-screw.zip) so project 1 / 2 are easy to tell apart",
        "After ✂ start a new project: says which file in Downloads holds project 1 (AI #1) and that it is needed later"
      ]
    },
    {
      "v": "3.6.1",
      "date": "2026-11-12",
      "th": [
        "ชุดเทรน YOLO: เหลือไฟล์โมเดลไฟล์เดียว ชื่อ_yolo.onnx (ไม่มี best.onnx ซ้ำ) · yolov8n.pt ตัวตั้งต้นเก็บไว้ใน .venv",
        "บอกชัดว่า 1 zip = 1 โมเดล — งาน AI 2 ชั้นต้องดาวน์โหลดจากทั้ง 2 โปรเจกต์ (START_HERE, ข้อความบนเว็บ, ตอนเทรนเสร็จ)"
      ],
      "en": [
        "YOLO kit: one model file, NAME_yolo.onnx (no duplicate best.onnx); the yolov8n.pt starting weights live inside .venv",
        "Says clearly that 1 zip = 1 model - a two-AI job needs a download from both projects (START_HERE, web message, end of training)"
      ]
    },
    {
      "v": "3.6.0",
      "date": "2026-11-12",
      "th": [
        "Download Dataset (YOLO) = ชุดเทรนคลิกเดียว: แตก zip → ดับเบิลคลิก TRAIN_WINDOWS.bat (Mac / Linux / Jetson: bash train_mac_linux.sh) → สร้าง .venv + ติดตั้งเองครั้งแรก → เทรน → ได้ ชื่อ_yolo.onnx ไปใส่ ROI Studio — ไม่ต้องย้าย dataset ไม่ต้องพิมพ์คำสั่ง",
        "zip ตั้งชื่อตามคลาส (เช่น screw-yolo-train.zip) ขนาดภาพเลือกให้จากรูป (รูปที่ตัด 320 · รูปเต็ม 416) 100 epochs",
        "train.py --onnx NAME: เทรนเสร็จ export ONNX ให้ด้วย · requirements.txt มี onnx + onnxslim แล้ว",
        "กรอบที่หมุนไว้แค่บางกรอบ (เช่น เผลอ scroll ตอนตีกรอบ) export เป็นกรอบตรงที่ครอบมัน — เป็น YOLO OBB เฉพาะเมื่อกรอบส่วนใหญ่หมุน (ROI Studio อ่าน YOLO ปกติ)"
      ],
      "en": [
        "Download Dataset (YOLO) is now a one-click training kit: unzip -> double-click TRAIN_WINDOWS.bat (Mac / Linux / Jetson: bash train_mac_linux.sh) -> makes its own .venv and installs on the first run -> trains -> NAME_yolo.onnx for ROI Studio. No moving the dataset, no typing commands",
        "The zip is named after the class (e.g. screw-yolo-train.zip); image size picked from the photos (crops 320, full photos 416), 100 epochs",
        "train.py --onnx NAME exports ONNX after training; requirements.txt now includes onnx + onnxslim",
        "A few rotated boxes (e.g. a stray scroll while labeling) export as the plain box around them; the dataset becomes YOLO OBB only when most boxes are rotated (ROI Studio reads plain YOLO)"
      ]
    },
    {
      "v": "3.5.3",
      "date": "2026-11-11",
      "th": [
        "หน้าตีกรอบ: ปุ่ม ∅ รูปนี้ไม่มี… → Background (ปุ่ม 0) — รูปที่ไม่มีชิ้นนั้นเลย (เช่น น็อตหายหมด) ย้ายไป Background ไม่ต้องค้างเป็นรูปขอบแดง",
        "Train Now: ถ้า Background มีไม่ถึง 10 รูป สร้างให้อัตโนมัติจากส่วนของรูปที่ไม่มีกรอบ (✨ ลบได้ด้วย Remove generated) — คลาสเดียวก็เทรนได้",
        "ข้อความเตือนตอนเทรนบอกตรง ๆ ว่าต้องทำอะไร"
      ],
      "en": [
        "Label dialog: ∅ None in this photo → Background (key 0) — a photo with none of the part (all screws missing) moves to Background instead of staying red",
        "Train Now: with fewer than 10 Background photos, Background is made automatically from the parts of your photos with no box (✨, Remove generated takes them out) — one part class is enough",
        "The training warning says exactly what to do"
      ]
    },
    {
      "v": "3.5.2",
      "date": "2026-11-11",
      "th": [
        "รูปที่ตัดรอใส่คลาส: บอกเป็น 3 ข้อว่า AI #2 แค่นับ — คลาสเดียวชื่อชิ้นเล็ก (เช่น screw) · ตีกรอบทุกตัวที่มี รูว่างไม่ต้องตี · OK / NG ดูจากจำนวนใน ROI Studio",
        "ตั้งชื่อคลาสเป็น ok / ng / no_screw ตอนมีรูปที่ตัดรออยู่ จะเตือนและให้ตั้งเป็นชื่อชิ้นส่วนแทน"
      ],
      "en": [
        "Crops waiting for a class: a 3-line guide that AI #2 only counts — one class named after the small part (e.g. screw), box each one that is there, empty hole = no box, OK / NG = the count in ROI Studio",
        "Naming that class ok / ng / no_screw while crops are waiting shows a warning and asks for the part name instead"
      ]
    },
    {
      "v": "3.5.1",
      "date": "2026-11-11",
      "th": [
        "✂ ตัดรูปตามกรอบ: ถ้ายังไม่มีกรอบ บอกชัดว่าคลาสไหนมีกี่รูป ต้องทำอะไร และมีปุ่ม ✏ ไปตีกรอบเลย",
        "ถ้าแยกคลาส ok / ng ไว้ แนะนำให้รวมเป็นคลาสเดียวตามชื่อชิ้นงาน (AI #1 แค่หาชิ้นงาน) และมีปุ่มรวมคลาสให้",
        "รายการคลาสบอกจำนวนกรอบ / รูป เลือกคลาสที่มีกรอบมากที่สุดให้เอง",
        "เริ่มโปรเจกต์ใหม่จากรูปที่ตัด: เอารูป Background ไปด้วย (ติ๊กออกได้) — AI #2 เทรนบนเว็บได้ทันที"
      ],
      "en": [
        "✂ Crop boxes: with no boxes yet it says which class has how many photos, what to do, and offers ✏ Draw boxes now",
        "Classes split into ok / ng get a tip to use ONE class named after the object (AI #1 only finds it), with a merge button",
        "The class list shows boxes / photos and picks the class with the most boxes",
        "Start a new project from crops keeps the Background photos (untick to drop) - AI #2 can train on the web straight away"
      ]
    },
    {
      "v": "3.5.0",
      "date": "2026-11-11",
      "th": [
        "ตารางตรวจจับ 14×14 สำหรับชิ้นเล็กที่อยู่ชิดกัน (หัวน็อต ขาพิน): เลือก Auto / 7×7 / 14×14 ข้างปุ่ม Train Now — Auto ใช้ 14×14 เมื่อกรอบเล็กหรือมีหลายกรอบต่อรูป เดิม 7×7 หัวน็อตที่อยู่ติดกันตกช่องเดียวกันจนนับไม่ครบ",
        "model.onnx ยังเป็น input 224 และ output0 แบบเดิม ใช้ได้ทันทีใน ROI Studio, runner และ predict.py",
        "Send to ROI Studio ตั้งชื่อไฟล์ตามคลาส (เช่น terminal.onnx, screw.onnx) ส่ง 2 โมเดลสำหรับ QC สองชั้นได้โดยชื่อไม่ชนกัน"
      ],
      "en": [
        "14×14 detection grid for small, close parts (screw heads, pins): Auto / 7×7 / 14×14 next to Train Now - Auto picks 14×14 when the boxes are small or many per photo. On 7×7, neighbouring screw heads fell in one cell and were under-counted",
        "model.onnx keeps its 224 input and output0 layout, so ROI Studio, the runner and predict.py use it unchanged",
        "Send to ROI Studio names the file after the class (e.g. terminal.onnx, screw.onnx), so the two models of a two-stage QC never clash"
      ]
    },
    {
      "v": "3.4.0",
      "date": "2026-11-11",
      "th": [
        "ปุ่ม 📦 เฉพาะรูป: ดาวน์โหลดรูปแยกโฟลเดอร์ตามคลาส (ทั้งหมดหรือเลือกบางคลาส) โดยไม่ต้องเทรน",
        "เพิ่มรูปจาก zip / โฟลเดอร์ที่แยกคลาส / ไฟล์รูป เข้าโปรเจกต์ที่มีอยู่ (ไม่ลบของเดิม) พร้อมบอกชนิดไฟล์ที่แนะนำ",
        "คลิกปุ่มถ่ายครั้งเดียว = ได้ 1 รูปทันที (กดค้าง = ถ่ายต่อเนื่อง)",
        "เตือนก่อนปิดแท็บ ถ้ามีรูปที่ยังไม่ได้บันทึก"
      ],
      "en": [
        "📦 Photos only: download photos sorted into one folder per class (all or some classes) - no training needed",
        "Add photos from a zip / class folders / photo files into the current project (nothing replaced), with the recommended file types shown",
        "One click on capture = one photo right away (hold = burst)",
        "Warning before closing the tab if photos were not saved"
      ]
    },
    {
      "v": "3.3.0",
      "date": "2026-11-11",
      "th": [
        "✂ ตัดรูปตามกรอบ (Crop boxes) ข้างปุ่ม Download Dataset: ตัดทุกกรอบของคลาสที่เลือก (เช่น terminal) ออกมาเป็นรูปใหม่ พร้อมขอบเผื่อ (ค่าเริ่มต้น 10%) สำหรับ QC สองขั้น — AI #1 หาชิ้นงาน, AI #2 หาหัวน็อตในชิ้นงาน ไม่ต้องใช้สคริปต์ Python",
        "เริ่มโปรเจกต์ Object Detection ใหม่จากรูปที่ตัดในคลิกเดียว (มีปุ่มบันทึกโปรเจกต์ 1 ก่อน) แล้วพิมพ์ชื่อคลาส เช่น screw + Add Class รูปทั้งหมดจะเข้าคลาสนั้นพร้อมลากกรอบ",
        "หรือดาวน์โหลดรูปที่ตัดเป็น .zip (JPG ล้วน) ชื่อไฟล์ = รูปต้นฉบับ + ลำดับกรอบ"
      ],
      "en": [
        "✂ Crop boxes (next to Download Dataset): cuts every box of one class (e.g. terminal) out as a new photo, with padding (default 10%) - for two-stage QC: AI #1 finds the part, AI #2 finds the screws in it, no Python script needed",
        "One click starts a new Object Detection project from the crops (with a Save project 1 first button); type the class name, e.g. screw, + Add Class and every crop is in it, ready to box",
        "Or download the crops as a .zip of plain JPGs, named after the source photo + box number"
      ]
    },
    {
      "v": "3.2.0",
      "date": "2026-10-28",
      "th": [
        "หลายกรอบในรูปเดียว: ลากบนที่ว่างอีกครั้ง = กรอบใหม่, คลิกกรอบเพื่อเลือก, Delete = ลบกรอบที่เลือก, X = ล้างทั้งหมด (เช่น label หัวน็อตทุกตัวบนชิ้นงาน)",
        "Download Dataset (YOLO) เขียน 1 บรรทัดต่อกรอบ, Save / Load Project เก็บทุกกรอบ (zip โปรเจกต์เก่ายังโหลดได้)",
        "Multi-object detector เทรนจากทุกกรอบในรูป และรูป ✨ ย้ายทุกกรอบไปพร้อมกัน"
      ],
      "en": [
        "Several boxes in one photo: drag on empty space for another box, click a box to select it, Delete removes it, X clears all (e.g. label every screw head on a part)",
        "Download Dataset (YOLO) writes one label line per box; Save / Load Project keeps every box (old project zips still load)",
        "The multi-object detector trains on every box of a photo, and generated ✨ photos move all boxes together"
      ]
    },
    {
      "v": "3.1.0",
      "date": "2026-10-26",
      "th": [
        "แสดงเวอร์ชันบนหน้าเว็บ และติดเลขเวอร์ชันไปกับทุกไฟล์ที่ export",
        "ปุ่ม + Background: ถ่ายฉากว่าง / มือ / ของที่ต้องไม่สนใจ โดยไม่ต้องลากกรอบ ช่วยลดการทักมือว่าเป็นวัตถุ",
        "ลดกรอบผี: นับเฉพาะช่องที่แรงที่สุดในบริเวณ, วัตถุใหม่ต้องเห็นติดกัน 3 เฟรม, กรอบความมั่นใจต่ำไม่แสดง",
        "หมายเลขวัตถุเรียง 1, 2, 3 และเทรนกับวัตถุที่วางชิดกัน",
        "predict.py --record บันทึกคลิปภาพดิบ (ไม่มีกรอบ) ไว้ทดสอบ / ทำ ROI"
      ],
      "en": [
        "Version shown on the page and stamped into every export",
        "+ Background: capture the empty scene / hands / things to ignore, no boxes needed - fewer boxes on hands",
        "Fewer ghost boxes: local-peak filter, new objects need 3 frames, low-confidence boxes are never shown",
        "Object numbers stay 1, 2, 3; detector trained with touching objects",
        "predict.py --record saves a raw clip (no overlay) for testing / ROI work"
      ]
    },
    {
      "v": "3.0.0",
      "date": "2026-10-22",
      "th": [
        "ตรวจจับได้ทุกชิ้นในภาพ (multi-object) เป็นค่าเริ่มต้น แต่ละชิ้นได้กรอบ, center (x, y) และหมายเลข",
        "ขั้นตอนเดียว: Download Model → คอม (predict.py, export.py --target pi|jetson) → บอร์ด",
        "ป้ายแสดง \"#1 Orange 100% (x, y)\" ฟอนต์เล็ก ไม่หลุดขอบภาพ"
      ],
      "en": [
        "Every object in view detected by default, each with box, center (x, y) and id",
        "One deploy flow: Download Model -> computer (predict.py, export.py --target pi|jetson) -> board",
        "Labels show \"#1 Orange 100% (x, y)\", small and always inside the frame"
      ]
    },
    {
      "v": "2.3.0",
      "date": "2026-10-21",
      "th": [
        "กรอบแนบวัตถุขึ้น (โมเดลกรอบแบบ spatial + รูปที่เลื่อนตำแหน่งอัตโนมัติ)",
        "กรอบนิ่งขึ้นมาก (One Euro filter + เฉลี่ยกับภาพกลับซ้าย-ขวา) ตั้งเป็นค่าเริ่มต้น"
      ],
      "en": [
        "Tighter boxes (spatial box model + automatic position-shifted copies)",
        "Much steadier boxes by default (One Euro filter + mirror average)"
      ]
    },
    {
      "v": "2.2.0",
      "date": "2026-10-21",
      "th": [
        "แก้ model.onnx ที่ตอบ \"none\" ตลอดบนอุปกรณ์ + ตรวจไฟล์ก่อนดาวน์โหลดทุกครั้ง",
        "ทุกไฟล์ export มีโปรเจกต์อยู่ข้างใน — Load Project เปิดกลับได้",
        "START_HERE.md คู่มือทีละเคสในทุก zip"
      ],
      "en": [
        "Fixed model.onnx answering \"none\" on devices; every export is verified before download",
        "Every export contains the project - Load Project opens it",
        "START_HERE.md case-by-case guide in every zip"
      ]
    },
    {
      "v": "2.1.0",
      "date": "2026-10-10",
      "th": [
        "Export เป็น ONNX ได้ในเบราว์เซอร์ (ต่อ TensorRT ได้) และส่งโมเดลไป ROI Studio ได้ในคลิกเดียว"
      ],
      "en": [
        "In-browser ONNX export (TensorRT-ready) and one-click Send to ROI Studio"
      ]
    },
    {
      "v": "2.0.0",
      "date": "2026-09",
      "th": [
        "Object Detection ในเบราว์เซอร์ (ลากกรอบ, กรอบเอียง), ภาษาไทย, Save/Load Project, Dataset YOLO"
      ],
      "en": [
        "In-browser object detection (boxes, rotated boxes), Thai UI, Save/Load Project, YOLO dataset"
      ]
    }
  ]
};
