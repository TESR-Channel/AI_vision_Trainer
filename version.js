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
  "version": "3.4.0",
  "date": "2026-11-11",
  "history": [
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
