# Changelog - TESR AI Vision Web Trainer

Current version: **v3.1.0** (2026-10-26). Also shown on the web page (click the version badge).
Generated from `version.js` by `tools/build_edge_files.py` - edit version.js, not this file.

## v3.1.0 - 2026-10-26

- Version shown on the page and stamped into every export
- + Background: capture the empty scene / hands / things to ignore, no boxes needed - fewer boxes on hands
- Fewer ghost boxes: local-peak filter, new objects need 3 frames, low-confidence boxes are never shown
- Object numbers stay 1, 2, 3; detector trained with touching objects
- predict.py --record saves a raw clip (no overlay) for testing / ROI work

- แสดงเวอร์ชันบนหน้าเว็บ และติดเลขเวอร์ชันไปกับทุกไฟล์ที่ export
- ปุ่ม + Background: ถ่ายฉากว่าง / มือ / ของที่ต้องไม่สนใจ โดยไม่ต้องลากกรอบ ช่วยลดการทักมือว่าเป็นวัตถุ
- ลดกรอบผี: นับเฉพาะช่องที่แรงที่สุดในบริเวณ, วัตถุใหม่ต้องเห็นติดกัน 3 เฟรม, กรอบความมั่นใจต่ำไม่แสดง
- หมายเลขวัตถุเรียง 1, 2, 3 และเทรนกับวัตถุที่วางชิดกัน
- predict.py --record บันทึกคลิปภาพดิบ (ไม่มีกรอบ) ไว้ทดสอบ / ทำ ROI

## v3.0.0 - 2026-10-22

- Every object in view detected by default, each with box, center (x, y) and id
- One deploy flow: Download Model -> computer (predict.py, export.py --target pi|jetson) -> board
- Labels show "#1 Orange 100% (x, y)", small and always inside the frame

- ตรวจจับได้ทุกชิ้นในภาพ (multi-object) เป็นค่าเริ่มต้น แต่ละชิ้นได้กรอบ, center (x, y) และหมายเลข
- ขั้นตอนเดียว: Download Model → คอม (predict.py, export.py --target pi|jetson) → บอร์ด
- ป้ายแสดง "#1 Orange 100% (x, y)" ฟอนต์เล็ก ไม่หลุดขอบภาพ

## v2.3.0 - 2026-10-21

- Tighter boxes (spatial box model + automatic position-shifted copies)
- Much steadier boxes by default (One Euro filter + mirror average)

- กรอบแนบวัตถุขึ้น (โมเดลกรอบแบบ spatial + รูปที่เลื่อนตำแหน่งอัตโนมัติ)
- กรอบนิ่งขึ้นมาก (One Euro filter + เฉลี่ยกับภาพกลับซ้าย-ขวา) ตั้งเป็นค่าเริ่มต้น

## v2.2.0 - 2026-10-21

- Fixed model.onnx answering "none" on devices; every export is verified before download
- Every export contains the project - Load Project opens it
- START_HERE.md case-by-case guide in every zip

- แก้ model.onnx ที่ตอบ "none" ตลอดบนอุปกรณ์ + ตรวจไฟล์ก่อนดาวน์โหลดทุกครั้ง
- ทุกไฟล์ export มีโปรเจกต์อยู่ข้างใน — Load Project เปิดกลับได้
- START_HERE.md คู่มือทีละเคสในทุก zip

## v2.1.0 - 2026-10-10

- In-browser ONNX export (TensorRT-ready) and one-click Send to ROI Studio

- Export เป็น ONNX ได้ในเบราว์เซอร์ (ต่อ TensorRT ได้) และส่งโมเดลไป ROI Studio ได้ในคลิกเดียว

## v2.0.0 - 2026-09

- In-browser object detection (boxes, rotated boxes), Thai UI, Save/Load Project, YOLO dataset

- Object Detection ในเบราว์เซอร์ (ลากกรอบ, กรอบเอียง), ภาษาไทย, Save/Load Project, Dataset YOLO
