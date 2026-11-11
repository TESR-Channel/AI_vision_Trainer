# Changelog - TESR AI Vision Web Trainer

Current version: **v3.5.2** (2026-11-11). Also shown on the web page (click the version badge).
Generated from `version.js` by `tools/build_edge_files.py` - edit version.js, not this file.

## v3.5.2 - 2026-11-11

- Crops waiting for a class: a 3-line guide that AI #2 only counts — one class named after the small part (e.g. screw), box each one that is there, empty hole = no box, OK / NG = the count in ROI Studio
- Naming that class ok / ng / no_screw while crops are waiting shows a warning and asks for the part name instead

- รูปที่ตัดรอใส่คลาส: บอกเป็น 3 ข้อว่า AI #2 แค่นับ — คลาสเดียวชื่อชิ้นเล็ก (เช่น screw) · ตีกรอบทุกตัวที่มี รูว่างไม่ต้องตี · OK / NG ดูจากจำนวนใน ROI Studio
- ตั้งชื่อคลาสเป็น ok / ng / no_screw ตอนมีรูปที่ตัดรออยู่ จะเตือนและให้ตั้งเป็นชื่อชิ้นส่วนแทน

## v3.5.1 - 2026-11-11

- ✂ Crop boxes: with no boxes yet it says which class has how many photos, what to do, and offers ✏ Draw boxes now
- Classes split into ok / ng get a tip to use ONE class named after the object (AI #1 only finds it), with a merge button
- The class list shows boxes / photos and picks the class with the most boxes
- Start a new project from crops keeps the Background photos (untick to drop) - AI #2 can train on the web straight away

- ✂ ตัดรูปตามกรอบ: ถ้ายังไม่มีกรอบ บอกชัดว่าคลาสไหนมีกี่รูป ต้องทำอะไร และมีปุ่ม ✏ ไปตีกรอบเลย
- ถ้าแยกคลาส ok / ng ไว้ แนะนำให้รวมเป็นคลาสเดียวตามชื่อชิ้นงาน (AI #1 แค่หาชิ้นงาน) และมีปุ่มรวมคลาสให้
- รายการคลาสบอกจำนวนกรอบ / รูป เลือกคลาสที่มีกรอบมากที่สุดให้เอง
- เริ่มโปรเจกต์ใหม่จากรูปที่ตัด: เอารูป Background ไปด้วย (ติ๊กออกได้) — AI #2 เทรนบนเว็บได้ทันที

## v3.5.0 - 2026-11-11

- 14×14 detection grid for small, close parts (screw heads, pins): Auto / 7×7 / 14×14 next to Train Now - Auto picks 14×14 when the boxes are small or many per photo. On 7×7, neighbouring screw heads fell in one cell and were under-counted
- model.onnx keeps its 224 input and output0 layout, so ROI Studio, the runner and predict.py use it unchanged
- Send to ROI Studio names the file after the class (e.g. terminal.onnx, screw.onnx), so the two models of a two-stage QC never clash

- ตารางตรวจจับ 14×14 สำหรับชิ้นเล็กที่อยู่ชิดกัน (หัวน็อต ขาพิน): เลือก Auto / 7×7 / 14×14 ข้างปุ่ม Train Now — Auto ใช้ 14×14 เมื่อกรอบเล็กหรือมีหลายกรอบต่อรูป เดิม 7×7 หัวน็อตที่อยู่ติดกันตกช่องเดียวกันจนนับไม่ครบ
- model.onnx ยังเป็น input 224 และ output0 แบบเดิม ใช้ได้ทันทีใน ROI Studio, runner และ predict.py
- Send to ROI Studio ตั้งชื่อไฟล์ตามคลาส (เช่น terminal.onnx, screw.onnx) ส่ง 2 โมเดลสำหรับ QC สองชั้นได้โดยชื่อไม่ชนกัน

## v3.4.0 - 2026-11-11

- 📦 Photos only: download photos sorted into one folder per class (all or some classes) - no training needed
- Add photos from a zip / class folders / photo files into the current project (nothing replaced), with the recommended file types shown
- One click on capture = one photo right away (hold = burst)
- Warning before closing the tab if photos were not saved

- ปุ่ม 📦 เฉพาะรูป: ดาวน์โหลดรูปแยกโฟลเดอร์ตามคลาส (ทั้งหมดหรือเลือกบางคลาส) โดยไม่ต้องเทรน
- เพิ่มรูปจาก zip / โฟลเดอร์ที่แยกคลาส / ไฟล์รูป เข้าโปรเจกต์ที่มีอยู่ (ไม่ลบของเดิม) พร้อมบอกชนิดไฟล์ที่แนะนำ
- คลิกปุ่มถ่ายครั้งเดียว = ได้ 1 รูปทันที (กดค้าง = ถ่ายต่อเนื่อง)
- เตือนก่อนปิดแท็บ ถ้ามีรูปที่ยังไม่ได้บันทึก

## v3.3.0 - 2026-11-11

- ✂ Crop boxes (next to Download Dataset): cuts every box of one class (e.g. terminal) out as a new photo, with padding (default 10%) - for two-stage QC: AI #1 finds the part, AI #2 finds the screws in it, no Python script needed
- One click starts a new Object Detection project from the crops (with a Save project 1 first button); type the class name, e.g. screw, + Add Class and every crop is in it, ready to box
- Or download the crops as a .zip of plain JPGs, named after the source photo + box number

- ✂ ตัดรูปตามกรอบ (Crop boxes) ข้างปุ่ม Download Dataset: ตัดทุกกรอบของคลาสที่เลือก (เช่น terminal) ออกมาเป็นรูปใหม่ พร้อมขอบเผื่อ (ค่าเริ่มต้น 10%) สำหรับ QC สองขั้น — AI #1 หาชิ้นงาน, AI #2 หาหัวน็อตในชิ้นงาน ไม่ต้องใช้สคริปต์ Python
- เริ่มโปรเจกต์ Object Detection ใหม่จากรูปที่ตัดในคลิกเดียว (มีปุ่มบันทึกโปรเจกต์ 1 ก่อน) แล้วพิมพ์ชื่อคลาส เช่น screw + Add Class รูปทั้งหมดจะเข้าคลาสนั้นพร้อมลากกรอบ
- หรือดาวน์โหลดรูปที่ตัดเป็น .zip (JPG ล้วน) ชื่อไฟล์ = รูปต้นฉบับ + ลำดับกรอบ

## v3.2.0 - 2026-10-28

- Several boxes in one photo: drag on empty space for another box, click a box to select it, Delete removes it, X clears all (e.g. label every screw head on a part)
- Download Dataset (YOLO) writes one label line per box; Save / Load Project keeps every box (old project zips still load)
- The multi-object detector trains on every box of a photo, and generated ✨ photos move all boxes together

- หลายกรอบในรูปเดียว: ลากบนที่ว่างอีกครั้ง = กรอบใหม่, คลิกกรอบเพื่อเลือก, Delete = ลบกรอบที่เลือก, X = ล้างทั้งหมด (เช่น label หัวน็อตทุกตัวบนชิ้นงาน)
- Download Dataset (YOLO) เขียน 1 บรรทัดต่อกรอบ, Save / Load Project เก็บทุกกรอบ (zip โปรเจกต์เก่ายังโหลดได้)
- Multi-object detector เทรนจากทุกกรอบในรูป และรูป ✨ ย้ายทุกกรอบไปพร้อมกัน

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
