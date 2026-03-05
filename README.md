# Patient Assist System

Patient Assist System (ระบบเกาะติดและช่วยเหลือผู้ป่วยอัจฉริยะ) เป็นระบบแสดงผลและแจ้งเตือนสถานะของผู้ป่วยแบบ Real-time โดยอาศัยเทคโนโลยี Computer Vision ผ่านโมเดล AI ของ MediaPipe ในการตรวจจับและวิเคราะห์ท่าทางของผู้ป่วยผ่านกล้องเว็บแคม

## 🌟 ฟีเจอร์หลัก (Key Features)

1. **การตรวจจับดวงตา (Eye Tracking & SOS)**
   - คำนวณอัตราการกะพริบตา (EAR - Eye Aspect Ratio)
   - แจ้งเตือนเมื่อหลับตาหรือกะพริบตาถี่ผิดปกติ
   - รองรับระบบ **SOS Signal**: กะพริบตา 2 ครั้ง ➜ หยุด ➜ กะพริบตา 3 ครั้ง เพื่อส่งสัญญาณขอความช่วยเหลือ

2. **การตรวจจับมือและนิ้ว (Hand Gesture Detection)**
   - ตรวจจับลักษณะการแบมือ (OPEN) เเละ กำมือ (FIST)
   - แจ้งเตือนฉุกเฉินเมื่อผู้ป่วย "กำมือค้าง" ขอความช่วยเหลือ

3. **การประเมินสรีระและท่านอน (Body Posture Tracking)**
   - จำแนกท่านอนแบบอัตโนมัติ (หงาย - SUPINE, ตะแคงซ้าย - LEFT_SIDE, ตะแคงขวา - RIGHT_SIDE, กึ่งนั่ง - FOWLERS)
   - แถบความคืบหน้าแจ้งเตือนเมื่อผู้ป่วยอยู่ในท่าเดิมนานเกินไป (เช่น 2 ชั่วโมง) เพื่อ **ป้องกันแผลกดทับ (Bedsores)**
   - แจ้งพยาบาลเมื่อถึงเวลาพลิกตัวผู้ป่วย

4. **การจัดการตารางอาหาร (Meal Schedule)**
   - ตั้งเวลาแจ้งเตือนมื้ออาหาร (เช้า, เที่ยง, เย็น)
   - บันทึกประวัติการทานอาหารลงระบบ

5. **ระบบจัดการข้อมูลหลังบ้าน (Database & History Log)**
   - รองรับ Database MySQL (หรือ fallback กลับไปใช้ in-memory state หากไม่ได้ต่อ DB)
   - รวบรวมข้อมูลเหตุการณ์แจ้งเตือน (Alarms) เเละประวัติการพลิกตัว
   - **Export Data**: สามารถ Export ข้อมูล Alarm Events และตารางอาหารออกมาเป็นไฟล์ `CSV`

---

## 🛠️ โครงสร้างของโปรเจค (Project Structure)

```text
patient-assist/
├── app.py                   # ตัวเริ่มจัดการแอปพลิเคชัน (Flask Router / API endpoints)
├── config.py                # ควบคุม Configuration (Environment & MySQL Configs)
├── .env.example             # ตัวอย่างตั้งค่า Environment Variables เช่น DB_HOST, DB_USER
├── core/                    # Core Business Logic
│   ├── detector.py          # คลาสหลักสำหรับประมวลผลวิดีโอ (จัดการ Threading/Frame skip)
│   ├── camera_manager.py    # จัดการการดึงภาพจากกล้อง
│   ├── alarm_manager.py     # จัดการคิวการแจ้งเตือนและ State ของ Alarm
│   ├── position_tracker.py  # ตรวจสอบเวลาท่านอนและเวลาแจ้งเตือนพลิกตัว
│   └── notification_manager.py # จัดการ Log แจ้งเตือนพลิกตัวและมื้ออาหาร
├── detectors/               # คลาสผู้รับผิดชอบการ Detect ด้วย MediaPipe
│   ├── face_detector.py
│   ├── hand_detector.py
│   └── body_detector.py     # จัดหมวดหมู่ท่านอน (KNN หรือ Heuristics)
├── renderers/               # ตัววาด Overlay ลงบน Video Frame
│   ├── eye_renderer.py
│   ├── hand_renderer.py
│   └── body_renderer.py
├── database/                # Repository Pattern จัดการ Entity DB
│   ├── db.py                # สร้าง Connection Session DB
│   ├── models.py            # ตาราง SQLAlchemy (AlarmEventDB, PositionRecordDB, etc.)
│   └── repository.py        # Function Query เพื่อดึงและสร้างข้อมูล
├── static/                  # JavaScript Module, CSS, เสียงแจ้งเตือน
│   ├── css/
│   ├── js/                  # (app.js, alarm.js, camera.js, notifications.js, utils.js)
│   └── audio/
└── templates/               # ไฟล์ HTML หน้าตาของ UI
```

---

## ⚙️ วิธีติดตั้งและใช้งานระบบ (Installation & Setup)

**1. ติดตั้ง Python และสร้าง Virtual Environment**
ควรใช้ Python 3.9+ ขึ้นไป
```bash
python -m venv venv
venv\Scripts\activate   # สำหรับ Windows 
# source venv/bin/activate สำหรับ Mac/Linux
```

**2. ติดตั้ง Dependencies**
```bash
pip install -r requirements.txt
```
*(กรณีที่ยังไม่มีไฟล์ `requirements.txt` จะใช้ตัวหลักดังนี้: `Flask`, `opencv-python`, `mediapipe`, `SQLAlchemy`, `PyMySQL`, `python-dotenv`)*

**3. ตั้งค่า Database (ตัวเลือก - Optional)**
หากต้องการบันทึกประวัติแบบถาวร ให้ตั้งค่า MySQL Server
- Copy `.env.example` เเล้วตั้งชื่อเป็น `.env`
- เข้าไปแก้การเชื่อมต่อ `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` ให้ตรงกับ MySQL ในเครื่อง
- เลือก `DB_ENABLED=True` บน `.env`

*(หมายเหตุ: หากไม่เปิด Database ระบบจะทำงานได้ปกติในรูปแบบ in-memory session - ข้อมูลจะลบเมื่อปิดเซิร์ฟเวอร์)*

**4. สั่งรันแอปพลิเคชัน**
```bash
python app.py
```

**5. ใช้งานผ่าน Web Browser**
เปิด Browser แล้วไปที่ URL: `http://localhost:5000`

---

## 💡 ทริคสำหรับการใช้งาน (Usage Guidelines)

1. **โหมดการตรวจจับ (Detection Modes):**
   - **ตา (EYE)**: โฟกัสเฉพาะเปลือกตา (ประหยัดพลังงานที่สุด)
   - **มือ (HAND)**: โฟกัสเฉพาะฝ่ามือและนิ้ว
   - **ร่างกาย (BODY)**: โฟกัสเฉพาะการหมุนไหล่/สะโพกและท่านอน
   - **รวม (ALL)**: ตรวจจับ 3 ประเภทพร้อมกัน (ระบบจะใช้ ThreadPool และ Frame-Skip เข้ามาช่วยให้แอปทำงานได้ลื่นไหล ไม่หน่วงกล้อง)

2. **UI Controls (Sidebar ซ้ายมือใต้ "การแสดงผล & ข้อมูล"):**
   - สามารถเปิด/ปิด Overlay เส้นกราฟิกบนจอได้อิสระ ช่วยให้พยาบาลไม่รำคาญตามากเกินไปเวลาดูจอภาพสด
   - สามารถกดปุ่ม **"📥 โหลด Alarm"** และ **"📥 โหลดตารางอาหาร"** เพื่อ Export รายงานรูปแบบ `CSV` นำไปเก็บเป็นประวัติให้ครอบครัวหรือแพทย์

---
*Created for Cola Project / Patient Assist Monitoring System.*
