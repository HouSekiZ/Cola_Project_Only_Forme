import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1ระบุ Path ของโมเดล
model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')

# 2ตั้งค่า Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, # ตรวจจับมือเดียวเพื่อความแม่นยำในการทำท่าทาง
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

#ฟังก์ชันเช็คว่า "กำมือ" หรือ "แบมือ"
def check_hand_state(landmarks):
    #นิ้วชี้ถึงนิ้วก้อย (จุดปลายคือ 8, 12, 16, 20) เทียบกับข้อต่อ (6, 10, 14, 18)
    finger_tips = [8, 12, 16, 20]
    finger_joints = [6, 10, 14, 18]
    
    fingers_open = []
    for tip, joint in zip(finger_tips, finger_joints):
        #ใน MediaPipe ค่า Y ยิ่งน้อยคือยิ่งสูง (0 อยู่บนสุด)
        if landmarks[tip].y < landmarks[joint].y:
            fingers_open.append(True) # นิ้วเหยียดตรง (แบ)
        else:
            fingers_open.append(False) # นิ้วพับ (กำ)
            
    #ถ้าทุกนิ้วพับหมด = กำมือ, ถ้าเหยียดหมด = แบมือ
    if all(not f for f in fingers_open):
        return "FIST"
    elif all(f for f in fingers_open):
        return "OPEN"
    return "UNKNOWN"

#เริ่มเปิดกล้อง
cap = cv2.VideoCapture(r"D:\Workshop\Project_Cola\20260217_212819.mp4")
last_state = "UNKNOWN"
alert_triggered = False

print("ระบบเริ่มทำงาน... กด 'q' เพื่อปิด")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #สั่งประมวลผล
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for landmarks in detection_result.hand_landmarks:
            # 1วาดจุดบนมือ
            for landmark in landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # 2ตรวจสอบสถานะมือ
            current_state = check_hand_state(landmarks)
            
            #ตรรกะแจ้งเตือน: ถ้าก่อนหน้าแบมือ แล้วตอนนี้กำมือ = สัญญาณขอความช่วยเหลือ
            if last_state == "OPEN" and current_state == "FIST":
                alert_triggered = True
            
            #ถ้าแบมืออีกครั้ง ให้รีเซ็ตการแจ้งเตือน (หรือจะค้างไว้ก็ได้)
            if current_state == "OPEN":
                alert_triggered = False
                
            last_state = current_state

    # 3แสดงผลแจ้งเตือนบนหน้าจอ
    if alert_triggered:
        #วาดกรอบสีแดงแจ้งเตือน
        cv2.rectangle(frame, (0,0), (w, 100), (0, 0, 255), -1)
        cv2.putText(frame, "!!!HELP ME PLS!!!", (50, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        print("ALERT: Help Signal Detected!") # แจ้งเตือนใน Console

    cv2.imshow('Emergency Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()