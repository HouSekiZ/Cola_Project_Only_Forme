import urllib.request

url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
filename = "hand_landmarker.task"

print("กำลังดาวน์โหลดโมเดล... กรุณารอสักครู่")
urllib.request.urlretrieve(url, filename)
print("ดาวน์โหลดเสร็จสมบูรณ์! คุณจะได้ไฟล์ hand_landmarker.task ของจริงมาใช้แล้วครับ")