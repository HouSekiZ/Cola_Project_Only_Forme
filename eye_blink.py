import cv2
import math
import time
import platform
import sys
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =========================================================
# CONFIG  <- แก้ค่าตรงนี้ได้เลย
# =========================================================
VIDEO_SOURCE   = 0   # 0 = webcam
# ── Gaze ──
GAZE_THRESHOLD = 45       # องศา สูงสุดที่ถือว่า "มองกล้อง"

# ── Distance ──
FACE_CLOSE_RATIO  = 0.18  # inter-eye / frame_width ≥ นี้ = "อยู่ใกล้"
FACE_FAR_RATIO    = 0.12  # inter-eye / frame_width ≤ นี้ = "อยู่ไกล"
# ระหว่างสองค่า = hysteresis zone (ไม่เปลี่ยน state)

# ── Blink detection ──
EAR_THRESHOLD  = 0.22     # ต่ำกว่านี้ = ตาปิด
MIN_BLINK_DUR  = 0.08     # วินาที ขั้นต่ำที่ตาต้องปิด (กรอง noise)
MAX_BLINK_DUR  = 0.6      # วินาที ถ้าปิดนานกว่านี้ไม่นับเป็น blink (นับเป็น hold)

# ── SOS pattern (Far mode): 2 blinks พัก 3 blinks ──
SOS_PHASE1     = 2        # กะพริบ phase 1
SOS_PHASE2     = 3        # กะพริบ phase 2
SOS_PAUSE_MIN  = 0.4      # พักระหว่าง phase ≥ วินาที
SOS_PAUSE_MAX  = 3.0      # พักระหว่าง phase ≤ วินาที (เกินนี้ = reset)
SOS_PHASE_WIN  = 4.0      # หน้าต่างเวลาในแต่ละ phase (วินาที)

# ── Near mode: กะพริบ N ครั้งเพื่อ alarm ──
NEAR_BLINK_N   = 3        # จำนวนครั้งที่ต้องกะพริบ
NEAR_BLINK_WIN = 8.0      # หน้าต่างเวลา (วินาที) นับใหม่ถ้าเกิน

# ── Digital zoom ──
ZOOM_FACTOR    = 2.5      # เท่า
ZOOM_DURATION  = 1.5      # วินาที ที่ใช้ zoom in animation

# ── Model ──
MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# ── Landmark indices ──
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
FACE_6PTS  = [1, 152, 263, 33, 287, 57]
# outer eye corners สำหรับวัดระยะห่าง
L_EYE_OUTER = 263
R_EYE_OUTER = 33

MODEL_3D = np.array([
    [ 0.0,     0.0,    0.0  ],
    [ 0.0,   -63.6,  -12.5 ],
    [-43.3,   32.7,  -26.0 ],
    [ 43.3,   32.7,  -26.0 ],
    [-28.9,  -28.9,  -24.1 ],
    [ 28.9,  -28.9,  -24.1 ],
], dtype=np.float64)

# ── States ──
MODE_FAR    = "FAR"
MODE_NEAR   = "NEAR"
MODE_ZOOM   = "ZOOMING"

SOS_IDLE    = "IDLE"      # รอ phase 1
SOS_P1      = "PHASE1"    # กำลังนับ blink phase 1
SOS_PAUSE   = "PAUSE"     # รอ pause ระหว่าง phase
SOS_P2      = "PHASE2"    # กำลังนับ blink phase 2
SOS_DONE    = "DONE"      # ครบ pattern แล้ว


# =========================================================
# AUTO-DOWNLOAD MODEL
# =========================================================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"[INFO] ดาวน์โหลด model (~29MB)...")
    try:
        def progress(c, bs, ts):
            print(f"\r  {c*bs*100//ts}%", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress)
        print(f"\r[INFO] ดาวน์โหลดสำเร็จ")
    except Exception as e:
        print(f"\n[ERROR] {e}\nดาวน์โหลดเอง: {MODEL_URL}\nวางที่: {MODEL_PATH}")
        sys.exit(1)


# =========================================================
# HELPERS
# =========================================================
def get_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def dist2d(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def calculate_ear(lms, indices, w, h):
    pts = np.array([get_px(lms[i], w, h) for i in indices])
    v1  = dist2d(pts[1], pts[5])
    v2  = dist2d(pts[2], pts[4])
    hz  = dist2d(pts[0], pts[3])
    return ((v1 + v2) / (2.0 * hz) if hz > 0 else 0.0), pts

def iris_center(lms, ids, w, h):
    pts = np.array([get_px(lms[i], w, h) for i in ids], dtype=float)
    return pts.mean(axis=0).astype(int)

def estimate_gaze(lms, w, h):
    img_pts = np.array([get_px(lms[i], w, h) for i in FACE_6PTS], dtype=np.float64)
    f = float(w)
    cam = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, False
    rmat, _ = cv2.Rodrigues(rvec)
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
        np.hstack((rmat, np.zeros((3,1)))))
    ph, yh = float(euler[0]), float(euler[1])

    li = iris_center(lms, LEFT_IRIS,  w, h).astype(float)
    ri = iris_center(lms, RIGHT_IRIS, w, h).astype(float)

    def eye_off(ic, eye_ids):
        eps = np.array([get_px(lms[i], w, h) for i in eye_ids], dtype=float)
        ex = (ic[0]-eps[:,0].min()) / max(eps[:,0].max()-eps[:,0].min(),1) - 0.5
        ey = (ic[1]-eps[:,1].min()) / max(eps[:,1].max()-eps[:,1].min(),1) - 0.5
        return ex, ey

    lx, ly = eye_off(li, LEFT_EYE)
    rx, ry = eye_off(ri, RIGHT_EYE)
    S = 40.0
    yaw   = yh + ((lx+rx)/2)*S
    pitch = ph + ((ly+ry)/2)*S
    return yaw, pitch, abs(yaw)<=GAZE_THRESHOLD and abs(pitch)<=GAZE_THRESHOLD

def face_distance_ratio(lms, w, h):
    """inter-eye distance / frame_width  (ใหญ่ = อยู่ใกล้)"""
    lp = get_px(lms[L_EYE_OUTER], w, h)
    rp = get_px(lms[R_EYE_OUTER], w, h)
    return dist2d(lp, rp) / w

def digital_zoom(frame, factor, cx_norm=0.5, cy_norm=0.5):
    """
    Zoom เข้าหาจุด (cx_norm, cy_norm) ซึ่งเป็นตำแหน่ง normalize (0–1)
    ของใบหน้า แล้ว resize กลับขนาดเดิม
    cx_norm, cy_norm = 0.5, 0.5 คือกึ่งกลางภาพ
    """
    h, w = frame.shape[:2]
    new_h = int(h / factor)
    new_w = int(w / factor)

    # จุดศูนย์กลางของ crop (pixel) ยึดตามตำแหน่งใบหน้า
    cx = int(cx_norm * w)
    cy = int(cy_norm * h)

    # คำนวณ crop box พร้อม clamp ไม่ให้เกินขอบภาพ
    x1 = max(0, min(cx - new_w // 2, w - new_w))
    y1 = max(0, min(cy - new_h // 2, h - new_h))

    cropped = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def beep():
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(1000, 500)
        except Exception:
            pass
    else:
        print("\a", end="", flush=True)

def put(frame, text, pos, color=(255,255,255), scale=0.65, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def draw_sos_dots(frame, phase, p1_count, p2_count):
    """วาด dots แสดง SOS progress"""
    x, y = 10, 120
    r = 9
    gap = 24
    # Phase 1 dots
    for i in range(SOS_PHASE1):
        filled = (phase in (SOS_P1, SOS_PAUSE, SOS_P2, SOS_DONE)) and i < p1_count
        col = (0, 220, 255) if filled else (50, 50, 50)
        cv2.circle(frame, (x + i*gap, y), r, col, -1)
        cv2.circle(frame, (x + i*gap, y), r, (180,180,180), 1)
    # separator
    sx = x + SOS_PHASE1*gap + 4
    cv2.putText(frame, "...", (sx, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    # Phase 2 dots
    ox = sx + 28
    for i in range(SOS_PHASE2):
        filled = phase in (SOS_P2, SOS_DONE) and i < p2_count
        col = (0, 100, 255) if filled else (50, 50, 50)
        cv2.circle(frame, (ox + i*gap, y), r, col, -1)
        cv2.circle(frame, (ox + i*gap, y), r, (180,180,180), 1)


# =========================================================
# STATE MACHINE
# =========================================================
class BlinkStateMachine:
    def __init__(self):
        self.reset()

    def reset(self):
        # blink edge detection
        self.eye_was_closed  = False
        self.close_start     = None

        # SOS (FAR mode)
        self.sos_phase       = SOS_IDLE
        self.p1_count        = 0
        self.p2_count        = 0
        self.phase_start     = None   # เวลาเริ่ม phase ปัจจุบัน
        self.pause_end       = None   # เวลาที่ตาลืมหลัง phase 1

        # NEAR mode blink counter
        self.near_blink_times = []   # wall-clock ของแต่ละ blink
        self.alarm_triggered  = False

    def reset_sos(self):
        self.sos_phase  = SOS_IDLE
        self.p1_count   = 0
        self.p2_count   = 0
        self.phase_start = None
        self.pause_end  = None

    def update(self, avg_ear, now, mode):
        """
        อัปเดต state machine ทุก frame
        คืนค่า: dict ของ events  {alarm, sos_done}
        """
        events = {"alarm": False, "sos_done": False}
        is_closed = avg_ear < EAR_THRESHOLD

        # ── ตรวจ blink edge (close → open) ─────────────────
        blink_happened = False

        if is_closed:
            if not self.eye_was_closed:
                self.close_start = now
        else:
            if self.eye_was_closed and self.close_start:
                blink_dur = now - self.close_start
                if MIN_BLINK_DUR <= blink_dur <= MAX_BLINK_DUR:
                    blink_happened = True

        self.eye_was_closed = is_closed

        # ── NEAR mode: กะพริบ NEAR_BLINK_N ครั้ง = alarm ────
        if mode == MODE_NEAR:
            # ตัด blink ที่เกิน window ออก
            self.near_blink_times = [
                t for t in self.near_blink_times
                if now - t <= NEAR_BLINK_WIN
            ]
            if blink_happened:
                self.near_blink_times.append(now)

            if len(self.near_blink_times) >= NEAR_BLINK_N and not self.alarm_triggered:
                events["alarm"] = True
                self.alarm_triggered = True
                self.near_blink_times = []
                beep()
            return events

        # ── FAR mode: SOS pattern ────────────────────────────
        if mode != MODE_FAR:
            return events

        now_t = now

        # ── Timeout check ────────────────────────────────────
        if self.sos_phase == SOS_P1 and self.phase_start:
            if now_t - self.phase_start > SOS_PHASE_WIN:
                self.reset_sos()

        if self.sos_phase == SOS_PAUSE and self.pause_end:
            pause_dur = now_t - self.pause_end
            if pause_dur > SOS_PAUSE_MAX:
                self.reset_sos()

        if self.sos_phase == SOS_P2 and self.phase_start:
            if now_t - self.phase_start > SOS_PHASE_WIN:
                self.reset_sos()

        # ── State transitions ─────────────────────────────────
        if blink_happened:
            if self.sos_phase == SOS_IDLE:
                self.sos_phase   = SOS_P1
                self.phase_start = now_t
                self.p1_count    = 1

            elif self.sos_phase == SOS_P1:
                self.p1_count += 1
                if self.p1_count >= SOS_PHASE1:
                    self.sos_phase = SOS_PAUSE
                    self.pause_end = None   # รอตาลืม

            elif self.sos_phase == SOS_PAUSE:
                # ตาลืมแล้วมีกะพริบ = เริ่ม phase 2
                if self.pause_end and (now_t - self.pause_end) >= SOS_PAUSE_MIN:
                    self.sos_phase   = SOS_P2
                    self.phase_start = now_t
                    self.p2_count    = 1
                else:
                    # pause ยังสั้นเกินไป reset
                    self.reset_sos()

            elif self.sos_phase == SOS_P2:
                self.p2_count += 1
                if self.p2_count >= SOS_PHASE2:
                    self.sos_phase = SOS_DONE
                    events["sos_done"] = True

        # บันทึกเวลาที่ตาลืมหลัง phase1 เสร็จ
        if self.sos_phase == SOS_PAUSE and not is_closed and self.pause_end is None:
            self.pause_end = now_t

        return events

    def get_sos_status(self):
        return self.sos_phase, self.p1_count, self.p2_count

    def get_near_count(self):
        """จำนวน blink ที่นับได้ใน NEAR mode"""
        return len(self.near_blink_times)


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_model()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] เปิด source ไม่ได้: {VIDEO_SOURCE}")
        return

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    landmarker_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    sm          = BlinkStateMachine()
    mode        = MODE_FAR        # เริ่มต้นสมมติอยู่ไกล
    zoom_start  = None            # เวลาเริ่ม animation zoom
    alarm_flash = 0               # counter สำหรับ flash effect
    last_ts_ms  = -1

    # ── ตำแหน่งใบหน้า (normalize 0–1) สำหรับ zoom center ──
    # ใช้ exponential moving average เพื่อให้การ pan นุ่มนวล
    face_cx     = 0.5             # ค่าเริ่มต้น = กึ่งกลาง
    face_cy     = 0.5
    SMOOTH_K    = 0.12            # 0=ไม่ขยับเลย  1=snap ทันที (ปรับได้)

    print("[INFO] ระบบเริ่มทำงาน  |  ESC/Q = ออก  |  R = reset")
    print(f"  FAR mode  : inter-eye/width ≤ {FACE_FAR_RATIO:.2f}")
    print(f"  NEAR mode : inter-eye/width ≥ {FACE_CLOSE_RATIO:.2f}")
    print(f"  SOS pattern: กะพริบ {SOS_PHASE1} ครั้ง  พัก  กะพริบ {SOS_PHASE2} ครั้ง")

    with mp_vision.FaceLandmarker.create_from_options(landmarker_opts) as landmarker:
        while cap.isOpened():
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ok, frame = cap.read()
            if not ok:
                if isinstance(VIDEO_SOURCE, str):
                    break
                continue

            # timestamp
            if isinstance(VIDEO_SOURCE, int):
                timestamp_ms = int(time.time() * 1000)
            else:
                timestamp_ms = int(pos_ms)
            if timestamp_ms <= last_ts_ms:
                timestamp_ms = last_ts_ms + 1
            last_ts_ms = timestamp_ms

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # ── Digital zoom animation ───────────────────────
            current_zoom = 1.0
            if mode == MODE_ZOOM and zoom_start is not None:
                elapsed_zoom = time.time() - zoom_start
                t = min(elapsed_zoom / ZOOM_DURATION, 1.0)
                current_zoom = 1.0 + (ZOOM_FACTOR - 1.0) * t
                if t >= 1.0:
                    mode = MODE_NEAR
                    sm.reset()
                    print("[INFO] Zoom เสร็จ → เข้าโหมด NEAR")
            elif mode == MODE_NEAR:
                current_zoom = ZOOM_FACTOR

            # ── Mediapipe (ใช้ frame ดิบก่อน zoom เสมอ) ──────
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            now = time.time()

            # ── อัปเดต face center (normalize 0–1) ด้วย EMA ──
            if result.face_landmarks:
                nose = result.face_landmarks[0][1]   # landmark #1 = nose tip
                raw_cx = float(nose.x)
                raw_cy = float(nose.y)
                # Exponential moving average → smooth pan
                face_cx = face_cx + SMOOTH_K * (raw_cx - face_cx)
                face_cy = face_cy + SMOOTH_K * (raw_cy - face_cy)

            # ── Apply digital zoom เข้าหาใบหน้า ─────────────
            display_frame = (digital_zoom(frame, current_zoom, face_cx, face_cy)
                             if current_zoom > 1.01 else frame.copy())

            # ── วาด gaze zone ─────────────────────────────────
            rx = int(w * math.tan(math.radians(GAZE_THRESHOLD)) * 0.3)
            ry = int(h * math.tan(math.radians(GAZE_THRESHOLD)) * 0.3)
            cv2.ellipse(display_frame, (w//2, h//2), (rx, ry), 0, 0, 360, (40,40,40), 1)

            if not result.face_landmarks:
                sm.reset()
                put(display_frame, "No face detected", (10, 35), (0, 0, 200))
                cv2.imshow("Patient Assistance System", display_frame)
                if cv2.waitKey(5) & 0xFF in (27, ord('q')):
                    break
                continue

            face_lm = result.face_landmarks[0]

            # ── Gaze ─────────────────────────────────────────
            try:
                yaw, pitch, gaze_ok = estimate_gaze(face_lm, w, h)
            except Exception:
                yaw, pitch, gaze_ok = 0.0, 0.0, False

            # ── Distance ratio ────────────────────────────────
            ratio = face_distance_ratio(face_lm, w, h)

            # อัปเดต mode ตาม distance (hysteresis)
            if mode not in (MODE_ZOOM, MODE_NEAR):
                if ratio >= FACE_CLOSE_RATIO:
                    mode = MODE_NEAR
                    sm.reset()
                elif ratio <= FACE_FAR_RATIO:
                    mode = MODE_FAR

            # ── EAR ──────────────────────────────────────────
            ear_l, pts_l = calculate_ear(face_lm, LEFT_EYE,  w, h)
            ear_r, pts_r = calculate_ear(face_lm, RIGHT_EYE, w, h)
            avg_ear = (ear_l + ear_r) / 2.0

            # วาดกรอบตา
            eye_col = (0, 220, 0) if gaze_ok else (70, 70, 70)
            cv2.polylines(display_frame, [pts_l], True, eye_col, 1)
            cv2.polylines(display_frame, [pts_r], True, eye_col, 1)
            if gaze_ok:
                for ids in (LEFT_IRIS, RIGHT_IRIS):
                    ic = iris_center(face_lm, ids, w, h)
                    cv2.circle(display_frame, tuple(ic), 3, (0,255,255), -1)

            # ── HUD: Mode badge ───────────────────────────────
            mode_colors = {
                MODE_FAR:  (200, 120, 0),
                MODE_ZOOM: (0,   200, 200),
                MODE_NEAR: (0,   200, 0),
            }
            mc = mode_colors.get(mode, (200,200,200))
            cv2.rectangle(display_frame, (w-160, 5), (w-5, 35), mc, -1)
            put(display_frame, f"MODE: {mode}", (w-155, 28), (0,0,0), 0.6, 2)

            # ── HUD: Gaze ──────────────────────────────────────
            if gaze_ok:
                put(display_frame, "LOOKING AT CAMERA", (w//2-160, 35), (0,220,0))
            else:
                put(display_frame,
                    f"LOOK AT CAMERA ({yaw:+.0f}/{pitch:+.0f} deg)",
                    (10, 35), (0, 100, 255), 0.55)

            # ── HUD: Distance ──────────────────────────────────
            bar_val = min(ratio / FACE_CLOSE_RATIO, 1.0)
            cv2.rectangle(display_frame, (10, 50), (150, 62), (40,40,40), -1)
            cv2.rectangle(display_frame, (10, 50), (10 + int(140*bar_val), 62), mc, -1)
            put(display_frame, f"dist:{ratio:.3f}", (155, 62), (150,150,150), 0.45, 1)

            # ── Logic (เฉพาะเมื่อมองกล้อง + mode ไม่ใช่ ZOOM) ──
            if gaze_ok and mode != MODE_ZOOM:
                events = sm.update(avg_ear, now, mode)

                if mode == MODE_FAR:
                    # แสดง SOS dots
                    sos_ph, p1, p2 = sm.get_sos_status()
                    draw_sos_dots(display_frame, sos_ph, p1, p2)

                    # label phase
                    phase_labels = {
                        SOS_IDLE:  "Blink 2x ... Blink 3x to call for help",
                        SOS_P1:    f"Phase 1: {p1}/{SOS_PHASE1} blinks",
                        SOS_PAUSE: f"Phase 1 done! Now pause, then blink 3x",
                        SOS_P2:    f"Phase 2: {p2}/{SOS_PHASE2} blinks",
                        SOS_DONE:  "SOS Pattern complete!",
                    }
                    put(display_frame, phase_labels.get(sos_ph, ""),
                        (10, 100), (0, 200, 255), 0.55)

                    if events["sos_done"]:
                        print("[INFO] SOS pattern detected → starting zoom")
                        mode = MODE_ZOOM
                        zoom_start = now
                        sm.reset_sos()

                elif mode == MODE_NEAR:
                    # แสดง blink dots (เหมือน SOS แต่ไม่มี phase)
                    near_n = sm.get_near_count()
                    x_dot, y_dot = 10, 120
                    r_dot = 10
                    gap_dot = 28
                    for i in range(NEAR_BLINK_N):
                        filled = i < near_n
                        col = (0, 200, 100) if filled else (50, 50, 50)
                        cv2.circle(display_frame, (x_dot + i*gap_dot, y_dot), r_dot, col, -1)
                        cv2.circle(display_frame, (x_dot + i*gap_dot, y_dot), r_dot, (180,180,180), 1)
                    put(display_frame,
                        f"Blink {NEAR_BLINK_N}x to call for help  ({near_n}/{NEAR_BLINK_N})",
                        (10, 100), (0, 200, 100), 0.55)

                    if events["alarm"]:
                        alarm_flash = 30
                        print("!!! ALARM: ต้องการความช่วยเหลือ !!!")
                        # === LINE Notify / HTTP ใส่ที่นี่ ===

            elif mode == MODE_ZOOM:
                put(display_frame, "Zooming in...", (w//2-100, h//2),
                    (0, 220, 220), 1.2, 3)

            elif not gaze_ok:
                sm.reset()
                put(display_frame,
                    "Please face camera to activate",
                    (10, 80), (0, 100, 255), 0.52)

            # ── ALARM flash overlay ───────────────────────────
            if alarm_flash > 0:
                alpha = 0.35 if alarm_flash % 6 < 3 else 0.0
                if alpha > 0:
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
                    cv2.addWeighted(overlay, alpha, display_frame, 1-alpha, 0, display_frame)
                put(display_frame, "!!! ALARM !!!",
                    (w//2-160, h//2), (255,255,255), 2.5, 4)
                alarm_flash -= 1

            # ── HUD ล่าง ──────────────────────────────────────
            put(display_frame,
                f"EAR:{avg_ear:.3f}  Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}  zoom:{current_zoom:.1f}x",
                (10, h-10), (120,120,120), 0.42, 1)

            cv2.imshow("Patient Assistance System", display_frame)
            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('r'):
                sm.reset()
                mode = MODE_FAR
                zoom_start = None
                alarm_flash = 0
                print("[INFO] Reset")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] ปิดระบบเรียบร้อย")


if __name__ == "__main__":
    main()
