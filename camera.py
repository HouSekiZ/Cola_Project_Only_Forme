import cv2
import math
import time
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =========================================================
# CONFIG & CONSTANTS
# =========================================================
# Eye Constants
GAZE_THRESHOLD = 45
FACE_CLOSE_RATIO = 0.18
FACE_FAR_RATIO = 0.12
EAR_THRESHOLD = 0.22
MIN_BLINK_DUR = 0.08
MAX_BLINK_DUR = 0.6

# SOS Pattern
SOS_PHASE1 = 2
SOS_PHASE2 = 3
SOS_PAUSE_MIN = 0.4
SOS_PAUSE_MAX = 3.0
SOS_PHASE_WIN = 4.0

# Near Mode
NEAR_BLINK_N = 3
NEAR_BLINK_WIN = 8.0

# Digital Zoom
ZOOM_FACTOR   = 2.5   # เท่าที่ zoom เข้า
ZOOM_DURATION = 1.5   # วินาที ที่ใช้ zoom animation

# Models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Landmark Indices
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
FACE_6PTS  = [1, 152, 263, 33, 287, 57]
L_EYE_OUTER = 263
R_EYE_OUTER = 33

# 3D Model for Pose
MODEL_3D = np.array([
    [0.0, 0.0, 0.0], [0.0, -63.6, -12.5], [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0], [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1],
], dtype=np.float64)

# States
MODE_FAR = "FAR"
MODE_NEAR = "NEAR"
MODE_ZOOM = "ZOOMING" # (Optional feature)

# =========================================================
# HELPERS
# =========================================================
def ensure_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(FACE_MODEL_PATH):
        print(f"Downloading Face Model...")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        
    if not os.path.exists(HAND_MODEL_PATH):
        print(f"Downloading Hand Model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

def get_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def dist2d(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def calculate_ear(lms, indices, w, h):
    pts = np.array([get_px(lms[i], w, h) for i in indices])
    v1 = dist2d(pts[1], pts[5])
    v2 = dist2d(pts[2], pts[4])
    hz = dist2d(pts[0], pts[3])
    return ((v1 + v2) / (2.0 * hz) if hz > 0 else 0.0), pts

def iris_center(lms, ids, w, h):
    pts = np.array([get_px(lms[i], w, h) for i in ids], dtype=float)
    return pts.mean(axis=0).astype(int)

def estimate_gaze(lms, w, h):
    """คำนวณ yaw/pitch และตรวจว่า gaze ตรงกล้องหรือไม่"""
    img_pts = np.array([get_px(lms[i], w, h) for i in FACE_6PTS], dtype=np.float64)
    f   = float(w)
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img_pts, cam, np.zeros((4, 1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, False
    rmat, _ = cv2.Rodrigues(rvec)
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
        np.hstack((rmat, np.zeros((3, 1)))))
    ph, yh = float(euler[0]), float(euler[1])

    li = iris_center(lms, LEFT_IRIS,  w, h).astype(float)
    ri = iris_center(lms, RIGHT_IRIS, w, h).astype(float)

    def eye_off(ic, eye_ids):
        eps = np.array([get_px(lms[i], w, h) for i in eye_ids], dtype=float)
        ex = (ic[0] - eps[:, 0].min()) / max(eps[:, 0].max() - eps[:, 0].min(), 1) - 0.5
        ey = (ic[1] - eps[:, 1].min()) / max(eps[:, 1].max() - eps[:, 1].min(), 1) - 0.5
        return ex, ey

    lx, ly = eye_off(li, LEFT_EYE)
    rx, ry = eye_off(ri, RIGHT_EYE)
    S     = 40.0
    yaw   = yh + ((lx + rx) / 2) * S
    pitch = ph + ((ly + ry) / 2) * S
    return yaw, pitch, abs(yaw) <= GAZE_THRESHOLD and abs(pitch) <= GAZE_THRESHOLD

def face_distance_ratio(lms, w, h):
    """inter-eye distance / frame_width  (ใหญ่ = อยู่ใกล้)"""
    lp = get_px(lms[L_EYE_OUTER], w, h)
    rp = get_px(lms[R_EYE_OUTER], w, h)
    return dist2d(lp, rp) / w

def digital_zoom(frame, factor, cx_norm=0.5, cy_norm=0.5):
    """Zoom เข้าหาจุด (cx_norm, cy_norm) แล้ว resize กลับขนาดเดิม"""
    h, w   = frame.shape[:2]
    new_h  = int(h / factor)
    new_w  = int(w / factor)
    cx     = int(cx_norm * w)
    cy     = int(cy_norm * h)
    x1     = max(0, min(cx - new_w // 2, w - new_w))
    y1     = max(0, min(cy - new_h // 2, h - new_h))
    return cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h),
                      interpolation=cv2.INTER_LINEAR)

def check_hand_state(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_joints = [6, 10, 14, 18]
    fingers_open = []
    for tip, joint in zip(finger_tips, finger_joints):
        if landmarks[tip].y < landmarks[joint].y:
            fingers_open.append(True)
        else:
            fingers_open.append(False)
    
    if all(not f for f in fingers_open): return "FIST"
    elif all(f for f in fingers_open): return "OPEN"
    return "UNKNOWN"

# =========================================================
# SOS STATE CONSTANTS
# =========================================================
SOS_IDLE  = "IDLE"
SOS_P1    = "PHASE1"
SOS_PAUSE = "PAUSE"
SOS_P2    = "PHASE2"
SOS_DONE  = "DONE"


# =========================================================
# STATE MACHINE CLASS  (ported from eye_blink.py — full logic)
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
        self.near_blink_times = []
        self.alarm_triggered  = False

    def reset_sos(self):
        self.sos_phase   = SOS_IDLE
        self.p1_count    = 0
        self.p2_count    = 0
        self.phase_start = None
        self.pause_end   = None

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
            return events

        # ── FAR mode: SOS pattern ─────────────────────────────
        if mode != MODE_FAR:
            return events

        # ── Timeout check ─────────────────────────────────────
        if self.sos_phase == SOS_P1 and self.phase_start:
            if now - self.phase_start > SOS_PHASE_WIN:
                self.reset_sos()

        if self.sos_phase == SOS_PAUSE and self.pause_end:
            if now - self.pause_end > SOS_PAUSE_MAX:
                self.reset_sos()

        if self.sos_phase == SOS_P2 and self.phase_start:
            if now - self.phase_start > SOS_PHASE_WIN:
                self.reset_sos()

        # ── State transitions ──────────────────────────────────
        if blink_happened:
            if self.sos_phase == SOS_IDLE:
                self.sos_phase   = SOS_P1
                self.phase_start = now
                self.p1_count    = 1

            elif self.sos_phase == SOS_P1:
                self.p1_count += 1
                if self.p1_count >= SOS_PHASE1:
                    self.sos_phase = SOS_PAUSE
                    self.pause_end = None   # รอตาลืม

            elif self.sos_phase == SOS_PAUSE:
                # ตาลืมแล้วมีกะพริบ = เริ่ม phase 2
                if self.pause_end and (now - self.pause_end) >= SOS_PAUSE_MIN:
                    self.sos_phase   = SOS_P2
                    self.phase_start = now
                    self.p2_count    = 1
                else:
                    self.reset_sos()   # pause สั้นเกินไป → reset

            elif self.sos_phase == SOS_P2:
                self.p2_count += 1
                if self.p2_count >= SOS_PHASE2:
                    self.sos_phase = SOS_DONE
                    events["sos_done"] = True

        # บันทึกเวลาที่ตาลืมหลัง phase1 เสร็จ
        if self.sos_phase == SOS_PAUSE and not is_closed and self.pause_end is None:
            self.pause_end = now

        return events

    def get_sos_status(self):
        """คืน (sos_phase, p1_count, p2_count)"""
        return self.sos_phase, self.p1_count, self.p2_count

    def get_near_count(self):
        """จำนวน blink ที่นับได้ใน NEAR mode"""
        return len(self.near_blink_times)

# =========================================================
# VIDEO CAMERA CLASS
# =========================================================
class VideoCamera(object):
    def __init__(self):
        ensure_models()
        self.video        = cv2.VideoCapture(0)
        self.alarm_active = False
        
        # --- Initialize Models ---
        base_opts_face = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=base_opts_face,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )
        
        base_opts_hand = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=base_opts_hand,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1
            )
        )

        # --- States ---
        self.sm              = BlinkStateMachine()
        self.blink_mode      = MODE_FAR   # FAR → ZOOM → NEAR
        self.zoom_start      = None       # เวลาเริ่ม zoom animation
        self.face_cx         = 0.5        # ตำแหน่งหน้า normalize (EMA)
        self.face_cy         = 0.5
        self.SMOOTH_K        = 0.12       # ความเร็ว pan
        self.alarm_flash     = 0          # frame counter สำหรับ flash overlay
        self.last_hand_state = "UNKNOWN"
        self.start_time      = time.time()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None, False

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        timestamp_ms = int((time.time() - self.start_time) * 1000)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        current_alarm = False

        # ── รัน Face + Hand พร้อมกันทุก frame ────────────────
        face_result = self.face_landmarker.detect_for_video(mp_img, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_img, timestamp_ms)

        # ── อัปเดต face center ด้วย EMA (ใช้ frame ดิบก่อน zoom) ──
        if face_result.face_landmarks:
            nose = face_result.face_landmarks[0][1]
            self.face_cx += self.SMOOTH_K * (float(nose.x) - self.face_cx)
            self.face_cy += self.SMOOTH_K * (float(nose.y) - self.face_cy)

        # ── Zoom animation (MODE_ZOOM → MODE_NEAR) ──────────────
        current_zoom = 1.0
        if self.blink_mode == MODE_ZOOM and self.zoom_start is not None:
            elapsed = time.time() - self.zoom_start
            t = min(elapsed / ZOOM_DURATION, 1.0)
            current_zoom = 1.0 + (ZOOM_FACTOR - 1.0) * t
            if t >= 1.0:
                self.blink_mode = MODE_NEAR
                self.sm.reset()
        elif self.blink_mode == MODE_NEAR:
            current_zoom = ZOOM_FACTOR

        # ── Apply digital zoom + คำนวณ crop box ─────────────
        if current_zoom > 1.01:
            zh, zw = frame.shape[:2]
            new_h  = int(zh / current_zoom)
            new_w  = int(zw / current_zoom)
            cx     = int(self.face_cx * zw)
            cy     = int(self.face_cy * zh)
            crop_x1 = max(0, min(cx - new_w // 2, zw - new_w))
            crop_y1 = max(0, min(cy - new_h // 2, zh - new_h))
            zoom_crop     = (crop_x1, crop_y1, new_w, new_h)
            display_frame = digital_zoom(frame, current_zoom, self.face_cx, self.face_cy)
        else:
            zoom_crop     = None
            display_frame = frame.copy()

        # ── Process Eye + Hand บน display_frame เดียวกัน ────
        display_frame, eye_alarm  = self.process_eye(
            display_frame, face_result, w, h, current_zoom, zoom_crop)
        display_frame, hand_alarm = self.process_hand(
            display_frame, hand_result, w, h)

        if eye_alarm or hand_alarm:
            self.alarm_active = True

        ret, jpeg = cv2.imencode('.jpg', display_frame)
        return jpeg.tobytes(), self.alarm_active



    def _lm_to_px(self, lm, w, h, zoom_crop):
        """แปลง landmark (normalized 0-1 ของ frame ดิบ) → pixel บน display_frame (zoomed)"""
        px = lm.x * w
        py = lm.y * h
        if zoom_crop is not None:
            cx1, cy1, cw, ch = zoom_crop
            px = (px - cx1) / cw * w
            py = (py - cy1) / ch * h
        return int(px), int(py)

    def process_eye(self, frame, result, w, h, current_zoom=1.0, zoom_crop=None):
        alarm = False
        now   = time.time()

        # ── Gaze ellipse zone (วาดทุก frame) ─────────────────
        import math
        rx = int(w * math.tan(math.radians(GAZE_THRESHOLD)) * 0.3)
        ry = int(h * math.tan(math.radians(GAZE_THRESHOLD)) * 0.3)
        cv2.ellipse(frame, (w // 2, h // 2), (rx, ry), 0, 0, 360, (40, 40, 40), 1)

        # ── ไม่เจอหน้า ────────────────────────────────────────
        if not result.face_landmarks:
            self.sm.reset()
            cv2.putText(frame, "No Face Detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            self._draw_alarm_flash(frame, w, h)
            return frame, False

        face_lm = result.face_landmarks[0]

        # ── Gaze check (ใช้ raw frame coords เสมอ) ───────────
        try:
            yaw, pitch, gaze_ok = estimate_gaze(face_lm, w, h)
        except Exception:
            yaw, pitch, gaze_ok = 0.0, 0.0, False

        # ── Distance ratio ────────────────────────────────────
        ratio = face_distance_ratio(face_lm, w, h)

        # ── อัปเดต mode (เฉพาะ FAR เท่านั้น) ─────────────────
        if self.blink_mode == MODE_FAR:
            if ratio <= FACE_FAR_RATIO:
                self.blink_mode = MODE_FAR  # ยืนยัน FAR (hysteresis boundary ล่าง)
            # ไม่ auto-switch เข้า NEAR จาก distance
            # ต้องกะพริบ SOS ครบก่อนเท่านั้น

        # ── EAR (คำนวณจาก raw frame coords) ──────────────────
        ear_l, pts_l_raw = calculate_ear(face_lm, LEFT_EYE,  w, h)
        ear_r, pts_r_raw = calculate_ear(face_lm, RIGHT_EYE, w, h)
        avg_ear = (ear_l + ear_r) / 2.0

        # ── แปลง eye pts → display coords (รองรับ zoom) ──────
        def remap_pts(pts_raw):
            if zoom_crop is None:
                return pts_raw
            cx1, cy1, cw, ch = zoom_crop
            remapped = pts_raw.copy().astype(float)
            remapped[:, 0] = (remapped[:, 0] - cx1) / cw * w
            remapped[:, 1] = (remapped[:, 1] - cy1) / ch * h
            return remapped.astype(int)

        pts_l = remap_pts(pts_l_raw)
        pts_r = remap_pts(pts_r_raw)

        # ── กรอบหน้า (NEAR mode เท่านั้น) ────────────────────
        if self.blink_mode == MODE_NEAR:
            dxs = [self._lm_to_px(lm, w, h, zoom_crop)[0] for lm in face_lm]
            dys = [self._lm_to_px(lm, w, h, zoom_crop)[1] for lm in face_lm]
            pad = 18
            fx1, fy1 = max(0, min(dxs) - pad), max(0, min(dys) - pad)
            fx2, fy2 = min(w, max(dxs) + pad), min(h, max(dys) + pad)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 220, 80), 2)
            clen = 18
            for cx_, cy_, dx, dy in [
                (fx1, fy1,  1,  1), (fx2, fy1, -1,  1),
                (fx1, fy2,  1, -1), (fx2, fy2, -1, -1),
            ]:
                cv2.line(frame, (cx_, cy_), (cx_ + dx * clen, cy_), (0, 255, 100), 3)
                cv2.line(frame, (cx_, cy_), (cx_, cy_ + dy * clen), (0, 255, 100), 3)

        # ── กรอบตา + iris dots (display coords) ───────────────
        eye_col = (0, 220, 0) if gaze_ok else (70, 70, 70)
        cv2.polylines(frame, [pts_l], True, eye_col, 1)
        cv2.polylines(frame, [pts_r], True, eye_col, 1)
        if gaze_ok:
            for ids in (LEFT_IRIS, RIGHT_IRIS):
                raw_ic = iris_center(face_lm, ids, w, h)
                if zoom_crop is not None:
                    cx1, cy1, cw, ch = zoom_crop
                    ic_x = int((raw_ic[0] - cx1) / cw * w)
                    ic_y = int((raw_ic[1] - cy1) / ch * h)
                else:
                    ic_x, ic_y = int(raw_ic[0]), int(raw_ic[1])
                cv2.circle(frame, (ic_x, ic_y), 3, (0, 255, 255), -1)

        # ── Mode badge (มุมบนขวา) ─────────────────────────────
        mode_colors = {
            MODE_FAR:  (200, 120,   0),
            MODE_ZOOM: (  0, 200, 200),
            MODE_NEAR: (  0, 200,   0),
        }
        mc = mode_colors.get(self.blink_mode, (200, 200, 200))
        cv2.rectangle(frame, (w - 160, 5), (w - 5, 35), mc, -1)
        cv2.putText(frame, f"MODE: {self.blink_mode}",
                    (w - 155, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # ── Gaze HUD ──────────────────────────────────────────
        if gaze_ok:
            cv2.putText(frame, "LOOKING AT CAMERA",
                        (w // 2 - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
        else:
            cv2.putText(frame,
                        f"LOOK AT CAMERA ({yaw:+.0f}/{pitch:+.0f} deg)",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

        # ── Distance bar ──────────────────────────────────────
        bar_val = min(ratio / FACE_CLOSE_RATIO, 1.0)
        cv2.rectangle(frame, (10, 50), (150, 62), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 50), (10 + int(140 * bar_val), 62), mc, -1)
        cv2.putText(frame, f"dist:{ratio:.3f}",
                    (155, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # ── ZOOM overlay ──────────────────────────────────────
        if self.blink_mode == MODE_ZOOM:
            cv2.putText(frame, "Zooming in...",
                        (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 220), 3)

        # ── Logic (เฉพาะเมื่อมองกล้อง + ไม่อยู่ใน ZOOM) ─────
        if gaze_ok and self.blink_mode != MODE_ZOOM:
            events = self.sm.update(avg_ear, now, self.blink_mode)

            if self.blink_mode == MODE_FAR:
                # SOS dots ──────────────────────────────────────
                sos_ph, p1, p2 = self.sm.get_sos_status()
                x_d, y_d, r_d, gap_d = 10, 120, 9, 24
                for i in range(SOS_PHASE1):
                    filled = (sos_ph in (SOS_P1, SOS_PAUSE, SOS_P2, SOS_DONE)) and i < p1
                    col = (0, 220, 255) if filled else (50, 50, 50)
                    cv2.circle(frame, (x_d + i * gap_d, y_d), r_d, col, -1)
                    cv2.circle(frame, (x_d + i * gap_d, y_d), r_d, (180, 180, 180), 1)
                sx = x_d + SOS_PHASE1 * gap_d + 4
                cv2.putText(frame, "...", (sx, y_d + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                ox = sx + 28
                for i in range(SOS_PHASE2):
                    filled = sos_ph in (SOS_P2, SOS_DONE) and i < p2
                    col = (0, 100, 255) if filled else (50, 50, 50)
                    cv2.circle(frame, (ox + i * gap_d, y_d), r_d, col, -1)
                    cv2.circle(frame, (ox + i * gap_d, y_d), r_d, (180, 180, 180), 1)

                phase_labels = {
                    SOS_IDLE:  "Blink 2x ... Blink 3x to call for help",
                    SOS_P1:    f"Phase 1: {p1}/{SOS_PHASE1} blinks",
                    SOS_PAUSE: "Phase 1 done! Now pause, then blink 3x",
                    SOS_P2:    f"Phase 2: {p2}/{SOS_PHASE2} blinks",
                    SOS_DONE:  "SOS Pattern complete!",
                }
                cv2.putText(frame, phase_labels.get(sos_ph, ""),
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

                if events["sos_done"]:
                    self.sm.reset_sos()
                    if ratio <= FACE_FAR_RATIO:
                        # หน้าไกลเกิน → zoom เข้าหาหน้าก่อน แล้วค่อยเข้า NEAR
                        self.blink_mode = MODE_ZOOM
                        self.zoom_start = now
                    else:
                        # หน้าอยู่ใกล้อยู่แล้ว → เข้า NEAR ทันทีไม่ต้อง zoom
                        self.blink_mode = MODE_NEAR
                        self.sm.reset()

            elif self.blink_mode == MODE_NEAR:
                # Near blink dots ───────────────────────────────
                near_n = self.sm.get_near_count()
                x_d, y_d, r_d, gap_d = 10, 120, 10, 28
                for i in range(NEAR_BLINK_N):
                    col = (0, 200, 100) if i < near_n else (50, 50, 50)
                    cv2.circle(frame, (x_d + i * gap_d, y_d), r_d, col, -1)
                    cv2.circle(frame, (x_d + i * gap_d, y_d), r_d, (180, 180, 180), 1)
                cv2.putText(frame,
                            f"Blink {NEAR_BLINK_N}x to call for help  ({near_n}/{NEAR_BLINK_N})",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2)

                if events["alarm"]:
                    alarm = True
                    self.alarm_flash  = 30
                    self.alarm_active = True

        elif not gaze_ok and self.blink_mode != MODE_ZOOM:
            self.sm.reset()
            cv2.putText(frame, "Please face camera to activate",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 100, 255), 2)

        # ── Alarm flash overlay ───────────────────────────────
        self._draw_alarm_flash(frame, w, h)

        # ── Bottom HUD ────────────────────────────────────────
        cv2.putText(frame,
                    f"EAR:{avg_ear:.3f}  Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}  zoom:{current_zoom:.1f}x",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        return frame, alarm

    def _draw_alarm_flash(self, frame, w, h):
        """Flash overlay แดงทั้งจอ + ข้อความ !!! ALARM !!! (30 frame)"""
        if self.alarm_flash > 0:
            alpha = 0.35 if self.alarm_flash % 6 < 3 else 0.0
            if alpha > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "!!! ALARM !!!",
                        (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (255, 255, 255), 4)
            self.alarm_flash -= 1
            # เมื่อ flash หมด → reset alarm_active ให้ frontend รู้ว่าจบแล้ว
            if self.alarm_flash == 0:
                self.alarm_active = False


    def process_hand(self, frame, result, w, h):
        alarm = False
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (255, 200, 0), -1)

                state = check_hand_state(landmarks)

                # Logic: Open → Fist = alarm
                if self.last_hand_state == "OPEN" and state == "FIST":
                    alarm = True
                    self.alarm_flash  = 30
                    self.alarm_active = True

                self.last_hand_state = state

                # label มุมล่างขวา ไม่ทับ eye HUD
                col = (0, 220, 255) if state == "OPEN" else \
                      (0, 80,  255) if state == "FIST" else (150, 150, 150)
                cv2.putText(frame, f"Hand: {state}",
                            (w - 185, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        if alarm:
            cv2.putText(frame, "HELP SIGNAL!",
                        (w // 2 - 130, h // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return frame, alarm