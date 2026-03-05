"""
Detector: orchestrates the full detection pipeline

Modes:
  ALL  — รัน EYE + HAND + BODY พร้อมกัน (โหมดรวม)
  EYE  — ตรวจกระพิบตาเท่านั้น
  HAND — ตรวจท่าทางมือเท่านั้น
  BODY — ตรวจท่านอนเท่านั้น
"""
import time
import cv2
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

import mediapipe as mp

from core.camera_manager import CameraManager
from core.alarm_manager import AlarmManager, AlarmType
from core.position_tracker import PositionTracker
from core.notification_manager import NotificationManager
from detectors.face_detector import FaceDetector
from detectors.hand_detector import HandDetector
from detectors.body_detector import BodyDetector, BodyResult
from renderers.eye_renderer import EyeRenderer
from renderers.hand_renderer import HandRenderer
from renderers.body_renderer import BodyRenderer
from renderers.thai_text import draw_hud_box
from utils.logger import setup_logger

logger = setup_logger('detector')


class Detector:
    """Orchestrator สำหรับ detection pipeline"""

    VALID_MODES = ['ALL', 'EYE', 'HAND', 'BODY']

    def __init__(self, camera_manager: CameraManager, alarm_manager: AlarmManager):
        self.camera_manager = camera_manager
        self.alarm_manager  = alarm_manager

        # Lazy-load detectors
        self._face_detector: Optional[FaceDetector] = None
        self._hand_detector: Optional[HandDetector] = None
        self._body_detector: Optional[BodyDetector] = None

        # Renderers
        self.eye_renderer  = EyeRenderer()
        self.hand_renderer = HandRenderer()
        self.body_renderer = BodyRenderer()

        self.current_mode = "ALL"

        # Position tracking & notifications
        self._position_tracker     = PositionTracker()
        self._notification_manager = NotificationManager()

        self._pose_timestamp_ms: int = 0

        # ── Frame-skip counters (ALL mode) ────────────────────────────────
        # eye: detect ทุก 2 frames (blink ไม่ต้องการ 30fps)
        self._EYE_SKIP         = 2
        self._eye_frame_count  = 0
        self._last_eye_det: Dict[str, Any] = _empty_detection()

        # hand: detect ทุก 3 frames
        self._HAND_SKIP        = 3
        self._hand_frame_count = 0
        self._last_hand_det:   Dict[str, Any] = _empty_detection()

        # body: detect ทุก 5 frames (pose เปลี่ยนช้า)
        self._BODY_SKIP        = 5
        self._body_frame_count = 0
        self._last_body_result: Optional[BodyResult] = None

        # Body position vote buffer
        self._VOTE_N      = 7
        self._pos_votes: deque = deque(maxlen=self._VOTE_N)

        # ── Resize scale สำหรับ detection (0.75 = ลด resolution 25%) ──
        self._DET_SCALE = 0.75

        # ThreadPool สำหรับ parallel detection (ALL mode)
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="det")

        # FPS
        self.frame_times: deque = deque(maxlen=30)
        self.fps: float = 0.0

        logger.info("Detector initialized (default mode: ALL)")

    # ── Lazy properties ────────────────────────────────────────────────────

    @property
    def face_detector(self) -> FaceDetector:
        if self._face_detector is None:
            logger.info("Lazy-loading FaceDetector...")
            self._face_detector = FaceDetector()
        return self._face_detector

    @property
    def hand_detector(self) -> HandDetector:
        if self._hand_detector is None:
            logger.info("Lazy-loading HandDetector...")
            self._hand_detector = HandDetector()
        return self._hand_detector

    # ── Mode helpers ───────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        mode = mode.upper()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Valid: {self.VALID_MODES}")
        if mode != self.current_mode:
            self.current_mode = mode
            if self._face_detector:
                self._face_detector.reset()
            if self._hand_detector:
                self._hand_detector.reset()
            logger.info(f"Mode changed to: {mode}")

    def _use_eye(self)  -> bool: return self.current_mode in ('ALL', 'EYE')
    def _use_hand(self) -> bool: return self.current_mode in ('ALL', 'HAND')
    def _use_body(self) -> bool: return self.current_mode in ('ALL', 'BODY')

    # ── Public API — delegation to sub-managers ─────────────────────────────

    def get_position_status(self) -> dict:
        """คืนสถานะท่านอนปัจจุบัน (ใช้โดย app.py /api/status)"""
        status = self._position_tracker.get_status()
        return {
            'current_position':     status.current_position,
            'current_duration':     status.current_duration,
            'time_until_reposition': status.time_until_reposition,
            'reposition_due':        status.reposition_due,
        }

    def reset_reposition_timer(self):
        """พยาบาลกด 'พลิกตัวแล้ว'"""
        self._position_tracker.reset_reposition_timer()
        self._notification_manager.acknowledge_reposition()

    def get_meal_status(self) -> list:
        """คืน meal status สำหรับ /api/meal_times GET"""
        return self._notification_manager.get_meal_status()

    def set_meal_times(self, meal_times: dict):
        """ตั้งเวลาอาหาร — รับจาก /api/meal_times POST"""
        self._notification_manager.set_meal_times(meal_times)

    def mark_meal_eaten(self, meal_name: str) -> bool:
        """บันทึกว่าทานอาหารมื้อนั้นแล้ว"""
        return self._notification_manager.mark_meal_eaten(meal_name)

    def pop_pending_notifications(self) -> list:
        """ดึง notifications ที่ยังไม่ได้ส่ง frontend"""
        return self._notification_manager.pop_pending()

    # ── Overlay toggle ──────────────────────────────────────────────────────

    # สถานะเริ่มต้น: แยก overlay ตามโหมด
    _overlay_states: Dict[str, Dict[str, bool]] = {
        'ALL': {'eye': True, 'hand': True, 'body': True},
        'EYE': {'eye': True, 'hand': False, 'body': False},
        'HAND': {'eye': False, 'hand': True, 'body': False},
        'BODY': {'eye': False, 'hand': False, 'body': True},
    }

    def toggle_overlay(self, name: str) -> bool:
        """สลับ on/off ของ overlay ตามชื่อ (eye/hand/body)
        Returns: สถานะใหม่ (True = แสดง)
        """
        name = name.lower()
        current_state = self._overlay_states[self.current_mode]
        if name not in current_state:
            raise ValueError(f"Invalid overlay name: {name} in mode {self.current_mode}")
        current_state[name] = not current_state[name]
        logger.info(f"Overlay '{name}' in mode '{self.current_mode}' = {current_state[name]}")
        return current_state[name]

    def get_overlay_state(self) -> Dict[str, bool]:
        """คืนสถานะ overlay ทั้งหมดของโหมดปัจจุบัน"""
        return dict(self._overlay_states[self.current_mode])


    def process_frame(self) -> Optional[Dict[str, Any]]:
        start = time.time()

        frame_data = self.camera_manager.get_frame()
        if frame_data is None:
            return None

        bgr_frame, _ = frame_data
        rgb_frame     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        eye_detection:  Dict[str, Any] = _empty_detection()
        hand_detection: Dict[str, Any] = _empty_detection()
        body_result:    Optional[BodyResult] = None

        try:
            if self.current_mode == 'ALL':
                # ── resize สำหรับ detection (ลด CPU/memory) ──
                h, w = rgb_frame.shape[:2]
                det_w = int(w * self._DET_SCALE)
                det_h = int(h * self._DET_SCALE)
                det_frame = cv2.resize(rgb_frame, (det_w, det_h),
                                       interpolation=cv2.INTER_LINEAR)

                # ── Selective futures: submit เฉพาะ frame ที่ถึงเวลา detect ──
                futures = {}

                self._eye_frame_count += 1
                if self._eye_frame_count >= self._EYE_SKIP:
                    self._eye_frame_count = 0
                    futures['eye'] = self._pool.submit(
                        self.face_detector.detect, det_frame, rgb_frame.shape[:2])

                self._hand_frame_count += 1
                if self._hand_frame_count >= self._HAND_SKIP:
                    self._hand_frame_count = 0
                    futures['hand'] = self._pool.submit(
                        self._run_hand_detection, det_frame, True)

                self._body_frame_count += 1
                if self._body_frame_count >= self._BODY_SKIP:
                    self._body_frame_count = 0
                    futures['body'] = self._pool.submit(
                        self._run_body_detection, det_frame, True)

                # รอ futures ที่ submit, ใช้ cache สำหรับ frame ที่ skip
                if 'eye' in futures:
                    self._last_eye_det = futures['eye'].result()
                eye_detection = self._last_eye_det

                if 'hand' in futures:
                    self._last_hand_det = futures['hand'].result()
                hand_detection = self._last_hand_det

                if 'body' in futures:
                    self._last_body_result = futures['body'].result()
                body_result = self._last_body_result

            else:
                # ── Single-mode: sequential ───────────────────────────────────
                if self._use_eye():
                    eye_detection = self.face_detector.detect(rgb_frame)
                if self._use_hand():
                    hand_detection = self._run_hand_detection(rgb_frame)
                if self._use_body():
                    body_result = self._run_body_detection(rgb_frame)

        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)

        # ── Render ─────────────────────────────────────────────────────────
        try:
            rendered_frame = self._render(bgr_frame, eye_detection,
                                          hand_detection, body_result)
        except Exception as e:
            logger.error(f"Render error: {e}", exc_info=True)
            rendered_frame = bgr_frame

        # ── Alarms ─────────────────────────────────────────────────────────
        if eye_detection.get('alarm'):
            self.alarm_manager.trigger_alarm(AlarmType.EYE_BLINK)
        if hand_detection.get('alarm'):
            self.alarm_manager.trigger_alarm(AlarmType.HAND_GESTURE)

        # ── Position tracking ──────────────────────────────────────────────
        if body_result is not None:
            # Vote: เก็บ position ทุก frame ที่ detect ได้
            self._pos_votes.append(body_result.position)
            # เสียงข้างมากจาก buffer → ส่ง tracker
            voted_pos = max(set(self._pos_votes), key=self._pos_votes.count)
            self._position_tracker.update(voted_pos)

        # ── Notification tick ──────────────────────────────────────────────
        tracker_status    = self._position_tracker.get_status()
        new_notifications = self._notification_manager.tick(
            reposition_due=tracker_status.reposition_due
        )
        for notif in new_notifications:
            if notif.type == "reposition":
                self.alarm_manager.trigger_alarm(
                    AlarmType.REPOSITION_REMINDER
                    if hasattr(AlarmType, 'REPOSITION_REMINDER')
                    else AlarmType.EYE_BLINK
                )

        # ── Encode (ALL mode ใช้ quality 55 เพื่อ throughput ดีขึ้น) ───────────
        jpeg_q = 55 if self.current_mode == 'ALL' else 70
        _, buffer = cv2.imencode('.jpg', rendered_frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        encoded_frame = buffer.tobytes()

        # ── FPS ────────────────────────────────────────────────────────────
        elapsed = time.time() - start
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 1:
            avg      = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg if avg > 0 else 0.0

        return {
            'encoded_frame':  encoded_frame,
            'mode':           self.current_mode,
            'fps':            round(self.fps, 1),
            'detection':      eye_detection if self._use_eye() else hand_detection,
            'eye_detection':  eye_detection,
            'hand_detection': hand_detection,
            'body': {
                'position':   body_result.position   if body_result else 'UNKNOWN',
                'confidence': body_result.confidence if body_result else 0.0,
            },
            'position_status': self.get_position_status(),
        }

    # ── Render ─────────────────────────────────────────────────────────────

    def _render(self, bgr_frame: np.ndarray,
                eye_det: dict, hand_det: dict,
                body_result: Optional[BodyResult]) -> np.ndarray:
        """วาด overlay ทุก layer ที่ active"""
        out = bgr_frame.copy()

        current_overlay = self._overlay_states[self.current_mode]
        is_all_mode = self.current_mode == 'ALL'

        # 1. Eye overlay
        if self._use_eye() and current_overlay.get('eye', False):
            out = self.eye_renderer.render(out, eye_det, is_all_mode=is_all_mode)

        # 2. Hand skeleton — วาดจุด landmark + เส้นเชื่อม
        if self._use_hand() and current_overlay.get('hand', False) and hand_det.get('detected'):
            data = hand_det.get('data') or {}
            landmarks = data.get('landmarks')
            alarm     = hand_det.get('alarm', False)
            if landmarks is not None:
                out = HandDetector.draw_landmarks(out, landmarks, alarm=alarm)
            # วาด HUD text box สำหรับมือ (ใช้ hand_renderer)
            out = self.hand_renderer.render(out, hand_det, is_all_mode=is_all_mode)

        # 3. Body skeleton + HUD (มุมบนขวา)
        if self._use_body() and body_result is not None and current_overlay.get('body', False):
            out = self._render_body_overlay(out, body_result)

        return out

    def _render_body_overlay(self, bgr_frame: np.ndarray,
                              result: BodyResult) -> np.ndarray:
        """วาด body skeleton + ป้ายท่านอน (มุมบนขวา)"""
        out = bgr_frame.copy()
        h, w = out.shape[:2]

        # วาด skeleton ถ้ามี landmarks
        if result.raw_landmarks is not None:
            out = BodyDetector.draw_landmarks(
                out, result.raw_landmarks, position=result.position
            )

        # ── ป้าย + เวลาพลิกตัว (มุมบนขวา ไม่ทับ eye/hand HUD ซ้ายบน) ──
        _POS_LABEL = {
            'SUPINE':     ('หงาย',       (100, 200, 100)),
            'LEFT_SIDE':  ('ตะแคงซ้าย', (100, 180, 255)),
            'RIGHT_SIDE': ('ตะแคงขวา',  (255, 180, 100)),
            'UNKNOWN':    ('ไม่ทราบ',    (180, 180, 180)),
        }
        label_th, color = _POS_LABEL.get(result.position, ('ไม่ทราบ', (180, 180, 180)))
        conf_pct = int(result.confidence * 100)

        status    = self._position_tracker.get_status()
        remaining = int(status.time_until_reposition)
        hrs, mins = divmod(remaining // 60, 60)

        if status.reposition_due:
            flip_text  = 'ถึงเวลาพลิกตัวแล้ว!'
            flip_color = (0, 0, 220)
        else:
            flip_text  = f'พลิกตัวใน: {hrs} ชม. {mins} นาที'
            flip_color = (0, 220, 220)

        border_color = (0, 0, 220) if status.reposition_due else color
        box_w  = 300
        box_x  = w - box_w - 8

        lines = [
            (f'ท่านอน : {label_th} ({conf_pct}%)', color),
            (flip_text, flip_color),
        ]

        # วาดกล่อง HUD มุมขวาบน
        line_h = 32
        pad_y  = 10
        box_h  = pad_y * 2 + line_h * len(lines)
        import cv2 as _cv2
        overlay = out.copy()
        _cv2.rectangle(overlay, (box_x, 4), (box_x + box_w, 4 + box_h), (20, 20, 20), -1)
        _cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
        _cv2.rectangle(out, (box_x, 4), (box_x + box_w, 4 + box_h), border_color, 1)

        from renderers.thai_text import put_thai
        y = 4 + pad_y + line_h - 4
        for text, c in lines:
            put_thai(out, text, (box_x + 10, y), font_size=20, color=c)
            y += line_h

        return out

    # ── Hand detection (with frame-skip) ───────────────────────────────────

    def _run_hand_detection(self, rgb_frame: np.ndarray, force: bool = False) -> Dict[str, Any]:
        """รัน hand detection พร้อม frame-skip ทุก _HAND_SKIP frames"""
        if not force:
            self._hand_frame_count += 1
            if self._hand_frame_count % self._HAND_SKIP != 0:
                return self._last_hand_det  # reuse ผลล่าสุด
        result = self.hand_detector.detect(rgb_frame)
        self._last_hand_det = result
        return result

    # ── Body detection ─────────────────────────────────────────────────────

    def _run_body_detection(self, rgb_frame: np.ndarray, force: bool = False) -> Optional[BodyResult]:
        # ── Lazy-load body detector ──────────────────────────────────────────
        if self._body_detector is None:
            self._body_detector = BodyDetector()
            if not self._body_detector.load():
                logger.error("BodyDetector failed to load — disabled")
                self._body_detector = None
                return None

        # ── Frame-skip: รัน MediaPipe ทุก _BODY_SKIP frames ──────────────────
        if not force:
            self._body_frame_count += 1
            if self._body_frame_count % self._BODY_SKIP != 0:
                return self._last_body_result

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._pose_timestamp_ms += self._BODY_SKIP * 33
        result = self._body_detector.detect(mp_image, self._pose_timestamp_ms)
        self._last_body_result = result
        return result

    # ── Position & Notification public API ────────────────────────────────

    def get_position_status(self) -> dict:
        s = self._position_tracker.get_status()
        return {
            'current_position':      s.current_position,
            'current_duration':      s.current_duration,
            'time_until_reposition': s.time_until_reposition,
            'reposition_due':        s.reposition_due,
            'reposition_interval':   s.reposition_interval,
        }

    def get_position_history(self, limit: int = 50) -> list:
        return self._position_tracker.get_history(limit)

    def reset_reposition_timer(self):
        self._position_tracker.reset_reposition_timer()
        self._notification_manager.acknowledge_reposition()

    def get_meal_status(self) -> list:
        return self._notification_manager.get_meal_status()

    def mark_meal_eaten(self, meal_name: str) -> bool:
        return self._notification_manager.mark_meal_eaten(meal_name)

    def set_meal_times(self, meal_times: dict) -> None:
        self._notification_manager.set_meal_times(meal_times)

    def pop_pending_notifications(self) -> list:
        return self._notification_manager.pop_pending()

    def get_fps(self) -> float:
        return round(self.fps, 1)


# ── module-level helper ────────────────────────────────────────────────────

def _empty_detection() -> Dict[str, Any]:
    return {'detected': False, 'alarm': False, 'confidence': 0.0, 'data': None}