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

    # สถานะเริ่มต้น: แสดงทุก overlay
    _overlay: Dict[str, bool] = {'eye': True, 'hand': True, 'body': True}

    def toggle_overlay(self, name: str) -> bool:
        """สลับ on/off ของ overlay ตามชื่อ (eye/hand/body)
        Returns: สถานะใหม่ (True = แสดง)
        """
        name = name.lower()
        if name not in self._overlay:
            raise ValueError(f"Invalid overlay name: {name}")
        self._overlay[name] = not self._overlay[name]
        logger.info(f"Overlay '{name}' = {self._overlay[name]}")
        return self._overlay[name]

    def get_overlay_state(self) -> Dict[str, bool]:
        """คืนสถานะ overlay ทั้งหมด"""
        return dict(self._overlay)


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
                        self.face_detector.detect, det_frame)

                self._hand_frame_count += 1
                if self._hand_frame_count >= self._HAND_SKIP:
                    self._hand_frame_count = 0
                    futures['hand'] = self._pool.submit(
                        self._run_hand_detection, det_frame)

                self._body_frame_count += 1
                if self._body_frame_count >= self._BODY_SKIP:
                    self._body_frame_count = 0
                    futures['body'] = self._pool.submit(
                        self._run_body_detection, det_frame)

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

        # 1. Eye overlay (ใช้ EyeRenderer เดิม)
        if self._use_eye() and self._overlay.get('eye', True) and eye_det.get('detected'):
            out = self.eye_renderer.render(out, eye_det)

        # 2. Hand skeleton — วาดจุด landmark + เส้นเชื่อม
        if self._use_hand() and self._overlay.get('hand', True) and hand_det.get('detected'):
            data = hand_det.get('data') or {}
            landmarks = data.get('landmarks')
            alarm     = hand_det.get('alarm', False)
            if landmarks is not None:
                out = HandDetector.draw_landmarks(out, landmarks, alarm=alarm)
            else:
                # fallback: ใช้ hand_renderer เดิมถ้าไม่มี landmarks
                out = self.hand_renderer.render(out, hand_det)

        # 3. Body skeleton — วาดจุด pose + เส้น skeleton (มุมบนขวา)
        if self._use_body() and body_result is not None:
            out = self._render_body_overlay(out, body_result)

        return out

    def _render_body_overlay(self, bgr_frame: np.ndarray,
                              result: BodyResult) -> np.ndarray:
        """วาด body skeleton + ป้ายท่านอน"""
        out = bgr_frame.copy()
        h, w = out.shape[:2]

        # วาด skeleton ถ้ามี landmarks
        if result.raw_landmarks is not None:
            out = BodyDetector.draw_landmarks(
                out, result.raw_landmarks, position=result.position
            )

        # ── ป้ายท่านอน (มุมบนขวา ไม่ทับ eye overlay) ──────────────────────
        position_labels = {
            'SUPINE':     ('Supine (หงาย)',           (100, 200, 100)),
            'LEFT_SIDE':  ('Left Side (ตะแคงซ้าย)',   (100, 180, 255)),
            'RIGHT_SIDE': ('Right Side (ตะแคงขวา)',   (255, 180, 100)),
            'UNKNOWN':    ('Unknown',                  (180, 180, 180)),
        }
        label, color = position_labels.get(result.position, ('Unknown', (180, 180, 180)))
        status  = self._position_tracker.get_status()

        bx1, by1 = w - 300, 8
        bx2, by2 = w - 8,   72

        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 2)

        conf_pct = int(result.confidence * 100)
        cv2.putText(out, f"{label} ({conf_pct}%)",
                    (bx1 + 8, by1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        remaining = int(status.time_until_reposition)
        hrs, mins = divmod(remaining // 60, 60)
        if status.reposition_due:
            time_str, text_color = "REPOSITION NOW!", (0, 0, 255)
        else:
            time_str, text_color = f"Flip in: {hrs}h {mins}m", (220, 220, 220)

        cv2.putText(out, time_str,
                    (bx1 + 8, by1 + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        return out

    # ── Hand detection (with frame-skip) ───────────────────────────────────

    def _run_hand_detection(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """รัน hand detection พร้อม frame-skip ทุก _HAND_SKIP frames"""
        self._hand_frame_count += 1
        if self._hand_frame_count % self._HAND_SKIP != 0:
            return self._last_hand_det  # reuse ผลล่าสุด
        result = self.hand_detector.detect(rgb_frame)
        self._last_hand_det = result
        return result

    # ── Body detection ─────────────────────────────────────────────────────

    def _run_body_detection(self, rgb_frame: np.ndarray) -> Optional[BodyResult]:
        # ── Lazy-load body detector ──────────────────────────────────────────
        if self._body_detector is None:
            self._body_detector = BodyDetector()
            if not self._body_detector.load():
                logger.error("BodyDetector failed to load — disabled")
                self._body_detector = None
                return None

        # ── Frame-skip: รัน MediaPipe ทุก _BODY_SKIP frames ──────────────────
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