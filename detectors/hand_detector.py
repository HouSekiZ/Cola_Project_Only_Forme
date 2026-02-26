"""
HandDetector: hand gesture detection using MediaPipe Hand Landmarker
Alarm triggers on OPEN → FIST transition
"""
import cv2
import numpy as np
import mediapipe as mp

from .base import BaseDetector
from utils.model_loader import load_hand_landmarker
from utils.geometry import check_hand_state
from utils.logger import setup_logger

logger = setup_logger('hand_detector')

# ── Hand skeleton connections (MediaPipe 21 landmarks) ─────────────────────
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

# ปลายนิ้ว — วาดจุดใหญ่กว่า
FINGERTIPS = {4, 8, 12, 16, 20}

# สี BGR
_C_LINE      = (0, 230, 118)    # เขียวสด — เส้น skeleton
_C_JOINT     = (255, 255, 255)  # ขาว — ข้อนิ้ว
_C_FINGERTIP = (0, 200, 255)    # ฟ้า — ปลายนิ้ว
_C_WRIST     = (255, 180, 0)    # ส้ม — ข้อมือ
_C_ALARM     = (0, 0, 255)      # แดง — เมื่อ alarm


class HandDetector(BaseDetector):
    """ตรวจจับมือ + ท่าทาง"""

    def __init__(self, config=None):
        super().__init__(config)
        logger.info("Loading hand landmarker model...")
        self.landmarker = load_hand_landmarker()
        logger.info("Hand landmarker loaded.")
        self.last_hand_state: str = None

    # ── Detection ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """ตรวจจับ hand gesture จาก RGB frame"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        try:
            result = self.landmarker.detect(mp_image)
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
            return self._no_detection()

        if not result.hand_landmarks:
            return self._no_detection()

        landmarks  = result.hand_landmarks[0]
        state      = check_hand_state(landmarks)
        alarm      = (self.last_hand_state == "OPEN" and state == "FIST")
        handedness = (result.handedness[0][0].display_name
                      if result.handedness else 'Unknown')

        if alarm:
            logger.info("HAND gesture OPEN→FIST detected! ALARM triggered.")

        self.last_hand_state = state

        return {
            'detected':   True,
            'confidence': 1.0,
            'alarm':      alarm,
            'data': {
                'state':      state,
                'last_state': self.last_hand_state,
                'handedness': handedness,
                'landmarks':  landmarks,   # ← ส่ง raw landmarks ไปให้ renderer
            }
        }

    # ── Drawing ────────────────────────────────────────────────────────────

    @staticmethod
    def draw_landmarks(bgr_frame: np.ndarray,
                       landmarks,
                       alarm: bool = False) -> np.ndarray:
        """
        วาด hand skeleton ลงบน BGR frame

        Args:
            bgr_frame:  frame BGR จาก OpenCV
            landmarks:  result.hand_landmarks[0]  (list of NormalizedLandmark)
            alarm:      True → วาดสีแดงทั้งหมด (alarm state)

        Returns:
            frame ที่วาดแล้ว (copy)
        """
        if landmarks is None:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        out  = bgr_frame.copy()

        # normalized → pixel coordinates
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        line_color = _C_ALARM if alarm else _C_LINE

        # ── เส้น skeleton ──────────────────────────────────────────────────
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], line_color, 2, cv2.LINE_AA)

        # ── จุด landmark ───────────────────────────────────────────────────
        for i, pt in enumerate(pts):
            if alarm:
                color, radius = _C_ALARM, 5
            elif i == 0:                    # ข้อมือ (WRIST)
                color, radius = _C_WRIST, 7
            elif i in FINGERTIPS:           # ปลายนิ้ว
                color, radius = _C_FINGERTIP, 6
            else:                           # ข้อนิ้วกลาง
                color, radius = _C_JOINT, 4

            cv2.circle(out, pt, radius,     color,    -1, cv2.LINE_AA)  # fill
            cv2.circle(out, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)  # outline

        return out

    # ── Helpers ────────────────────────────────────────────────────────────

    def _no_detection(self) -> dict:
        return {
            'detected':   False,
            'confidence': 0.0,
            'alarm':      False,
            'data':       None,
        }

    def reset(self):
        self.last_hand_state = None