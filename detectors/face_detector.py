"""
FaceDetector: eye blink detection using MediaPipe Face Landmarker
"""
import time
import numpy as np
import mediapipe as mp
from collections import deque

from .base import BaseDetector
from core.state_machine import BlinkStateMachine
from utils.geometry import calculate_ear, estimate_gaze, face_distance_ratio
from utils.model_loader import load_face_landmarker
from utils.logger import setup_logger
from config import LEFT_EYE, RIGHT_EYE, FACE_CONSTANTS, ZOOM_CONSTANTS

logger = setup_logger('face_detector')

# EAR smoothing — moving average window
_EAR_SMOOTH_N = 5


class FaceDetector(BaseDetector):
    """ตรวจจับใบหน้า + การกระพิบตา"""

    def __init__(self, config=None):
        super().__init__(config)

        logger.info("Loading face landmarker model...")
        self.landmarker = load_face_landmarker()
        logger.info("Face landmarker loaded.")

        # Blink state machine
        self.state_machine = BlinkStateMachine()

        # Blink mode: FAR, NEAR, ZOOMING
        self.blink_mode = "FAR"
        self.zoom_start: float = 0.0

        # EAR smoothing buffer
        self._ear_buf: deque = deque(maxlen=_EAR_SMOOTH_N)

    def detect(self, frame: np.ndarray) -> dict:
        """ตรวจจับ face + blink จาก frame"""
        h, w = frame.shape[:2]
        now = time.time()

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        try:
            result = self.landmarker.detect(mp_image)
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return self._no_detection()

        if not result.face_landmarks:
            self._ear_buf.clear()   # reset buffer เมื่อไม่เห็นหน้า
            return self._no_detection()

        face_lm = result.face_landmarks[0]

        # Calculate raw EAR
        ear_left, pts_left   = calculate_ear(face_lm, LEFT_EYE, w, h)
        ear_right, pts_right = calculate_ear(face_lm, RIGHT_EYE, w, h)
        raw_ear = (ear_left + ear_right) / 2.0

        # ── EAR Smoothing (moving average) ────────────────────────────────
        self._ear_buf.append(raw_ear)
        smoothed_ear = sum(self._ear_buf) / len(self._ear_buf)

        yaw, pitch, gaze_ok = estimate_gaze(face_lm, w, h)
        ratio = face_distance_ratio(face_lm, w, h)

        # Update blink mode based on face distance
        self._update_blink_mode(ratio, now)

        # Only process blink if gaze is towards camera
        events = {}
        if gaze_ok:
            # ส่ง smoothed EAR เข้า state machine แทนค่าดิบ
            events = self.state_machine.update(smoothed_ear, now, self.blink_mode)

        alarm = events.get('alarm', False) or events.get('sos_done', False)

        return {
            'detected': True,
            'confidence': 1.0,
            'alarm': alarm,
            'data': {
                'ear': smoothed_ear,        # แสดงค่า smooth ใน HUD
                'ear_raw': raw_ear,         # เก็บค่าดิบไว้ด้วยถ้าต้องการ debug
                'ear_left': ear_left,
                'ear_right': ear_right,
                'eye_pts_left': pts_left,
                'eye_pts_right': pts_right,
                'gaze': {'yaw': yaw, 'pitch': pitch, 'ok': gaze_ok},
                'distance_ratio': ratio,
                'blink_mode': self.blink_mode,
                'events': events,
                'sos_phase': self.state_machine.sos_phase,
                'p1_count': self.state_machine.p1_count,
                'near_blinks': len(self.state_machine.near_blinks)
            }
        }

    def _update_blink_mode(self, ratio: float, now: float):
        """Update blink mode based on face distance"""
        close_ratio = FACE_CONSTANTS['CLOSE_RATIO']
        far_ratio = FACE_CONSTANTS['FAR_RATIO']
        zoom_dur = ZOOM_CONSTANTS['DURATION']

        if self.blink_mode == "FAR" and ratio > close_ratio:
            self.blink_mode = "NEAR"
            logger.debug("Switched to NEAR mode")
        elif self.blink_mode == "NEAR" and ratio < far_ratio:
            self.blink_mode = "FAR"
            logger.debug("Switched to FAR mode")
        elif self.blink_mode == "ZOOMING":
            if now - self.zoom_start > zoom_dur:
                self.blink_mode = "NEAR"
                logger.debug("Zoom complete → NEAR mode")

    def _no_detection(self) -> dict:
        return {
            'detected': False,
            'confidence': 0.0,
            'alarm': False,
            'data': None
        }

    def reset(self):
        self.state_machine.reset()
        self.blink_mode = "FAR"
        self.zoom_start = 0.0
        self._ear_buf.clear()
