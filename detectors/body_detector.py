"""
Body Position Detector
ตรวจจับท่านอนผู้ป่วยติดเตียง 3 ท่า:
  - SUPINE     (หงาย)
  - LEFT_SIDE  (ตะแคงซ้าย)
  - RIGHT_SIDE (ตะแคงขวา)

อ้างอิง: Chiang et al. (2022) - ใช้ shoulder depth difference
Accuracy: 94.52%–99.71% (Setting 4-5)
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from detectors.base import BaseDetector
from utils.logger import setup_logger

logger = setup_logger("body_detector")

# ── Constants ──────────────────────────────────────────────────────────────
POSE_MODEL_PATH          = "models/pose_landmarker_heavy.task"
SHOULDER_DEPTH_THRESHOLD = 0.08
MIN_VISIBILITY           = 0.5

# ── Landmark indices ───────────────────────────────────────────────────────
LM_NOSE           = 0
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24

# ── Pose skeleton connections (MediaPipe 33 landmarks) ────────────────────
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_CONNECTIONS = [
    # Head
    (0, 1), (1, 2), (2, 3), (3, 7),       # nose → left eye path → left ear
    (0, 4), (4, 5), (5, 6), (6, 8),       # nose → right eye path → right ear
    (9, 10),                               # mouth left ↔ right

    # Torso
    (11, 12),                              # left shoulder ↔ right shoulder
    (11, 23), (12, 24),                    # shoulders → hips
    (23, 24),                              # left hip ↔ right hip

    # Left arm
    (11, 13), (13, 15),                    # shoulder → elbow → wrist
    (15, 17), (15, 19), (15, 21),          # wrist → pinky / index / thumb
    (17, 19),                              # pinky ↔ index

    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),

    # Left leg
    (23, 25), (25, 27),                    # hip → knee → ankle
    (27, 29), (27, 31),                    # ankle → heel / foot index
    (29, 31),

    # Right leg
    (24, 26), (26, 28),
    (28, 30), (28, 32),
    (30, 32),
]

# กลุ่ม landmark สำหรับสีต่างกัน
_FACE_LMS    = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
_TORSO_LMS   = {11, 12, 23, 24}
_ARM_LMS     = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
_LEG_LMS     = {25, 26, 27, 28, 29, 30, 31, 32}

# สี BGR
_C_FACE      = (180, 230, 255)  # ฟ้าอ่อน — ใบหน้า
_C_TORSO     = (0, 230, 118)    # เขียว — ลำตัว
_C_ARM       = (255, 200, 0)    # เหลือง — แขน
_C_LEG       = (0, 180, 255)    # ฟ้าสว่าง — ขา
_C_LINE      = (200, 200, 200)  # เทาอ่อน — เส้นเชื่อม
_C_JOINT     = (255, 255, 255)  # ขาว — จุดข้อต่อ


# ── Data class ─────────────────────────────────────────────────────────────
@dataclass
class BodyResult:
    position:            str   = "UNKNOWN"
    confidence:          float = 0.0
    shoulder_depth_diff: float = 0.0
    landmarks_visible:   bool  = False
    # raw landmarks จาก MediaPipe (ส่งไปให้ draw_landmarks ใช้)
    raw_landmarks:       object = field(default=None, repr=False)


# ── Main class ─────────────────────────────────────────────────────────────
class BodyDetector(BaseDetector):

    def __init__(self, model_path: str = POSE_MODEL_PATH):
        self._model_path  = model_path
        self._detector    = None
        self._last_result = BodyResult()
        self._loaded      = False

    # ── BaseDetector interface ─────────────────────────────────────────────

    def load(self) -> bool:
        try:
            base_options = mp_python.BaseOptions(
                model_asset_path=self._model_path
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._detector = mp_vision.PoseLandmarker.create_from_options(options)
            self._loaded   = True
            logger.info("BodyDetector: model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"BodyDetector: failed to load model — {e}")
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def reset(self):
        self._last_result = BodyResult()

    def release(self):
        if self._detector:
            self._detector.close()
            self._detector = None
            self._loaded   = False

    # ── Detect ────────────────────────────────────────────────────────────

    def detect(self, mp_image, timestamp_ms: int) -> BodyResult:
        if not self._loaded or self._detector is None:
            return BodyResult()
        try:
            detection     = self._detector.detect_for_video(mp_image, timestamp_ms)
            result        = self._parse(detection)
            self._last_result = result
            return result
        except Exception as e:
            logger.warning(f"BodyDetector.detect error: {e}")
            return BodyResult()

    @property
    def last_result(self) -> BodyResult:
        return self._last_result

    # ── Drawing ────────────────────────────────────────────────────────────

    @staticmethod
    def draw_landmarks(bgr_frame: np.ndarray,
                       landmarks,
                       position: str = "UNKNOWN") -> np.ndarray:
        """
        วาด pose skeleton ทั้งตัวลงบน BGR frame

        Args:
            bgr_frame:  frame BGR จาก OpenCV
            landmarks:  detection.pose_landmarks[0]  (list of NormalizedLandmark)
            position:   ท่านอนปัจจุบัน — ใช้กำหนดสีกล่อง

        Returns:
            frame ที่วาดแล้ว (copy)
        """
        if landmarks is None:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        out  = bgr_frame.copy()

        # normalized → pixel
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        def lm_visible(idx: int) -> bool:
            return (idx < len(landmarks) and
                    landmarks[idx].visibility >= MIN_VISIBILITY)

        # ── เส้น skeleton ──────────────────────────────────────────────────
        for a, b in POSE_CONNECTIONS:
            if lm_visible(a) and lm_visible(b):
                cv2.line(out, pts[a], pts[b], _C_LINE, 2, cv2.LINE_AA)

        # ── จุด landmark ───────────────────────────────────────────────────
        for i, pt in enumerate(pts):
            if not lm_visible(i):
                continue

            if i in _FACE_LMS:
                color, radius = _C_FACE,  4
            elif i in _TORSO_LMS:
                color, radius = _C_TORSO, 6
            elif i in _ARM_LMS:
                color, radius = _C_ARM,   5
            elif i in _LEG_LMS:
                color, radius = _C_LEG,   5
            else:
                color, radius = _C_JOINT, 4

            cv2.circle(out, pt, radius,     color,    -1, cv2.LINE_AA)
            cv2.circle(out, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        return out

    # ── Internal helpers ───────────────────────────────────────────────────

    def _parse(self, detection) -> BodyResult:
        if not detection.pose_landmarks:
            return BodyResult(position="UNKNOWN", landmarks_visible=False)

        lms      = detection.pose_landmarks[0]
        left_sh  = lms[LM_LEFT_SHOULDER]
        right_sh = lms[LM_RIGHT_SHOULDER]

        if (left_sh.visibility  < MIN_VISIBILITY or
                right_sh.visibility < MIN_VISIBILITY):
            return BodyResult(position="UNKNOWN", landmarks_visible=False,
                              raw_landmarks=lms)

        depth_diff = right_sh.z - left_sh.z
        confidence = min(left_sh.visibility, right_sh.visibility)
        position   = self._classify_position(depth_diff)

        return BodyResult(
            position=position,
            confidence=confidence,
            shoulder_depth_diff=depth_diff,
            landmarks_visible=True,
            raw_landmarks=lms,    # ← เก็บไว้ให้ detector.py เรียก draw
        )

    @staticmethod
    def _classify_position(depth_diff: float) -> str:
        if depth_diff > SHOULDER_DEPTH_THRESHOLD:
            return "LEFT_SIDE"
        elif depth_diff < -SHOULDER_DEPTH_THRESHOLD:
            return "RIGHT_SIDE"
        else:
            return "SUPINE"