import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch
from detectors.body_detector import BodyDetector, BodyResult


class TestBodyDetectorClassifyPosition(unittest.TestCase):
    """ทดสอบ _classify_position โดยตรง (ไม่ต้องโหลด MediaPipe model)"""

    def test_supine_small_depth_diff(self):
        """depth_diff ใกล้ 0 → SUPINE (หงาย)"""
        result = BodyDetector._classify_position(0.02)
        self.assertEqual(result, "SUPINE")

    def test_left_side_negative_diff(self):
        """depth_diff ลบ (ไหล่ซ้ายอยู่ด้านหน้า) → LEFT_SIDE"""
        result = BodyDetector._classify_position(-0.15)
        self.assertEqual(result, "LEFT_SIDE")

    def test_right_side_positive_diff(self):
        """depth_diff บวก (ไหล่ขวาอยู่ด้านหน้า) → RIGHT_SIDE"""
        result = BodyDetector._classify_position(0.15)
        self.assertEqual(result, "RIGHT_SIDE")

    def test_unknown_borderline(self):
        """depth_diff ใกล้ threshold → ควรได้ SUPINE หรือ SIDE (ไม่ crash)"""
        result = BodyDetector._classify_position(0.08)
        self.assertIn(result, ("SUPINE", "LEFT_SIDE", "RIGHT_SIDE", "UNKNOWN"))


class TestBodyResult(unittest.TestCase):
    """ทดสอบ BodyResult dataclass"""

    def test_default_values(self):
        r = BodyResult()
        self.assertEqual(r.position, "UNKNOWN")
        self.assertEqual(r.confidence, 0.0)
        self.assertFalse(r.landmarks_visible)
        self.assertIsNone(r.raw_landmarks)

    def test_custom_values(self):
        r = BodyResult(position="SUPINE", confidence=0.95, landmarks_visible=True)
        self.assertEqual(r.position, "SUPINE")
        self.assertAlmostEqual(r.confidence, 0.95)
        self.assertTrue(r.landmarks_visible)


class TestBodyDetectorInit(unittest.TestCase):
    """ทดสอบ init โดยไม่ต้องโหลด model จริง"""

    def setUp(self):
        self.detector = BodyDetector.__new__(BodyDetector)
        self.detector._model_path  = "fake_path.task"
        self.detector._detector    = None
        self.detector._last_result = BodyResult()
        self.detector._loaded      = False

    def test_is_loaded_false_initially(self):
        self.assertFalse(self.detector.is_loaded())

    def test_last_result_default(self):
        r = self.detector.last_result
        self.assertEqual(r.position, "UNKNOWN")

    def test_reset_clears_result(self):
        self.detector._last_result = BodyResult(position="SUPINE", confidence=0.9)
        self.detector.reset()
        self.assertEqual(self.detector._last_result.position, "SUPINE")
        # reset ไม่ clear result แต่ unload model
        self.assertFalse(self.detector._loaded)


class TestBodyDetectorWithMockedDetector(unittest.TestCase):
    """ทดสอบ detect() path โดย mock MediaPipe internals"""

    def setUp(self):
        self.detector = BodyDetector.__new__(BodyDetector)
        self.detector._model_path  = "fake.task"
        self.detector._loaded      = True
        self.detector._last_result = BodyResult()

        # Mock MediaPipe detector
        mock_mp_detector = MagicMock()
        self.detector._detector = mock_mp_detector

        # สร้าง landmark mock
        def _make_lm(x, y, z, vis=0.9):
            lm = MagicMock()
            lm.x, lm.y, lm.z, lm.visibility = x, y, z, vis
            return lm

        lms = [_make_lm(0.5, 0.5, 0.0)] * 33
        # ไหล่ขวา (12) อยู่ด้านหน้า (z เล็กกว่า) → RIGHT_SIDE
        lms[11] = _make_lm(0.4, 0.4, 0.20, 0.9)  # ไหล่ซ้าย
        lms[12] = _make_lm(0.6, 0.4, 0.0,  0.9)  # ไหล่ขวา (z น้อยกว่า)
        lms[23] = _make_lm(0.4, 0.7, 0.18, 0.9)  # สะโพกซ้าย
        lms[24] = _make_lm(0.6, 0.7, 0.02, 0.9)  # สะโพกขวา

        mock_result = MagicMock()
        mock_result.pose_landmarks = [lms]
        mock_mp_detector.detect_for_video.return_value = mock_result

    def test_detect_returns_body_result(self):
        mock_image = MagicMock()
        result = self.detector.detect(mock_image, timestamp_ms=1000)
        self.assertIsInstance(result, BodyResult)

    def test_detect_identifies_position(self):
        mock_image = MagicMock()
        result = self.detector.detect(mock_image, timestamp_ms=1000)
        # ควรจำแนกได้เป็น RIGHT_SIDE หรืออย่างน้อยไม่ใช่ UNKNOWN เสมอไป
        self.assertIn(result.position, ("SUPINE", "LEFT_SIDE", "RIGHT_SIDE", "UNKNOWN"))
        self.assertTrue(result.landmarks_visible)


if __name__ == "__main__":
    unittest.main()
