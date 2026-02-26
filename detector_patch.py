"""
detector_patch.py  —  Phase 5 Integration Guide
================================================
ไฟล์นี้ไม่ใช่ไฟล์ที่รันตรงๆ แต่เป็น snippet ให้ copy-paste
เข้าไปใน core/detector.py ที่มีอยู่แล้ว

ส่วนที่ต้องเพิ่ม:
  1. import BodyDetector, PositionTracker, NotificationManager
  2. สร้าง instance ใน __init__
  3. เรียก body_detector.detect() ใน process_frame()
  4. เรียก position_tracker.update() และ notification_manager.tick()
  5. รวม body data เข้า return dict

แก้ core/detector.py ตามตัวอย่างด้านล่าง:
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. เพิ่ม imports ต่อท้าย import block เดิม
# ─────────────────────────────────────────────────────────────────────────────
IMPORTS_TO_ADD = """
from detectors.body_detector import BodyDetector, BodyResult
from core.position_tracker import PositionTracker
from core.notification_manager import NotificationManager
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. ใน Detector.__init__()  เพิ่มหลัง self._hand_detector = ...
# ─────────────────────────────────────────────────────────────────────────────
INIT_ADDITIONS = """
        # Body detection (lazy-load เหมือน face/hand)
        self._body_detector: Optional[BodyDetector] = None

        # Position tracking & notifications
        self._position_tracker   = PositionTracker()
        self._notification_manager = NotificationManager()

        # Timestamp counter สำหรับ Pose model (ต้องเพิ่มขึ้นตลอด)
        self._pose_timestamp_ms: int = 0
"""

# ─────────────────────────────────────────────────────────────────────────────
# 3. ใน process_frame() — เพิ่มหลัง hand detection block
# ─────────────────────────────────────────────────────────────────────────────
PROCESS_FRAME_ADDITION = """
        # ── Body Detection ──────────────────────────────────────────────────
        body_result: Optional[BodyResult] = None
        if self._mode == "BODY" or self._body_detect_always:
            body_result = self._run_body_detection(mp_image)

        # ── Position Tracking ───────────────────────────────────────────────
        if body_result is not None:
            self._position_tracker.update(body_result.position)

        # ── Notification Tick ───────────────────────────────────────────────
        tracker_status = self._position_tracker.get_status()
        new_notifications = self._notification_manager.tick(
            reposition_due=tracker_status.reposition_due
        )
        if new_notifications:
            for notif in new_notifications:
                # ส่งต่อไปยัง alarm_manager (หรือ WebSocket) ที่มีอยู่แล้ว
                self._alarm_manager.push_notification(notif.to_dict())
"""

# ─────────────────────────────────────────────────────────────────────────────
# 4. helper method ที่ต้องเพิ่มใน Detector class
# ─────────────────────────────────────────────────────────────────────────────
HELPER_METHOD = """
    def _run_body_detection(self, mp_image) -> Optional["BodyResult"]:
        # Lazy-load
        if self._body_detector is None:
            self._body_detector = BodyDetector()
            if not self._body_detector.load():
                logger.error("Detector: BodyDetector failed to load")
                self._body_detector = None
                return None

        self._pose_timestamp_ms += 33   # ~30 FPS
        return self._body_detector.detect(mp_image, self._pose_timestamp_ms)

    def get_position_status(self) -> dict:
        status = self._position_tracker.get_status()
        return {
            "current_position":      status.current_position,
            "current_duration":      status.current_duration,
            "time_until_reposition": status.time_until_reposition,
            "reposition_due":        status.reposition_due,
            "reposition_interval":   status.reposition_interval,
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
"""

# ─────────────────────────────────────────────────────────────────────────────
# 5. เพิ่ม body data ใน return dict ของ process_frame()
# ─────────────────────────────────────────────────────────────────────────────
RETURN_DICT_ADDITION = """
        # เพิ่มใน return dict เดิม:
        result.update({
            "body": {
                "position":   body_result.position if body_result else "UNKNOWN",
                "confidence": body_result.confidence if body_result else 0.0,
            },
            "position_status": self.get_position_status(),
        })
"""
