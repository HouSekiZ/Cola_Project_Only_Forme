"""
Notification Manager
จัดการการแจ้งเตือน 2 ประเภท:
  1. Repositioning Reminder  — ทุก 2 ชั่วโมง (ป้องกันแผลกดทับ)
  2. Meal Time Notifications  — เช้า 07:00 / เที่ยง 12:00 / เย็น 18:00
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from collections import deque

from utils.logger import setup_logger

logger = setup_logger("notification_manager")

# ── constants ──────────────────────────────────────────────────────────────
DEFAULT_MEAL_TIMES = {
    "breakfast": "07:00",
    "lunch":     "12:00",
    "dinner":    "18:00",
}

MEAL_NOTIFY_WINDOW_MINUTES = 15   # แจ้งเตือนถ้าอยู่ภายใน 15 นาทีหลังเวลาอาหาร
MAX_NOTIFICATION_HISTORY   = 100


# ── data classes ──────────────────────────────────────────────────────────
@dataclass
class Notification:
    type: str           # "reposition" | "meal"
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type":          self.type,
            "title":         self.title,
            "message":       self.message,
            "timestamp":     self.timestamp,
            "acknowledged":  self.acknowledged,
            "extra":         self.extra,
        }


@dataclass
class MealStatus:
    name: str           # breakfast | lunch | dinner
    scheduled_time: str # "HH:MM"
    eaten: bool = False
    eaten_at: Optional[float] = None
    notified: bool = False   # ส่งการแจ้งเตือนแล้วหรือยัง (วันนี้)

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "scheduled_time": self.scheduled_time,
            "eaten":          self.eaten,
            "eaten_at":       self.eaten_at,
        }


# ── main class ─────────────────────────────────────────────────────────────
class NotificationManager:
    """
    ตรวจสอบและส่ง notification สำหรับ:
      - Repositioning (ดึงข้อมูลจาก PositionTracker)
      - Meal times (schedule ตาม config)

    Usage:
        nm = NotificationManager()
        nm.set_meal_times({"breakfast": "07:00", "lunch": "12:00", "dinner": "18:00"})

        # เรียกทุก ~1 วินาที
        new_notifications = nm.tick(position_tracker_status)

        nm.mark_meal_eaten("lunch")
        nm.acknowledge_reposition()
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Reposition state
        self._reposition_notified = False   # ป้องกัน spam
        self._pending_notifications: deque[Notification] = deque(maxlen=50)
        self._history: deque[Notification] = deque(maxlen=MAX_NOTIFICATION_HISTORY)

        # Meal state (reset ทุกวันเที่ยงคืน)
        self._meals: dict[str, MealStatus] = {}
        self._meal_reset_date: Optional[str] = None  # "YYYY-MM-DD"
        self.set_meal_times(DEFAULT_MEAL_TIMES)

    # ── public API ─────────────────────────────────────────────────────────

    def tick(self, reposition_due: bool) -> list[Notification]:
        """
        เรียกทุก ~1 วินาที
        Args:
            reposition_due: จาก PositionTracker.get_status().reposition_due
        Returns:
            list ของ Notification ใหม่ที่เพิ่งเกิด
        """
        new: list[Notification] = []
        self._reset_meals_if_new_day()

        with self._lock:
            # --- Reposition check ---
            reposition_notif = self._check_reposition(reposition_due)
            if reposition_notif:
                new.append(reposition_notif)

            # --- Meal check ---
            meal_notifs = self._check_meals()
            new.extend(meal_notifs)

            # เก็บ history
            for n in new:
                self._history.append(n)
                self._pending_notifications.append(n)

        return new

    def pop_pending(self) -> list[Notification]:
        """ดึง notification ที่ยังไม่ได้ส่ง frontend (drain queue)"""
        with self._lock:
            items = list(self._pending_notifications)
            self._pending_notifications.clear()
        return items

    def acknowledge_reposition(self):
        """พยาบาลกด 'รับทราบ' หรือ 'พลิกตัวแล้ว'"""
        with self._lock:
            self._reposition_notified = False
        logger.info("NotificationManager: reposition acknowledged")

    def mark_meal_eaten(self, meal_name: str) -> bool:
        """บันทึกว่าทานอาหารแล้ว"""
        with self._lock:
            if meal_name not in self._meals:
                return False
            self._meals[meal_name].eaten = True
            self._meals[meal_name].eaten_at = time.time()
            logger.info(f"NotificationManager: meal '{meal_name}' marked as eaten")
            return True

    def set_meal_times(self, meal_times: dict[str, str]):
        """
        ตั้งเวลาอาหาร
        Args:
            meal_times: {"breakfast": "07:00", "lunch": "12:00", "dinner": "18:00"}
        """
        with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            self._meal_reset_date = today
            self._meals = {
                name: MealStatus(name=name, scheduled_time=t)
                for name, t in meal_times.items()
            }
        logger.info(f"NotificationManager: meal times set — {meal_times}")

    def get_meal_status(self) -> list[dict]:
        with self._lock:
            return [m.to_dict() for m in self._meals.values()]

    def get_notification_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            items = list(self._history)
            items.reverse()
            return [n.to_dict() for n in items[:limit]]

    # ── internal helpers ───────────────────────────────────────────────────

    def _check_reposition(self, reposition_due: bool) -> Optional[Notification]:
        """ส่ง notification ครั้งเดียวเมื่อถึงเวลาพลิกตัว"""
        if reposition_due and not self._reposition_notified:
            self._reposition_notified = True
            notif = Notification(
                type="reposition",
                title="ถึงเวลาพลิกตัวผู้ป่วย",
                message="ผู้ป่วยอยู่ในท่าเดิมนานเกิน 2 ชั่วโมง กรุณาพลิกตัวเพื่อป้องกันแผลกดทับ",
                extra={"sound": "reposition_notification"},
            )
            logger.warning("NotificationManager: REPOSITION DUE")
            return notif

        # ถ้า reposition ได้รับการ reset แล้ว → รีเซ็ต flag เพื่อให้แจ้งครั้งต่อไปได้
        if not reposition_due and self._reposition_notified:
            self._reposition_notified = False

        return None

    def _check_meals(self) -> list[Notification]:
        """ตรวจสอบมื้ออาหารที่ถึงเวลาแต่ยังไม่ได้แจ้งเตือน"""
        now = datetime.now()
        result = []

        for meal in self._meals.values():
            if meal.notified or meal.eaten:
                continue

            scheduled = self._parse_meal_time(meal.scheduled_time, now)
            if scheduled is None:
                continue

            delta_minutes = (now - scheduled).total_seconds() / 60

            # แจ้งเตือนถ้าผ่านเวลาอาหารไปแล้วแต่ไม่เกิน window
            if 0 <= delta_minutes <= MEAL_NOTIFY_WINDOW_MINUTES:
                meal.notified = True
                label = {"breakfast": "เช้า", "lunch": "เที่ยง", "dinner": "เย็น"}.get(
                    meal.name, meal.name
                )
                notif = Notification(
                    type="meal",
                    title=f"เวลาอาหาร{label}",
                    message=f"ถึงเวลาอาหาร{label} ({meal.scheduled_time}) กรุณาให้อาหารผู้ป่วย",
                    extra={"meal": meal.name, "sound": "meal_notification"},
                )
                logger.info(f"NotificationManager: meal notification — {meal.name}")
                result.append(notif)

        return result

    def _reset_meals_if_new_day(self):
        """Reset meal status ทุกวันเที่ยงคืน"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._meal_reset_date:
            with self._lock:
                self._meal_reset_date = today
                for meal in self._meals.values():
                    meal.eaten = False
                    meal.eaten_at = None
                    meal.notified = False
            logger.info("NotificationManager: daily meal status reset")

    @staticmethod
    def _parse_meal_time(time_str: str, reference: datetime) -> Optional[datetime]:
        """แปลง "HH:MM" เป็น datetime วันนี้"""
        try:
            h, m = map(int, time_str.split(":"))
            return reference.replace(hour=h, minute=m, second=0, microsecond=0)
        except ValueError:
            logger.error(f"NotificationManager: invalid meal time format '{time_str}'")
            return None