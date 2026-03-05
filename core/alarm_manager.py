"""
AlarmManager: thread-safe alarm queue, history, and acknowledgment
"""
import time
import threading
from queue import Queue, Empty
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import setup_logger

logger = setup_logger('alarm_manager')


class AlarmType(Enum):
    EYE_BLINK = "eye_blink"
    HAND_GESTURE = "hand_gesture"
    BODY_POSITION = "body_position"


@dataclass
class AlarmEvent:
    type: AlarmType
    timestamp: float
    metadata: Dict = field(default_factory=dict)
    acknowledged: bool = False


class AlarmManager:
    """จัดการ alarm queue และ notifications (thread-safe)"""

    def __init__(self, max_history: int = 100):
        self.queue: Queue = Queue()
        self.history: List[AlarmEvent] = []
        self.max_history = max_history
        self.lock = threading.Lock()
        self.active_alarm: Optional[AlarmEvent] = None
        # Optional callback: fn(alarm_type: str, metadata: dict)
        self.on_alarm_triggered = None

    def trigger_alarm(self, alarm_type: AlarmType, metadata: Dict = None):
        """เพิ่ม alarm ใหม่เข้า queue"""
        event = AlarmEvent(
            type=alarm_type,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        with self.lock:
            self.queue.put(event)
            self.history.append(event)
            if len(self.history) > self.max_history:
                self.history.pop(0)

            # Set as active if none currently active
            if self.active_alarm is None:
                self.active_alarm = event
                logger.info(f"Alarm triggered: {alarm_type.value}")

        # เรียก callback นอก lock เพื่อหลีกเลี่ยง deadlock
        if self.on_alarm_triggered:
            try:
                self.on_alarm_triggered(alarm_type.value, metadata or {})
            except Exception as e:
                logger.warning(f"on_alarm_triggered callback error: {e}")

    def get_active_alarm(self) -> Optional[AlarmEvent]:
        """ดึง alarm ที่ active อยู่ (ไม่ acknowledge อัตโนมัติ)"""
        with self.lock:
            return self.active_alarm

    def acknowledge_alarm(self) -> bool:
        """รับทราบ alarm ปัจจุบัน และดึงอันถัดไปจาก queue"""
        with self.lock:
            if self.active_alarm is not None:
                self.active_alarm.acknowledged = True
                logger.info(f"Alarm acknowledged: {self.active_alarm.type.value}")
                self.active_alarm = None

                # Pull next from queue if available
                try:
                    self.active_alarm = self.queue.get_nowait()
                    logger.info(f"Next alarm activated: {self.active_alarm.type.value}")
                except Empty:
                    pass

                return True
            return False

    def get_history(self, limit: int = 10) -> List[Dict]:
        """ดึงประวัติ alarm ล่าสุด"""
        with self.lock:
            recent = self.history[-limit:]
            return [
                {
                    'type': e.type.value,
                    'timestamp': e.timestamp,
                    'acknowledged': e.acknowledged,
                    'metadata': {k: str(v) for k, v in e.metadata.items()
                                 if not hasattr(v, '__len__') or len(str(v)) < 100}
                }
                for e in reversed(recent)
            ]

    def has_active_alarm(self) -> bool:
        """เช็คว่ามี alarm ที่รอ acknowledge หรือไม่"""
        with self.lock:
            return self.active_alarm is not None
