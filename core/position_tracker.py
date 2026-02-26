"""
Position Tracker
ติดตามท่านอนผู้ป่วย:
  - นับเวลาที่อยู่ท่าเดิมติดต่อกัน
  - บันทึก position history
  - เช็คว่าถึงเวลาพลิกตัวหรือยัง (default: 2 ชั่วโมง)
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from utils.logger import setup_logger

logger = setup_logger("position_tracker")

# ── constants ──────────────────────────────────────────────────────────────
DEFAULT_REPOSITION_INTERVAL = 2 * 60 * 60   # 2 ชั่วโมง (วินาที)
MAX_HISTORY_ENTRIES         = 200            # เก็บ history สูงสุด 200 รายการ
POSITION_CONFIRM_SECONDS    = 3.0            # ต้องอยู่ท่าเดิมนานกว่านี้จึง "ยืนยัน"


# ── data classes ──────────────────────────────────────────────────────────
@dataclass
class PositionEntry:
    """บันทึกหนึ่งช่วงที่ผู้ป่วยอยู่ในท่าหนึ่ง"""
    position: str
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    @property
    def duration(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration, 1),
        }


@dataclass
class TrackerStatus:
    current_position: str = "UNKNOWN"
    current_duration: float = 0.0          # วินาทีที่อยู่ท่าปัจจุบัน
    reposition_interval: float = DEFAULT_REPOSITION_INTERVAL
    time_until_reposition: float = 0.0     # วินาทีที่เหลือก่อนต้องพลิก
    reposition_due: bool = False
    last_repositioned_at: Optional[float] = None
    total_entries: int = 0


# ── main class ─────────────────────────────────────────────────────────────
class PositionTracker:
    """
    รับ position string ต่อเนื่องและติดตามสถานะ

    Usage:
        tracker = PositionTracker(reposition_interval=7200)
        tracker.update("SUPINE")
        status = tracker.get_status()
        history = tracker.get_history()
        tracker.reset_reposition_timer()
    """

    def __init__(self, reposition_interval: float = DEFAULT_REPOSITION_INTERVAL):
        self._interval = reposition_interval
        self._history: deque[PositionEntry] = deque(maxlen=MAX_HISTORY_ENTRIES)

        self._current_entry: Optional[PositionEntry] = None
        self._pending_position: Optional[str] = None
        self._pending_since: float = 0.0

        self._last_repositioned_at: float = time.time()

    # ── public API ─────────────────────────────────────────────────────────
    def update(self, position: str) -> bool:
        """
        อัปเดตท่าปัจจุบัน (เรียกทุก frame)

        Returns:
            True ถ้าท่าเพิ่งเปลี่ยน (confirmed)
        """
        now = time.time()

        # กรอง UNKNOWN ออก — ไม่นับเป็น position จริง
        if position == "UNKNOWN":
            self._pending_position = None
            return False

        # --- debounce: ต้องอยู่ท่าเดิมนาน POSITION_CONFIRM_SECONDS ก่อนยืนยัน ---
        if position != self._pending_position:
            self._pending_position = position
            self._pending_since = now
            return False

        confirmed_position = position
        if now - self._pending_since < POSITION_CONFIRM_SECONDS:
            return False  # ยังรอ confirm

        # --- ถ้าท่าเดียวกับปัจจุบัน ไม่ต้องทำอะไร ---
        if (self._current_entry is not None and
                self._current_entry.position == confirmed_position):
            return False

        # --- ท่าเปลี่ยนแล้ว → บันทึก entry เดิม, เริ่ม entry ใหม่ ---
        changed = self._commit_new_position(confirmed_position, now)
        return changed

    def reset_reposition_timer(self):
        """พยาบาลกด 'พลิกตัวแล้ว' → reset timer"""
        self._last_repositioned_at = time.time()
        logger.info("PositionTracker: reposition timer reset")

    def set_interval(self, seconds: float):
        """ปรับ interval (วินาที)"""
        self._interval = max(60.0, seconds)

    def get_status(self) -> TrackerStatus:
        now = time.time()
        current_pos = self._current_entry.position if self._current_entry else "UNKNOWN"
        current_dur = self._current_entry.duration if self._current_entry else 0.0

        elapsed_since_reposition = now - self._last_repositioned_at
        time_until = max(0.0, self._interval - elapsed_since_reposition)
        due = elapsed_since_reposition >= self._interval

        return TrackerStatus(
            current_position=current_pos,
            current_duration=round(current_dur, 1),
            reposition_interval=self._interval,
            time_until_reposition=round(time_until, 1),
            reposition_due=due,
            last_repositioned_at=self._last_repositioned_at,
            total_entries=len(self._history),
        )

    def get_history(self, limit: int = 50) -> list[dict]:
        """คืน history ล่าสุด limit รายการ (เรียงใหม่ → เก่า)"""
        entries = list(self._history)
        entries.reverse()
        return [e.to_dict() for e in entries[:limit]]

    # ── internal ──────────────────────────────────────────────────────────
    def _commit_new_position(self, new_position: str, now: float) -> bool:
        # ปิด entry เดิม
        if self._current_entry is not None:
            self._current_entry.ended_at = now
            self._history.append(self._current_entry)
            logger.debug(
                f"Position ended: {self._current_entry.position} "
                f"({self._current_entry.duration:.1f}s)"
            )

        # เปิด entry ใหม่
        self._current_entry = PositionEntry(position=new_position, started_at=now)
        logger.info(f"Position changed → {new_position}")
        return True