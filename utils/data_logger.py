"""
Data Logger
บันทึกข้อมูลกิจกรรมผู้ป่วยลงไฟล์ JSON:
  - Position history (ท่านอน + ระยะเวลา)
  - Alarm events   (blink / hand / flip)
  - Meal records   (มื้ออาหารที่ทาน)

Usage:
    logger = DataLogger()
    logger.log_position("SUPINE", 3600.0)
    logger.log_alarm("blink", {"mode": "NEAR", "blinks": 3})
    logger.log_meal("breakfast")
    records = logger.get_records()
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from collections import deque

from utils.logger import setup_logger

_log = setup_logger("data_logger")

# ── constants ──────────────────────────────────────────────────────────────
DEFAULT_LOG_FILE    = Path("logs") / "patient_log.json"
MAX_IN_MEMORY       = 500          # จำนวนสูงสุดที่เก็บใน RAM
FLUSH_EVERY_N       = 10           # เขียนไฟล์ทุกๆ N รายการที่เพิ่มเข้ามา

AlarmType = Literal["blink", "hand", "flip"]

# ── data classes ──────────────────────────────────────────────────────────

@dataclass
class PositionRecord:
    position: str
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlarmRecord:
    alarm_type: str           # "blink" | "hand" | "flip"
    details: dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MealRecord:
    meal_name: str            # "breakfast" | "lunch" | "dinner"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


# ── main class ─────────────────────────────────────────────────────────────

class DataLogger:
    """
    Thread-safe (single-threaded) logger สำหรับบันทึกข้อมูลผู้ป่วย

    ข้อมูลเก็บทั้ง in-memory และ persist ลงไฟล์ JSON อัตโนมัติ
    """

    def __init__(self, log_file: str | Path = DEFAULT_LOG_FILE):
        self._log_file = Path(log_file)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        # in-memory queues (deque จำกัดขนาด)
        self._positions: deque[PositionRecord] = deque(maxlen=MAX_IN_MEMORY)
        self._alarms:    deque[AlarmRecord]    = deque(maxlen=MAX_IN_MEMORY)
        self._meals:     deque[MealRecord]     = deque(maxlen=MAX_IN_MEMORY)

        self._write_count = 0

        # โหลดข้อมูลเดิม (ถ้ามี)
        self._load()

    # ── public API ─────────────────────────────────────────────────────────

    def log_position(self, position: str, duration_seconds: float) -> None:
        """บันทึกท่านอน + ระยะเวลาที่อยู่ท่านั้น"""
        record = PositionRecord(position=position, duration_seconds=round(duration_seconds, 1))
        self._positions.append(record)
        _log.info(f"Position logged: {position} ({duration_seconds:.0f}s)")
        self._maybe_flush()

    def log_alarm(self, alarm_type: str, details: Optional[dict] = None) -> None:
        """บันทึก alarm event"""
        record = AlarmRecord(alarm_type=alarm_type, details=details or {})
        self._alarms.append(record)
        _log.info(f"Alarm logged: {alarm_type} — {details}")
        self._maybe_flush()

    def log_meal(self, meal_name: str) -> None:
        """บันทึกว่าผู้ป่วยทานอาหารแล้ว"""
        record = MealRecord(meal_name=meal_name)
        self._meals.append(record)
        _log.info(f"Meal logged: {meal_name}")
        self._maybe_flush()

    def get_records(self, limit: int = 50) -> dict:
        """คืน records ล่าสุด limit รายการในแต่ละหมวด"""
        def tail(dq: deque, n: int) -> list:
            items = list(dq)
            return [r.to_dict() for r in items[-n:]][::-1]

        return {
            "positions":  tail(self._positions, limit),
            "alarms":     tail(self._alarms,    limit),
            "meals":      tail(self._meals,      limit),
        }

    def flush(self) -> None:
        """บันทึกข้อมูลทั้งหมดลงไฟล์ทันที"""
        data = {
            "positions": [r.to_dict() for r in self._positions],
            "alarms":    [r.to_dict() for r in self._alarms],
            "meals":     [r.to_dict() for r in self._meals],
            "saved_at":  datetime.now().isoformat(),
        }
        try:
            with open(self._log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            _log.debug(f"Flushed data to {self._log_file}")
        except OSError as exc:
            _log.error(f"Failed to write log file: {exc}")

    # ── internal ──────────────────────────────────────────────────────────

    def _maybe_flush(self) -> None:
        self._write_count += 1
        if self._write_count >= FLUSH_EVERY_N:
            self._write_count = 0
            self.flush()

    def _load(self) -> None:
        """โหลดข้อมูลจากไฟล์ (ถ้ามี) เข้า in-memory queue"""
        if not self._log_file.exists():
            return
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data.get("positions", []):
                self._positions.append(PositionRecord(**{
                    k: item[k] for k in ("position", "duration_seconds", "timestamp")
                    if k in item
                }))

            for item in data.get("alarms", []):
                self._alarms.append(AlarmRecord(**{
                    k: item[k] for k in ("alarm_type", "details", "timestamp")
                    if k in item
                }))

            for item in data.get("meals", []):
                self._meals.append(MealRecord(**{
                    k: item[k] for k in ("meal_name", "timestamp")
                    if k in item
                }))

            _log.info(
                f"Loaded {len(self._positions)} positions, "
                f"{len(self._alarms)} alarms, "
                f"{len(self._meals)} meals from {self._log_file}"
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            _log.warning(f"Could not load log file ({exc}), starting fresh")
