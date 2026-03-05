"""
Database Repository Layer
Data access functions สำหรับแต่ละ domain

ทุก function รับ session จากภายนอก (dependency injection)
เพื่อให้ test ได้ง่ายและไม่ผูกกับ session lifecycle

การใช้งาน:
    from database import repo
    from database.db import get_db

    with get_db() as db:
        repo.save_alarm(db, alarm_type="eye_blink")
        record = repo.start_position(db, position="SUPINE")
        repo.end_position(db, record_id=record.id)
"""
import json
from datetime import datetime, date
from typing import Optional, List

from sqlalchemy.orm import Session

from .models import AlarmEventDB, PositionRecordDB, MealRecordDB, NotificationDB
from utils.logger import setup_logger

logger = setup_logger("database.repo")


# ── Alarm Events ─────────────────────────────────────────────────────────────

def save_alarm(db: Session, alarm_type: str, metadata: dict = None) -> AlarmEventDB:
    """บันทึก alarm event ใหม่"""
    event = AlarmEventDB(
        alarm_type=alarm_type,
        triggered_at=datetime.now(),
        metadata_json=json.dumps(metadata or {}),
    )
    db.add(event)
    db.flush()
    logger.debug(f"Saved alarm: {alarm_type} (id={event.id})")
    return event


def acknowledge_alarm_db(db: Session, alarm_type: str) -> bool:
    """ทำเครื่องหมาย alarm ที่ยัง acknowledge ล่าสุดว่า acknowledged"""
    event = (
        db.query(AlarmEventDB)
        .filter_by(alarm_type=alarm_type, acknowledged=False)
        .order_by(AlarmEventDB.triggered_at.desc())
        .first()
    )
    if event:
        event.acknowledged    = True
        event.acknowledged_at = datetime.now()
        db.flush()
        return True
    return False


def get_alarm_history(db: Session, limit: int = 50) -> List[dict]:
    """ดึงประวัติ alarm ล่าสุด"""
    events = (
        db.query(AlarmEventDB)
        .order_by(AlarmEventDB.triggered_at.desc())
        .limit(limit)
        .all()
    )
    return [e.to_dict() for e in events]


# ── Position Records ─────────────────────────────────────────────────────────

def start_position(db: Session, position: str, started_at: datetime = None) -> PositionRecordDB:
    """เริ่ม record ท่านอนใหม่"""
    record = PositionRecordDB(
        position=position,
        started_at=started_at or datetime.now(),
    )
    db.add(record)
    db.flush()
    logger.debug(f"Position started: {position} (id={record.id})")
    return record


def end_position(db: Session, record_id: int, ended_at: datetime = None) -> bool:
    """ปิด record ท่านอน และคำนวณ duration"""
    record = db.get(PositionRecordDB, record_id)
    if not record:
        return False
    end = ended_at or datetime.now()
    record.ended_at     = end
    record.duration_sec = (end - record.started_at).total_seconds()
    db.flush()
    logger.debug(f"Position ended: {record.position} ({record.duration_sec:.0f}s)")
    return True


def get_position_history(db: Session, limit: int = 50,
                         date_from: datetime = None) -> List[dict]:
    """ดึงประวัติท่านอน"""
    q = db.query(PositionRecordDB).order_by(PositionRecordDB.started_at.desc())
    if date_from:
        q = q.filter(PositionRecordDB.started_at >= date_from)
    return [r.to_dict() for r in q.limit(limit).all()]


def get_position_stats_today(db: Session) -> dict:
    """สรุปท่านอนวันนี้ (แยกตาม position)"""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    records = (
        db.query(PositionRecordDB)
        .filter(PositionRecordDB.started_at >= today_start)
        .filter(PositionRecordDB.duration_sec.isnot(None))
        .all()
    )
    stats = {}
    for r in records:
        stats.setdefault(r.position, 0.0)
        stats[r.position] += r.duration_sec
    return {k: round(v, 1) for k, v in stats.items()}


# ── Meal Records ─────────────────────────────────────────────────────────────

def upsert_meal(db: Session, meal_name: str, scheduled_time: str,
                record_date: date = None) -> MealRecordDB:
    """สร้างหรือดึง meal record วันนี้"""
    today = record_date or datetime.now().date()
    record = (
        db.query(MealRecordDB)
        .filter_by(record_date=today, meal=meal_name)
        .first()
    )
    if not record:
        record = MealRecordDB(
            record_date=today,
            meal=meal_name,
            scheduled_time=scheduled_time,
        )
        db.add(record)
        db.flush()
    return record


def mark_meal_eaten_db(db: Session, meal_name: str) -> bool:
    """บันทึก meal ว่าทานแล้ว"""
    today = datetime.now().date()
    record = (
        db.query(MealRecordDB)
        .filter_by(record_date=today, meal=meal_name)
        .first()
    )
    if not record:
        return False
    record.eaten    = True
    record.eaten_at = datetime.now()
    db.flush()
    logger.info(f"Meal eaten recorded: {meal_name}")
    return True


def get_meal_history(db: Session, days: int = 7) -> List[dict]:
    """ดึงประวัติอาหาร N วันล่าสุด"""
    records = (
        db.query(MealRecordDB)
        .order_by(MealRecordDB.record_date.desc(), MealRecordDB.meal)
        .limit(days * 3)  # 3 มื้อต่อวัน
        .all()
    )
    return [r.to_dict() for r in records]


# ── Notifications ─────────────────────────────────────────────────────────────

def save_notification(db: Session, notif_type: str,
                      title: str, message: str = None) -> NotificationDB:
    """บันทึก notification"""
    notif = NotificationDB(
        notif_type=notif_type,
        title=title,
        message=message,
        created_at=datetime.now(),
    )
    db.add(notif)
    db.flush()
    return notif


def get_notification_history(db: Session, limit: int = 30) -> List[dict]:
    """ดึงประวัติ notification"""
    records = (
        db.query(NotificationDB)
        .order_by(NotificationDB.created_at.desc())
        .limit(limit)
        .all()
    )
    return [r.to_dict() for r in records]
