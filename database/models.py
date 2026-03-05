"""
SQLAlchemy ORM Models สำหรับ Patient Assist System

Tables:
    alarm_events      — ประวัติ alarm ทุกประเภท
    position_records  — บันทึกท่านอน (เริ่ม/จบ/ระยะเวลา)
    meal_records      — บันทึกมื้ออาหารรายวัน
    notifications     — ประวัติการแจ้งเตือน
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Date, Time, Text, Index
)
from .db import Base


# ── Alarm Events ─────────────────────────────────────────────────────────────
class AlarmEventDB(Base):
    """บันทึก alarm ทุกครั้งที่เกิดขึ้น"""
    __tablename__ = "alarm_events"

    id              = Column(Integer,  primary_key=True, autoincrement=True)
    alarm_type      = Column(String(50),  nullable=False, index=True)   # eye_blink / hand_gesture / body_position
    triggered_at    = Column(DateTime,    nullable=False, default=datetime.now)
    acknowledged    = Column(Boolean,     nullable=False, default=False)
    acknowledged_at = Column(DateTime,    nullable=True)
    metadata_json   = Column(Text,        nullable=True)   # JSON extras

    __table_args__ = (
        Index("ix_alarm_triggered", "triggered_at"),
    )

    def __repr__(self):
        return f"<AlarmEvent {self.alarm_type} @ {self.triggered_at}>"

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "alarm_type":     self.alarm_type,
            "triggered_at":   self.triggered_at.isoformat() if self.triggered_at else None,
            "acknowledged":   self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


# ── Position Records ─────────────────────────────────────────────────────────
class PositionRecordDB(Base):
    """บันทึกช่วงเวลาที่ผู้ป่วยอยู่ในท่าหนึ่ง"""
    __tablename__ = "position_records"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    position     = Column(String(20),  nullable=False, index=True)   # SUPINE / LEFT_SIDE / RIGHT_SIDE
    started_at   = Column(DateTime,    nullable=False, index=True)
    ended_at     = Column(DateTime,    nullable=True)
    duration_sec = Column(Float,       nullable=True)                 # วินาทีที่อยู่ท่านี้

    __table_args__ = (
        Index("ix_position_started", "started_at"),
    )

    def __repr__(self):
        return f"<PositionRecord {self.position} {self.duration_sec:.0f}s>"

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "position":     self.position,
            "started_at":   self.started_at.isoformat() if self.started_at else None,
            "ended_at":     self.ended_at.isoformat() if self.ended_at else None,
            "duration_sec": round(self.duration_sec, 1) if self.duration_sec else None,
        }


# ── Meal Records ─────────────────────────────────────────────────────────────
class MealRecordDB(Base):
    """บันทึกสถานะอาหารรายวัน"""
    __tablename__ = "meal_records"

    id             = Column(Integer,    primary_key=True, autoincrement=True)
    record_date    = Column(Date,       nullable=False, index=True)
    meal           = Column(String(20), nullable=False)                # breakfast / lunch / dinner
    scheduled_time = Column(String(10), nullable=False)                # "HH:MM"
    eaten          = Column(Boolean,    nullable=False, default=False)
    eaten_at       = Column(DateTime,   nullable=True)

    __table_args__ = (
        Index("ix_meal_date_meal", "record_date", "meal"),
    )

    def __repr__(self):
        return f"<MealRecord {self.record_date} {self.meal} eaten={self.eaten}>"

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "record_date":    self.record_date.isoformat() if self.record_date else None,
            "meal":           self.meal,
            "scheduled_time": self.scheduled_time,
            "eaten":          self.eaten,
            "eaten_at":       self.eaten_at.isoformat() if self.eaten_at else None,
        }


# ── Notifications ─────────────────────────────────────────────────────────────
class NotificationDB(Base):
    """ประวัติการแจ้งเตือนทุกประเภท"""
    __tablename__ = "notifications"

    id           = Column(Integer,    primary_key=True, autoincrement=True)
    notif_type   = Column(String(30), nullable=False, index=True)   # reposition / meal
    title        = Column(String(100), nullable=False)
    message      = Column(Text,        nullable=True)
    created_at   = Column(DateTime,    nullable=False, default=datetime.now, index=True)
    acknowledged = Column(Boolean,     nullable=False, default=False)

    def __repr__(self):
        return f"<Notification {self.notif_type} @ {self.created_at}>"

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "type":         self.notif_type,
            "title":        self.title,
            "message":      self.message,
            "created_at":   self.created_at.isoformat() if self.created_at else None,
            "acknowledged": self.acknowledged,
        }
