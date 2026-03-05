"""
Database Connection Manager
MySQL + SQLAlchemy สำหรับ Patient Assist System

ตัวแปร environment ที่ใช้:
    DB_HOST      (default: localhost)
    DB_PORT      (default: 3306)
    DB_NAME      (default: patient_assist)
    DB_USER      (default: root)
    DB_PASSWORD  (default: "")
"""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from utils.logger import setup_logger

logger = setup_logger("database")


# ── Base class สำหรับ ORM models ────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Connection URL ───────────────────────────────────────────────────────────
def _build_url() -> str:
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "3306")
    name     = os.getenv("DB_NAME",     "patient_assist")
    user     = os.getenv("DB_USER",     "root")
    password = os.getenv("DB_PASSWORD", "Kay254898!")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


# ── Engine (lazy init) ───────────────────────────────────────────────────────
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        url = _build_url()
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,          # recycle connection ทุก 1 ชั่วโมง
            pool_pre_ping=True,         # ตรวจ connection ก่อนใช้
            echo=False,
        )
        logger.info(f"Database engine created: {url.split('@')[1]}")
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


@contextmanager
def get_db():
    """Context manager สำหรับ database session"""
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error (rolled back): {e}")
        raise
    finally:
        session.close()


def init_db():
    """สร้าง tables ถ้ายังไม่มี (safe to call multiple times)"""
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        # ทดสอบ connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database tables initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def is_db_available() -> bool:
    """ตรวจสอบว่า DB เชื่อมต่อได้หรือไม่"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
