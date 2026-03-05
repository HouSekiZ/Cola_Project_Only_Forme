"""
BodyRenderer: HUD overlay for body position detection mode
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional

from .base import BaseRenderer
from .thai_text import draw_hud_box
from detectors.body_detector import BodyDetector, BodyResult


# ── color palette (BGR) ────────────────────────────────────────────────────
_C_WHITE   = (255, 255, 255)
_C_GREEN   = (80,  200, 80)
_C_YELLOW  = (0,   220, 220)
_C_ORANGE  = (0,   160, 255)
_C_RED     = (60,  60,  220)
_C_GRAY    = (160, 160, 160)
_C_ALARM   = (0,   0,   220)

# ท่า → ไอคอน + ชื่อภาษาไทย
_POSITION_INFO: Dict[str, Dict] = {
    "SUPINE":     {"icon": "🛏️",  "label_th": "หงาย",        "color": _C_GREEN},
    "LEFT_SIDE":  {"icon": "◀️",  "label_th": "ตะแคงซ้าย",   "color": _C_YELLOW},
    "RIGHT_SIDE": {"icon": "▶️",  "label_th": "ตะแคงขวา",    "color": _C_YELLOW},
    "UNKNOWN":    {"icon": "❓",  "label_th": "ไม่ทราบ",       "color": _C_GRAY},
}


class BodyRenderer(BaseRenderer):
    """
    วาด HUD สำหรับโหมดตรวจจับท่านอน

    detection dict ที่คาดว่าจะได้รับ (จาก Detector.process_frame):
    {
        "detected": bool,
        "alarm":    bool,          # flip-reminder alarm
        "data": {
            "position":              str,   # "SUPINE" | "LEFT_SIDE" | "RIGHT_SIDE" | "UNKNOWN"
            "confidence":            float,
            "current_duration":      float, # วินาทีที่อยู่ท่าปัจจุบัน
            "time_until_reposition": float, # วินาทีที่เหลือ
            "reposition_due":        bool,
            "raw_landmarks":         object | None,
        }
    }
    """

    def render(self, frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        output = frame.copy()

        alarm = detection.get("alarm", False)

        # ── แฟลชแดงเมื่อถึงเวลาพลิกตัว ──
        if alarm:
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (output.shape[1], output.shape[0]),
                          _C_ALARM, -1)
            cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)

        if not detection.get("detected"):
            _draw_hud_box(output, [("ไม่พบร่างกาย", _C_GRAY)], top=4)
            return output

        data = detection.get("data", {})
        position   = data.get("position", "UNKNOWN")
        confidence = data.get("confidence", 0.0)
        duration   = data.get("current_duration", 0.0)
        remaining  = data.get("time_until_reposition", 0.0)
        due        = data.get("reposition_due", False)
        raw_lms    = data.get("raw_landmarks")

        # ── วาด skeleton (ถ้ามี landmarks) ──
        if raw_lms is not None:
            BodyDetector.draw_landmarks(output, raw_lms, position)

        # ── HUD text ──
        info      = _POSITION_INFO.get(position, _POSITION_INFO["UNKNOWN"])
        pos_color = info["color"]
        label_th  = info["label_th"]

        lines = []
        if alarm:
            lines.append(("🚨  พลิกตัวผู้ป่วยด่วน!", _C_ALARM))

        lines += [
            (f"ท่านอน       : {label_th}",                   pos_color),
            (f"ความมั่นใจ  : {confidence:.0%}",               _C_WHITE),
            (f"อยู่ท่านี้นาน: {_fmt_duration(duration)}",      _C_WHITE),
        ]

        if due:
            lines.append(("⚠️  ถึงเวลาพลิกตัวแล้ว!", _C_ORANGE))
        else:
            lines.append((f"พลิกตัวอีก  : {_fmt_duration(remaining)}", _C_YELLOW))

        border_color = _C_ALARM if alarm else (0, 180, 180)
        _draw_hud_box(output, lines, top=4, border_color=border_color)

        # ── progress bar เวลาก่อนพลิกตัว ──
        self._draw_progress_bar(output, remaining, due)

        return output

    # ── internal helpers ───────────────────────────────────────────────────

    def _draw_progress_bar(
        self,
        frame: np.ndarray,
        remaining: float,
        due: bool,
        interval: float = 7200.0,
    ) -> None:
        """วาด progress bar แนวนอนที่ด้านล่างของ frame"""
        h, w = frame.shape[:2]
        bar_h  = 8
        bar_y  = h - bar_h - 4
        bar_x1 = 10
        bar_x2 = w - 10
        bar_w  = bar_x2 - bar_x1

        elapsed = max(0.0, interval - remaining)
        pct     = min(1.0, elapsed / interval)
        filled  = int(bar_w * pct)

        # background track
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h),
                      (60, 60, 60), -1)

        # filled portion
        if filled > 0:
            color = _C_ALARM if due else (_C_ORANGE if pct > 0.85 else _C_GREEN)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + filled, bar_y + bar_h),
                          color, -1)


# ── module helper ──────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """แปลงวินาที → 'X ชม. Y นาที' หรือ 'Y นาที'"""
    if seconds <= 0:
        return "0 นาที"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h} ชม. {m} นาที"
    return f"{m} นาที"



