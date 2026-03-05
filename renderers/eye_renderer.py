"""
EyeRenderer: HUD overlay for eye blink detection mode
"""
import cv2
import numpy as np
from typing import Dict, Any

from .base import BaseRenderer
from .thai_text import draw_hud_box
from config import EYE_CONSTANTS


class EyeRenderer(BaseRenderer):
    """วาด HUD สำหรับโหมดดวงตา"""

    # Colors (BGR)
    COLOR_GREEN  = (80,  200, 80)
    COLOR_RED    = (60,  60,  220)
    COLOR_YELLOW = (0,   220, 220)
    COLOR_WHITE  = (240, 240, 240)
    COLOR_ALARM  = (0,   0,   220)
    COLOR_GRAY   = (160, 160, 160)
    COLOR_CYAN   = (220, 200, 0)

    # แปล sos_phase เป็นภาษาไทย
    _SOS_PHASE_TH = {
        "IDLE":     "รอสัญญาณ",
        "P1":       "กะพริบ 1 ครั้ง",
        "WAIT_P2":  "รอกะพริบอีกครั้ง",
        "P2":       "กะพริบ 2 ครั้ง",
        "CONFIRM":  "กำลังยืนยัน",
        "SOS":      "ขอความช่วยเหลือ",
    }

    def render(self, frame: np.ndarray, detection: Dict[str, Any], is_all_mode: bool = False) -> np.ndarray:
        output = frame.copy()

        if not detection.get('detected'):
            draw_hud_box(output, [("ไม่พบใบหน้า", self.COLOR_GRAY)], top=4)
            return output

        data  = detection.get('data', {})
        alarm = detection.get('alarm', False)

        # Flash red overlay if alarm
        if alarm:
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (output.shape[1], output.shape[0]),
                          self.COLOR_ALARM, -1)
            cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

        # Draw eye landmarks
        for pt in data.get('eye_pts_left', []):
            cv2.circle(output, pt, 2, self.COLOR_GREEN, -1)
        for pt in data.get('eye_pts_right', []):
            cv2.circle(output, pt, 2, self.COLOR_GREEN, -1)

        # ── ดึงค่าข้อมูลทั้งหมด ──────────────────────────────────────────
        ear        = data.get('ear', 0.0)
        blink_mode = data.get('blink_mode', 'FAR')
        sos_phase  = data.get('sos_phase', 'IDLE')
        p1_count   = data.get('p1_count', 0)
        near_blinks= data.get('near_blinks', 0)
        gaze       = data.get('gaze', {})
        gaze_ok    = gaze.get('ok', False)

        threshold  = EYE_CONSTANTS['EAR_THRESHOLD']
        ear_color  = self.COLOR_RED   if ear < threshold else self.COLOR_GREEN
        gaze_text  = "มองหน้ากล้อง" if gaze_ok else "ไม่มองกล้อง"
        gaze_color = self.COLOR_GREEN if gaze_ok else self.COLOR_YELLOW
        mode_label = "ไกล" if blink_mode == "FAR" else "ใกล้"
        sos_th     = self._SOS_PHASE_TH.get(sos_phase, sos_phase)

        # ── สร้างรายการบรรทัด ─────────────────────────────────────────────
        lines = []

        if alarm:
            lines.append(("!! แจ้งเตือน SOS !!", self.COLOR_ALARM))

        if not is_all_mode:
            lines += [
                (f"โหมด    : {mode_label}",              self.COLOR_WHITE),
                (f"EAR     : {ear:.3f}",                  ear_color),
                (f"สายตา   : {gaze_text}",               gaze_color),
            ]

        if blink_mode == "FAR":
            lines.append((f"SOS ขั้นตอน : {sos_th}", self.COLOR_YELLOW))
            if not is_all_mode:
                lines.append((f"กะพริบรอบ 1 : {p1_count} ครั้ง", self.COLOR_CYAN))
        else:  # NEAR
            lines.append((f"กะพริบ : {near_blinks}/3 ครั้ง", self.COLOR_YELLOW))

        border_color = self.COLOR_ALARM if alarm else self.COLOR_CYAN
        draw_hud_box(output, lines, top=4, border_color=border_color)

        return output
