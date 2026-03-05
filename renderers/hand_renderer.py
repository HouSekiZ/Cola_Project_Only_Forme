"""
HandRenderer: HUD overlay for hand gesture detection mode
"""
import cv2
import numpy as np
from typing import Dict, Any

from .base import BaseRenderer
from .thai_text import draw_hud_box


class HandRenderer(BaseRenderer):
    """วาด HUD สำหรับโหมดมือ"""

    COLOR_GREEN  = (80,  200, 80)
    COLOR_RED    = (60,  60,  220)
    COLOR_YELLOW = (0,   220, 220)
    COLOR_WHITE  = (240, 240, 240)
    COLOR_ALARM  = (0,   0,   220)
    COLOR_GRAY   = (160, 160, 160)
    COLOR_CYAN   = (220, 200, 0)

    # แปลสถานะมือเป็นภาษาไทย
    _STATE_TH = {
        'OPEN':    ('มือเปิด',  (80,  200, 80)),
        'FIST':    ('กำมือ',    (60,  60,  220)),
        'UNKNOWN': ('ไม่ทราบ',  (0,   220, 220)),
    }

    # แปลมือซ้าย/ขวา
    _HAND_TH = {
        'Left':  'มือซ้าย',
        'Right': 'มือขวา',
        '':      '-',
    }

    def render(self, frame: np.ndarray, detection: Dict[str, Any], is_all_mode: bool = False) -> np.ndarray:
        output = frame.copy()

        alarm = detection.get('alarm', False)

        # Flash red overlay if alarm
        if alarm:
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (output.shape[1], output.shape[0]),
                          self.COLOR_ALARM, -1)
            cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

        if not detection.get('detected'):
            draw_hud_box(output, [("ไม่พบมือ", self.COLOR_GRAY)], top=180)
            return output

        data       = detection.get('data', {})
        state      = data.get('state', 'UNKNOWN')
        last_state = data.get('last_state', '')
        handedness = data.get('handedness', '')

        state_text, state_color = self._STATE_TH.get(state,      (state,      self.COLOR_WHITE))
        prev_text,  _           = self._STATE_TH.get(last_state, (last_state, self.COLOR_GRAY))
        hand_th = self._HAND_TH.get(handedness, handedness)

        # ── สร้างรายการบรรทัด ─────────────────────────────────────────────
        lines = []
        if alarm:
            lines.append(("!! สัญญาณขอความช่วยเหลือ !!", self.COLOR_ALARM))

        lines += [
            (f"มือ      : {hand_th}",   self.COLOR_WHITE),
            (f"ท่าทาง  : {state_text}", state_color),
        ]
        if not is_all_mode:
            lines.append((f"ก่อนหน้า : {prev_text}", self.COLOR_GRAY))

        border_color = self.COLOR_ALARM if alarm else self.COLOR_CYAN
        draw_hud_box(output, lines, top=180, border_color=border_color)

        return output
