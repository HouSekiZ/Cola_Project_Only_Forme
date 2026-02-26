"""
HandRenderer: HUD overlay for hand gesture detection mode
"""
import cv2
import numpy as np
from typing import Dict, Any

from .base import BaseRenderer


class HandRenderer(BaseRenderer):
    """วาด HUD สำหรับโหมดมือ"""

    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_ALARM = (0, 0, 220)

    def render(self, frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        output = frame.copy()

        alarm = detection.get('alarm', False)

        # Flash red if alarm
        if alarm:
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (output.shape[1], output.shape[0]),
                          self.COLOR_ALARM, -1)
            cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

        if not detection.get('detected'):
            cv2.putText(output, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_RED, 2)
            return output

        data = detection.get('data', {})
        state = data.get('state', 'UNKNOWN')
        last_state = data.get('last_state', '')
        handedness = data.get('handedness', '')

        state_color = {
            'OPEN': self.COLOR_GREEN,
            'FIST': self.COLOR_RED,
            'UNKNOWN': self.COLOR_YELLOW
        }.get(state, self.COLOR_WHITE)

        lines = [
            (f"Hand: {handedness}", self.COLOR_WHITE),
            (f"State: {state}", state_color),
            (f"Prev: {last_state}", self.COLOR_WHITE),
        ]

        if alarm:
            lines.insert(0, ("!! HELP SIGNAL !!", self.COLOR_ALARM))

        y = 30
        for text, color in lines:
            cv2.putText(output, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)
            y += 30

        return output
