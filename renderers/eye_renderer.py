"""
EyeRenderer: HUD overlay for eye blink detection mode
"""
import cv2
import numpy as np
from typing import Dict, Any

from .base import BaseRenderer
from config import EYE_CONSTANTS


class EyeRenderer(BaseRenderer):
    """วาด HUD สำหรับโหมดดวงตา"""

    # Colors (BGR)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_ALARM = (0, 0, 220)

    def render(self, frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        output = frame.copy()

        if not detection.get('detected'):
            self._draw_no_face(output)
            return output

        data = detection.get('data', {})
        alarm = detection.get('alarm', False)

        # Flash red if alarm
        if alarm:
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (output.shape[1], output.shape[0]),
                          self.COLOR_ALARM, -1)
            cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

        # Draw eye landmarks
        for pt in data.get('eye_pts_left', []):
            cv2.circle(output, pt, 2, self.COLOR_GREEN, -1)
        for pt in data.get('eye_pts_right', []):
            cv2.circle(output, pt, 2, self.COLOR_GREEN, -1)

        # HUD panel
        ear = data.get('ear', 0.0)
        blink_mode = data.get('blink_mode', 'FAR')
        sos_phase = data.get('sos_phase', 'IDLE')
        p1_count = data.get('p1_count', 0)
        near_blinks = data.get('near_blinks', 0)
        gaze = data.get('gaze', {})

        threshold = EYE_CONSTANTS['EAR_THRESHOLD']
        ear_color = self.COLOR_RED if ear < threshold else self.COLOR_GREEN

        lines = [
            (f"Mode: {blink_mode}", self.COLOR_WHITE),
            (f"EAR: {ear:.3f}", ear_color),
            (f"Gaze OK: {gaze.get('ok', False)}", self.COLOR_WHITE),
        ]

        if blink_mode == "FAR":
            lines.append((f"SOS Phase: {sos_phase}  P1: {p1_count}", self.COLOR_YELLOW))
        elif blink_mode == "NEAR":
            lines.append((f"Blinks: {near_blinks}/3", self.COLOR_YELLOW))

        if alarm:
            lines.insert(0, ("!! ALARM !!", self.COLOR_ALARM))

        y = 30
        for text, color in lines:
            cv2.putText(output, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)
            y += 30

        return output

    def _draw_no_face(self, frame: np.ndarray):
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_RED, 2)
