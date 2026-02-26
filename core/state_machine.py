"""
BlinkStateMachine: extracted blink detection logic (FAR SOS + NEAR modes)
"""
import time
from typing import Dict

from config import SOS_CONSTANTS, NEAR_CONSTANTS, EYE_CONSTANTS
from utils.logger import setup_logger

logger = setup_logger('state_machine')


class BlinkStateMachine:
    """
    State machine สำหรับตรวจจับการกระพิบตา

    FAR mode: SOS Pattern (2 blinks → pause → 3 blinks)
    NEAR mode: 3 blinks within 8 seconds
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all state"""
        # Blink tracking
        self.eye_closed = False
        self.close_start: float = 0.0
        self.last_blink_end: float = 0.0

        # FAR / SOS state
        self.sos_phase = "IDLE"   # IDLE, PHASE1, PAUSE, PHASE2, DONE
        self.p1_count = 0
        self.p2_count = 0
        self.p1_start: float = 0.0
        self.pause_start: float = 0.0

        # NEAR state
        self.near_blinks: list = []

        logger.debug("BlinkStateMachine reset")

    def update(self, ear: float, now: float, mode: str) -> Dict[str, bool]:
        """
        Update state machine with current EAR value.

        Args:
            ear: Eye Aspect Ratio
            now: current timestamp
            mode: 'FAR' or 'NEAR'

        Returns:
            dict with keys: 'blink' (bool), 'alarm' (bool), 'sos_done' (bool)
        """
        events = {'blink': False, 'alarm': False, 'sos_done': False}

        threshold = EYE_CONSTANTS['EAR_THRESHOLD']
        min_dur = EYE_CONSTANTS['MIN_BLINK_DUR']
        max_dur = EYE_CONSTANTS['MAX_BLINK_DUR']

        # Detect blink open/close transitions
        if ear < threshold:
            if not self.eye_closed:
                self.eye_closed = True
                self.close_start = now
        else:
            if self.eye_closed:
                self.eye_closed = False
                duration = now - self.close_start

                if min_dur <= duration <= max_dur:
                    events['blink'] = True
                    self.last_blink_end = now

                    if mode == 'FAR':
                        self._handle_sos_blink(now, events)
                    elif mode == 'NEAR':
                        self._handle_near_blink(now, events)

        # Check SOS timeouts
        if mode == 'FAR':
            self._check_sos_timeouts(now)

        return events

    def _handle_sos_blink(self, now: float, events: Dict):
        """Handle blink in SOS (FAR) mode"""
        sos = SOS_CONSTANTS

        if self.sos_phase == "IDLE":
            self.sos_phase = "PHASE1"
            self.p1_count = 1
            self.p1_start = now
            logger.debug(f"SOS PHASE1 started, count=1")

        elif self.sos_phase == "PHASE1":
            self.p1_count += 1
            logger.debug(f"SOS PHASE1 blink, count={self.p1_count}")

            if self.p1_count >= sos['PHASE1']:
                self.sos_phase = "PAUSE"
                self.pause_start = now
                logger.debug("SOS PAUSE started")

        elif self.sos_phase == "PAUSE":
            pause_dur = now - self.pause_start
            if sos['PAUSE_MIN'] <= pause_dur <= sos['PAUSE_MAX']:
                self.sos_phase = "PHASE2"
                self.p2_count = 1
                logger.debug(f"SOS PHASE2 started, count=1")
            else:
                # Too soon or too late → reset
                self.reset()

        elif self.sos_phase == "PHASE2":
            self.p2_count += 1
            logger.debug(f"SOS PHASE2 blink, count={self.p2_count}")

            if self.p2_count >= sos['PHASE2']:
                self.sos_phase = "DONE"
                events['sos_done'] = True
                events['alarm'] = True
                logger.info("SOS pattern completed! ALARM triggered.")
                self.reset()

    def _handle_near_blink(self, now: float, events: Dict):
        """Handle blink in NEAR mode"""
        near = NEAR_CONSTANTS
        window = near['BLINK_WIN']
        required = near['BLINK_N']

        # Keep only blinks within window
        self.near_blinks = [t for t in self.near_blinks if now - t <= window]
        self.near_blinks.append(now)

        logger.debug(f"NEAR blinks in window: {len(self.near_blinks)}")

        if len(self.near_blinks) >= required:
            events['alarm'] = True
            logger.info("NEAR blink pattern completed! ALARM triggered.")
            self.near_blinks.clear()

    def _check_sos_timeouts(self, now: float):
        """Reset SOS state if phase windows expire"""
        sos = SOS_CONSTANTS

        if self.sos_phase == "PHASE1":
            if now - self.p1_start > sos['PHASE_WIN']:
                logger.debug("SOS PHASE1 timeout → reset")
                self.reset()

        elif self.sos_phase == "PAUSE":
            if now - self.pause_start > sos['PAUSE_MAX']:
                logger.debug("SOS PAUSE timeout → reset")
                self.reset()

        elif self.sos_phase == "PHASE2":
            # Phase2 window starts from pause_start + phase1
            if now - self.pause_start > sos['PHASE_WIN'] + sos['PAUSE_MAX']:
                logger.debug("SOS PHASE2 timeout → reset")
                self.reset()