import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from core.state_machine import BlinkStateMachine


class TestBlinkStateMachine(unittest.TestCase):

    def setUp(self):
        self.sm = BlinkStateMachine()

    def test_initial_state(self):
        self.assertEqual(self.sm.sos_phase, "IDLE")
        self.assertEqual(self.sm.p1_count, 0)
        self.assertFalse(self.sm.eye_closed)
        self.assertEqual(len(self.sm.near_blinks), 0)

    def test_single_blink_far(self):
        """A single blink in FAR mode should not trigger alarm"""
        # Eye closes
        events = self.sm.update(ear=0.10, now=1.0, mode="FAR")
        self.assertFalse(events['blink'])

        # Eye opens after valid duration
        events = self.sm.update(ear=0.30, now=1.2, mode="FAR")
        self.assertTrue(events['blink'])
        self.assertFalse(events['alarm'])

    def test_sos_pattern_far(self):
        """Full SOS pattern (2 blinks + pause + 3 blinks) should trigger alarm"""
        t = 0.0

        def blink(start, duration=0.15):
            nonlocal t
            t = start
            self.sm.update(ear=0.10, now=t, mode="FAR")
            t = start + duration
            return self.sm.update(ear=0.30, now=t, mode="FAR")

        # Phase 1: 2 blinks
        blink(1.0)
        blink(1.4)

        # Pause (0.6 seconds)
        self.sm.update(ear=0.30, now=2.2, mode="FAR")

        # Phase 2: 3 blinks
        blink(2.2)
        blink(2.6)
        events = blink(3.0)

        self.assertTrue(events['alarm'] or events['sos_done'])

    def test_near_mode_3_blinks(self):
        """3 blinks within window in NEAR mode should trigger alarm"""
        t = 0.0

        def blink(start):
            self.sm.update(ear=0.10, now=start, mode="NEAR")
            return self.sm.update(ear=0.30, now=start + 0.15, mode="NEAR")

        blink(1.0)
        blink(1.5)
        events = blink(2.0)

        self.assertTrue(events['alarm'])

    def test_near_mode_timeout_resets(self):
        """Blinks outside the 8s window should not accumulate"""
        def blink(start):
            self.sm.update(ear=0.10, now=start, mode="NEAR")
            return self.sm.update(ear=0.30, now=start + 0.15, mode="NEAR")

        blink(1.0)
        blink(1.5)
        # 3rd blink happens way outside the window
        events = blink(20.0)  # 18 seconds later

        self.assertFalse(events['alarm'])

    def test_reset(self):
        """Reset should clear all state"""
        self.sm.update(ear=0.10, now=1.0, mode="FAR")
        self.sm.update(ear=0.30, now=1.2, mode="FAR")
        self.sm.reset()

        self.assertEqual(self.sm.sos_phase, "IDLE")
        self.assertEqual(self.sm.p1_count, 0)
        self.assertEqual(len(self.sm.near_blinks), 0)
        self.assertFalse(self.sm.eye_closed)


if __name__ == '__main__':
    unittest.main()
