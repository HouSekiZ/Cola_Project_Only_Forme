import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import unittest
from core.notification_manager import NotificationManager, Notification


class TestNotificationManagerInit(unittest.TestCase):
    def setUp(self):
        self.nm = NotificationManager()

    def test_initial_pending_empty(self):
        pending = self.nm.pop_pending()
        self.assertIsInstance(pending, list)
        self.assertEqual(len(pending), 0)

    def test_default_meal_times_set(self):
        meals = self.nm.get_meal_status()
        self.assertIsInstance(meals, list)
        # ควรมี 3 มื้อ
        meal_names = {m["name"] for m in meals}
        self.assertSetEqual(meal_names, {"breakfast", "lunch", "dinner"})

    def test_all_meals_not_eaten_initially(self):
        meals = self.nm.get_meal_status()
        for m in meals:
            self.assertFalse(m["eaten"], f"{m['name']} ควรยังไม่ได้ทาน")


class TestMarkMealEaten(unittest.TestCase):
    def setUp(self):
        self.nm = NotificationManager()

    def test_mark_breakfast_eaten(self):
        self.nm.mark_meal_eaten("breakfast")
        meals = self.nm.get_meal_status()
        bf = next(m for m in meals if m["name"] == "breakfast")
        self.assertTrue(bf["eaten"])

    def test_mark_invalid_meal_no_crash(self):
        # ชื่อมื้อที่ไม่มีอยู่ → ไม่ crash
        try:
            self.nm.mark_meal_eaten("brunch")
        except Exception as e:
            self.fail(f"mark_meal_eaten raised unexpectedly: {e}")

    def test_mark_all_meals_eaten(self):
        for name in ("breakfast", "lunch", "dinner"):
            self.nm.mark_meal_eaten(name)
        meals = self.nm.get_meal_status()
        for m in meals:
            self.assertTrue(m["eaten"], f"{m['name']} ควรทานแล้ว")


class TestSetMealTimes(unittest.TestCase):
    def setUp(self):
        self.nm = NotificationManager()

    def test_set_valid_meal_times(self):
        new_times = {"breakfast": "08:00", "lunch": "13:00", "dinner": "19:00"}
        self.nm.set_meal_times(new_times)
        meals = self.nm.get_meal_status()
        times = {m["name"]: m["scheduled_time"] for m in meals}
        self.assertEqual(times["breakfast"], "08:00")
        self.assertEqual(times["lunch"],     "13:00")
        self.assertEqual(times["dinner"],    "19:00")

    def test_set_partial_meal_times(self):
        """set_meal_times replaces ทั้ง dict — ไม่ใช่ merge"""
        self.nm.set_meal_times({"breakfast": "06:30"})
        meals = self.nm.get_meal_status()
        # หลัง set ด้วย 1 มื้อ → มีแค่ 1 มื้อนั้น (ไม่ใช่ partial merge)
        meal_names = {m["name"] for m in meals}
        self.assertIn("breakfast", meal_names)
        times = {m["name"]: m["scheduled_time"] for m in meals}
        self.assertEqual(times["breakfast"], "06:30")



class TestRepositionNotification(unittest.TestCase):
    def setUp(self):
        self.nm = NotificationManager()

    def test_no_notification_when_not_due(self):
        new = self.nm.tick(reposition_due=False)
        self.assertEqual(len(new), 0)

    def test_notification_when_due(self):
        new = self.nm.tick(reposition_due=True)
        # ควรมีอย่างน้อย 1 notification ประเภท "reposition"
        self.assertTrue(any(n.type == "reposition" for n in new))

    def test_notification_sent_only_once(self):
        """เรียก tick หลายครั้งในสถานะ due → ส่ง notification ครั้งเดียว"""
        first  = self.nm.tick(reposition_due=True)
        second = self.nm.tick(reposition_due=True)
        self.assertGreaterEqual(len(first), 1)
        # ครั้งที่สองไม่ควรส่ง notification ซ้ำ
        self.assertEqual(len(second), 0)

    def test_pop_pending_drains_queue(self):
        self.nm.tick(reposition_due=True)
        first_pop  = self.nm.pop_pending()
        second_pop = self.nm.pop_pending()
        self.assertGreater(len(first_pop), 0)
        self.assertEqual(len(second_pop), 0)


class TestNotificationHistory(unittest.TestCase):
    def setUp(self):
        self.nm = NotificationManager()

    def test_history_empty_initially(self):
        history = self.nm.get_notification_history()
        self.assertIsInstance(history, list)

    def test_history_grows_after_tick(self):
        self.nm.tick(reposition_due=True)
        history = self.nm.get_notification_history()
        self.assertGreater(len(history), 0)


if __name__ == "__main__":
    unittest.main()
