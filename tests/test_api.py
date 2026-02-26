import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock


class TestAPI(unittest.TestCase):

    def setUp(self):
        # Patch heavy dependencies before importing app
        mock_cam = MagicMock()
        mock_cam.running = True

        mock_det = MagicMock()
        mock_det.current_mode = 'EYE'
        mock_det.get_fps.return_value = 30.0

        mock_alarm = MagicMock()
        mock_alarm.get_active_alarm.return_value = None
        mock_alarm.has_active_alarm.return_value = False
        mock_alarm.get_history.return_value = []

        with patch('core.camera_manager.CameraManager', return_value=mock_cam), \
             patch('core.alarm_manager.AlarmManager', return_value=mock_alarm), \
             patch('core.detector.Detector', return_value=mock_det):
            import app as flask_app
            flask_app.camera_manager = mock_cam
            flask_app.alarm_manager = mock_alarm
            flask_app.detector = mock_det
            self.client = flask_app.app.test_client()
            self.client.testing = True

    def test_health_check(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
        self.assertIn('components', data)

    def test_get_status_no_alarm(self):
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('alarm', data)
        self.assertFalse(data['alarm'])

    def test_set_mode_eye(self):
        response = self.client.post('/api/set_mode',
                                    json={'mode': 'EYE'},
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_set_mode_hand(self):
        response = self.client.post('/api/set_mode',
                                    json={'mode': 'HAND'},
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_set_mode_invalid(self):
        response = self.client.post('/api/set_mode',
                                    json={'mode': 'BODY'},
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_alarm_history(self):
        response = self.client.get('/api/alarm_history')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('history', data)

    def test_not_found(self):
        response = self.client.get('/api/doesnotexist')
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
