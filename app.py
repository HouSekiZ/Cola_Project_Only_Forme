"""
Patient Assist System - Flask Backend
Slim routes only, all logic in core/ and detectors/
"""
import os
import time
import atexit
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS

from core.camera_manager import CameraManager
from core.alarm_manager import AlarmManager, AlarmType
from core.detector import Detector
from config import Config, DevelopmentConfig, ProductionConfig, TestingConfig
from utils.logger import setup_logger

# ── App Setup ──────────────────────────────────────────────
app = Flask(__name__)

env = os.getenv('FLASK_ENV', 'development')
if env == 'production':
    app.config.from_object(ProductionConfig)
elif env == 'testing':
    app.config.from_object(TestingConfig)
else:
    app.config.from_object(DevelopmentConfig)

CORS(app)

logger = setup_logger(
    name='patient_assist',
    log_file=app.config.get('LOG_FILE', 'logs/patient_assist.log'),
    level=app.config.get('LOG_LEVEL', 'INFO')
)

# ── Global Instances ───────────────────────────────────────
camera_manager = CameraManager(initial_camera_index=0)
alarm_manager = AlarmManager()
detector = Detector(camera_manager, alarm_manager)

VALID_MODES = ['ALL', 'EYE', 'HAND', 'BODY']


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ── Video Streaming ────────────────────────────────────────

def generate_frames():
    while True:
        try:
            frame_data = detector.process_frame()
            if frame_data is None:
                time.sleep(0.03)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + frame_data['encoded_frame'] + b'\r\n')
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Alarm API ──────────────────────────────────────────────

@app.route('/api/status')
def get_status():
    active = alarm_manager.get_active_alarm()
    return jsonify({
        'alarm': active is not None,
        'alarm_type': active.type.value if active else None,
        'timestamp': active.timestamp if active else None
    })


@app.route('/api/acknowledge', methods=['POST'])
def acknowledge_alarm():
    success = alarm_manager.acknowledge_alarm()
    return jsonify({'success': success})


@app.route('/api/alarm_history')
def get_alarm_history():
    limit = request.args.get('limit', 10, type=int)
    history = alarm_manager.get_history(limit=limit)
    return jsonify({'history': history})


# ── Camera Management API ──────────────────────────────────

@app.route('/api/cameras')
def list_cameras():
    try:
        cameras = CameraManager.list_cameras()
        return jsonify(cameras)
    except Exception as e:
        logger.error(f"List cameras error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/select_camera', methods=['POST'])
def select_camera():
    try:
        data = request.get_json()
        if not data or 'index' not in data:
            return jsonify({'status': 'error', 'message': 'Missing index'}), 400
        index = int(data['index'])
        success = camera_manager.switch_camera(index)
        if success:
            return jsonify({'status': 'ok', 'camera_index': index})
        return jsonify({'status': 'error', 'message': 'Failed to switch camera'}), 500
    except Exception as e:
        logger.error(f"select_camera error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Mode Management API ────────────────────────────────────

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """Change detection mode: ALL / EYE / HAND / BODY"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        mode = data.get('mode', '').upper()
        if mode not in VALID_MODES:
            return jsonify({'error': f'Invalid mode: {mode}. Valid: {VALID_MODES}'}), 400

        detector.set_mode(mode)
        logger.info(f"Mode set to: {mode}")
        return jsonify({'status': 'ok', 'mode': mode})

    except Exception as e:
        logger.error(f"set_mode error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Body Position API ──────────────────────────────────────

@app.route('/api/body_position')
def get_body_position():
    try:
        return jsonify(detector.get_position_status())
    except Exception as e:
        logger.error(f"get_body_position error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/position_history')
def get_position_history():
    limit = request.args.get('limit', 50, type=int)
    try:
        return jsonify(detector.get_position_history(limit=limit))
    except Exception as e:
        logger.error(f"get_position_history error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset_position_timer', methods=['POST'])
def reset_position_timer():
    try:
        detector.reset_reposition_timer()
        return jsonify({'ok': True, 'message': 'Reposition timer reset'})
    except Exception as e:
        logger.error(f"reset_position_timer error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Meal Notification API ──────────────────────────────────

@app.route('/api/meal_times', methods=['GET'])
def get_meal_times():
    try:
        return jsonify(detector.get_meal_status())
    except Exception as e:
        logger.error(f"get_meal_times error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/meal_times', methods=['POST'])
def set_meal_times():
    data = request.get_json(silent=True) or {}
    meal_times = {k: v for k, v in data.items()
                  if k in {'breakfast', 'lunch', 'dinner'}}
    if not meal_times:
        return jsonify({'error': 'No valid meal times provided'}), 400
    try:
        detector.set_meal_times(meal_times)
        return jsonify({'ok': True, 'meal_times': meal_times})
    except Exception as e:
        logger.error(f"set_meal_times error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/meal_eaten/<string:meal_name>', methods=['POST'])
def mark_meal_eaten(meal_name: str):
    if meal_name not in {'breakfast', 'lunch', 'dinner'}:
        return jsonify({'error': f'Invalid meal name'}), 400
    try:
        success = detector.mark_meal_eaten(meal_name)
        if success:
            return jsonify({'ok': True, 'meal': meal_name})
        return jsonify({'error': 'Meal not found'}), 404
    except Exception as e:
        logger.error(f"mark_meal_eaten error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Notifications API ──────────────────────────────────────

@app.route('/api/notifications/pending')
def get_pending_notifications():
    try:
        notifications = detector.pop_pending_notifications()
        return jsonify([n.to_dict() if hasattr(n, 'to_dict') else n
                        for n in notifications])
    except Exception as e:
        logger.error(f"get_pending_notifications error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Health Check ───────────────────────────────────────────

@app.route('/api/health')
def health_check():
    camera_ok = camera_manager.running
    return jsonify({
        'status': 'ok' if camera_ok else 'degraded',
        'timestamp': time.time(),
        'components': {
            'camera': 'ok' if camera_ok else 'error',
            'alarm': 'ok',
            'detector': 'ok'
        },
        'metrics': {
            'fps': detector.get_fps(),
            'active_alarms': 1 if alarm_manager.has_active_alarm() else 0,
            'current_mode': detector.current_mode
        }
    })


# ══════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ══════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════

def _cleanup():
    try:
        camera_manager.close()
    except Exception:
        pass

atexit.register(_cleanup)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 5000),
        debug=app.config.get('DEBUG', False)
    )