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

# ── Database (optional — graceful fallback) ──
_db_available = False
try:
    from database import init_db, is_db_available, get_db, repo
    DB_ENABLED = True
except ImportError:
    DB_ENABLED = False

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

# ── DB Init (non-blocking, warn ถ้าไม่มี MySQL) ──
if DB_ENABLED and app.config.get('DB_ENABLED', True):
    _db_available = init_db()
    if _db_available:
        logger.info("MySQL database connected and tables ready.")

        # ── Register alarm DB hook ──────────────────────────────
        def _alarm_db_hook(alarm_type: str, metadata: dict):
            """บันทึก alarm event ลง DB ทุกครั้งที่ trigger"""
            try:
                with get_db() as db:
                    repo.save_alarm(db, alarm_type=alarm_type, metadata=metadata)
            except Exception as exc:
                logger.warning(f"DB save_alarm failed: {exc}")

        alarm_manager.on_alarm_triggered = _alarm_db_hook
        # ───────────────────────────────────────────────────────

    else:
        logger.warning("MySQL not available — running without database persistence.")


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
    active = alarm_manager.get_active_alarm()
    success = alarm_manager.acknowledge_alarm()

    # ── DB: บันทึก acknowledged_at ──
    if success and active and _db_available:
        try:
            with get_db() as db:
                repo.acknowledge_alarm_db(db, active.type.value)
        except Exception as e:
            logger.warning(f"DB acknowledge_alarm failed: {e}")

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


# ── Overlay Toggle API ─────────────────────────────────────

@app.route('/api/overlay/state')
def get_overlay_state():
    """คืนสถานะ overlay ทุกตัว"""
    return jsonify(detector.get_overlay_state())


@app.route('/api/overlay/<string:name>', methods=['POST'])
def toggle_overlay(name: str):
    """toggle overlay eye/hand/body"""
    try:
        new_state = detector.toggle_overlay(name)
        return jsonify({'overlay': name, 'visible': new_state})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


# ── Export API ─────────────────────────────────────────────

import csv
import io
from flask import make_response

@app.route('/api/export/alarms.csv')
def export_alarms_csv():
    """Export alarm events เป็น CSV"""
    try:
        rows = []
        if _db_available:
            with get_db() as db:
                rows = repo.get_alarm_history(db, limit=1000)
        else:
            rows = [{
                'id': i+1,
                'alarm_type': a.type.value,
                'triggered_at': a.timestamp,
                'acknowledged': False,
            } for i, a in enumerate(alarm_manager.get_history())]

        si = io.StringIO()
        writer = csv.DictWriter(si, fieldnames=['id','alarm_type','triggered_at','acknowledged','acknowledged_at'],
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        output = make_response(si.getvalue())
        output.headers['Content-Disposition'] = 'attachment; filename=alarm_events.csv'
        output.headers['Content-type'] = 'text/csv; charset=utf-8'
        return output
    except Exception as e:
        logger.error(f"export_alarms_csv error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/meals.csv')
def export_meals_csv():
    """Export meal records เป็น CSV"""
    try:
        rows = []
        if _db_available:
            with get_db() as db:
                rows = repo.get_meal_history(db, days=30)
        si = io.StringIO()
        writer = csv.DictWriter(si, fieldnames=['id','record_date','meal','scheduled_time','eaten','eaten_at'],
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        output = make_response(si.getvalue())
        output.headers['Content-Disposition'] = 'attachment; filename=meal_records.csv'
        output.headers['Content-type'] = 'text/csv; charset=utf-8'
        return output
    except Exception as e:
        logger.error(f"export_meals_csv error: {e}")
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
        # ── DB: upsert meal records สำหรับวันนี้ ──
        if _db_available:
            try:
                with get_db() as db:
                    for name, t in meal_times.items():
                        repo.upsert_meal(db, meal_name=name, scheduled_time=t)
            except Exception as e:
                logger.warning(f"DB set_meal_times failed: {e}")
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
        # ── DB: บันทึกว่าทานแล้ว ──
        if success and _db_available:
            try:
                with get_db() as db:
                    repo.mark_meal_eaten_db(db, meal_name)
            except Exception as e:
                logger.warning(f"DB mark_meal_eaten failed: {e}")
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
        result = [n.to_dict() if hasattr(n, 'to_dict') else n for n in notifications]
        # ── DB: บันทึก notifications ใน background ──
        if _db_available and notifications:
            try:
                with get_db() as db:
                    for n in notifications:
                        d = n.to_dict() if hasattr(n, 'to_dict') else n
                        repo.save_notification(
                            db,
                            notif_type=d.get('type', 'unknown'),
                            title=d.get('title', ''),
                            message=d.get('message', ''),
                        )
            except Exception as e:
                logger.warning(f"DB save_notification failed: {e}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"get_pending_notifications error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Database API ──────────────────────────────────────────────────────

@app.route('/api/db/status')
def db_status():
    return jsonify({
        'enabled':   DB_ENABLED,
        'available': _db_available if DB_ENABLED else False,
    })


@app.route('/api/db/alarm_history')
def db_alarm_history():
    if not _db_available:
        return jsonify({'error': 'Database not available'}), 503
    limit = request.args.get('limit', 50, type=int)
    try:
        with get_db() as db:
            return jsonify(repo.get_alarm_history(db, limit=limit))
    except Exception as e:
        logger.error(f"db_alarm_history error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/db/position_history')
def db_position_history():
    if not _db_available:
        return jsonify({'error': 'Database not available'}), 503
    limit = request.args.get('limit', 50, type=int)
    try:
        with get_db() as db:
            return jsonify(repo.get_position_history(db, limit=limit))
    except Exception as e:
        logger.error(f"db_position_history error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/db/position_stats')
def db_position_stats():
    if not _db_available:
        return jsonify({'error': 'Database not available'}), 503
    try:
        with get_db() as db:
            return jsonify(repo.get_position_stats_today(db))
    except Exception as e:
        logger.error(f"db_position_stats error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/db/meal_history')
def db_meal_history():
    if not _db_available:
        return jsonify({'error': 'Database not available'}), 503
    days = request.args.get('days', 7, type=int)
    try:
        with get_db() as db:
            return jsonify(repo.get_meal_history(db, days=days))
    except Exception as e:
        logger.error(f"db_meal_history error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/db/notification_history')
def db_notification_history():
    if not _db_available:
        return jsonify({'error': 'Database not available'}), 503
    limit = request.args.get('limit', 30, type=int)
    try:
        with get_db() as db:
            return jsonify(repo.get_notification_history(db, limit=limit))
    except Exception as e:
        logger.error(f"db_notification_history error: {e}")
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