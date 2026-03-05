import os
from dotenv import load_dotenv

load_dotenv()

# ── Detection Constants ────────────────────────────────────

# Eye Detection
EYE_CONSTANTS = {
    'EAR_THRESHOLD': 0.22,
    'MIN_BLINK_DUR': 0.08,
    'MAX_BLINK_DUR': 0.6,
    'GAZE_THRESHOLD': 45
}

# Face Distance
FACE_CONSTANTS = {
    'CLOSE_RATIO': 0.18,
    'FAR_RATIO': 0.12
}

# SOS Pattern (FAR Mode)
SOS_CONSTANTS = {
    'PHASE1': 2,
    'PHASE2': 3,
    'PAUSE_MIN': 0.4,
    'PAUSE_MAX': 3.0,
    'PHASE_WIN': 4.0
}

# Near Mode
NEAR_CONSTANTS = {
    'BLINK_N': 3,
    'BLINK_WIN': 8.0
}

# Digital Zoom
ZOOM_CONSTANTS = {
    'FACTOR': 2.5,
    'DURATION': 1.5
}

# Landmark Indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]


class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # Camera
    DEFAULT_CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
    FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', 640))
    FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', 480))

    # Detection
    EAR_THRESHOLD = float(os.getenv('EAR_THRESHOLD', 0.22))
    GAZE_THRESHOLD = float(os.getenv('GAZE_THRESHOLD', 45))

    # Alarm
    ALARM_HISTORY_SIZE = int(os.getenv('ALARM_HISTORY_SIZE', 100))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/patient_assist.log')

    # ── MySQL Database ──────────────────────────────
    DB_HOST     = os.getenv('DB_HOST',     'localhost')
    DB_PORT     = int(os.getenv('DB_PORT', '3306'))
    DB_NAME     = os.getenv('DB_NAME',     'patient_assist')
    DB_USER     = os.getenv('DB_USER',     'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_ENABLED  = os.getenv('DB_ENABLED',  'true').lower() == 'true'



class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HOST = '0.0.0.0'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HOST = 'localhost'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
