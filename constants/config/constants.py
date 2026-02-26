# config/constants.py

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
LEFT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
