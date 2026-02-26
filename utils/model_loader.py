"""
MediaPipe model loading utilities
"""
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BASE_DIR, 'models')


def _model_path(filename: str) -> str:
    path = os.path.join(_MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Please download it to the models/ directory."
        )
    return path


def load_face_landmarker():
    """Load MediaPipe Face Landmarker (VIDEO mode)"""
    model_path = _model_path('face_landmarker.task')

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def load_hand_landmarker():
    """Load MediaPipe Hand Landmarker"""
    model_path = _model_path('hand_landmarker.task')

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_vision.HandLandmarker.create_from_options(options)
