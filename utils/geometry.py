"""
Math helpers: EAR, gaze estimation, face distance, hand state
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional


def calculate_ear(landmarks, indices: List[int], w: int, h: int) -> Tuple[float, List]:
    """
    Calculate Eye Aspect Ratio (EAR) from landmarks.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        landmarks: MediaPipe face landmarks
        indices: 6 landmark indices for the eye
        w, h: frame width and height

    Returns:
        (ear_value, eye_points)
    """
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))

    if len(pts) < 6:
        return 0.0, []

    p1, p2, p3, p4, p5, p6 = pts

    vertical1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

    if horizontal == 0:
        return 0.0, pts

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear, pts


def estimate_gaze(landmarks, w: int, h: int) -> Tuple[float, float, bool]:
    """
    Estimate head yaw/pitch using solvePnP.

    Returns:
        (yaw_degrees, pitch_degrees, is_looking_at_camera)
    """
    # 3D model points (generic head model)
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip (1)
        (0.0, -330.0, -65.0),   # Chin (152)
        (-225.0, 170.0, -135.0),  # Left eye corner (263)
        (225.0, 170.0, -135.0),   # Right eye corner (33)
        (-150.0, -150.0, -125.0), # Left mouth (287)
        (150.0, -150.0, -125.0),  # Right mouth (57)
    ], dtype=np.float64)

    key_indices = [1, 152, 263, 33, 287, 57]
    image_points = []

    try:
        for idx in key_indices:
            lm = landmarks[idx]
            image_points.append((lm.x * w, lm.y * h))

        image_points = np.array(image_points, dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        if not success:
            return 0.0, 0.0, False

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

        yaw = angles[1]
        pitch = angles[0]

        gaze_threshold = 45
        is_looking = abs(yaw) < gaze_threshold and abs(pitch) < gaze_threshold

        return float(yaw), float(pitch), is_looking

    except Exception:
        return 0.0, 0.0, False


def face_distance_ratio(landmarks, w: int, h: int) -> float:
    """
    Estimate how close the face is by inter-eye distance / frame width.

    Returns:
        ratio (higher = closer)
    """
    try:
        left_eye = landmarks[263]
        right_eye = landmarks[33]

        dx = (left_eye.x - right_eye.x) * w
        dy = (left_eye.y - right_eye.y) * h
        eye_dist = np.sqrt(dx ** 2 + dy ** 2)

        return eye_dist / w
    except Exception:
        return 0.0


def check_hand_state(landmarks) -> str:
    """
    Classify hand gesture from landmarks.

    Returns:
        'OPEN', 'FIST', or 'UNKNOWN'
    """
    finger_tips = [8, 12, 16, 20]
    finger_joints = [6, 10, 14, 18]

    try:
        extended = 0
        for tip_idx, joint_idx in zip(finger_tips, finger_joints):
            tip = landmarks[tip_idx]
            joint = landmarks[joint_idx]
            if tip.y < joint.y:
                extended += 1

        if extended == 4:
            return "OPEN"
        elif extended == 0:
            return "FIST"
        else:
            return "UNKNOWN"
    except Exception:
        return "UNKNOWN"
