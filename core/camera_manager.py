"""
CameraManager: handles camera I/O, lifecycle, and switching (thread-safe)
"""
import threading
import cv2
from typing import Optional, List, Dict, Tuple

from utils.logger import setup_logger

logger = setup_logger('camera_manager')


class CameraManager:
    """จัดการ camera lifecycle และ frame streaming (thread-safe)"""

    def __init__(self, initial_camera_index: int = 0):
        self.camera_index = initial_camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.running = False
        self._init_camera()

    def _init_camera(self):
        """เปิดกล้อง"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                logger.info(f"Released previous camera")

            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.running = False
                raise RuntimeError(f"Cannot open camera {self.camera_index}")

            self.running = True
            logger.info(f"Camera {self.camera_index} opened successfully")

    def get_frame(self) -> Optional[Tuple]:
        """อ่าน frame จากกล้อง (thread-safe)"""
        with self.lock:
            if not self.running or self.cap is None:
                return None
            success, frame = self.cap.read()
            if not success:
                logger.warning("Failed to read frame from camera")
                return None
            return (frame, success)

    def switch_camera(self, new_index: int) -> bool:
        """เปลี่ยนกล้อง"""
        try:
            self.camera_index = new_index
            self._init_camera()
            logger.info(f"Switched to camera {new_index}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to camera {new_index}: {e}")
            return False

    @staticmethod
    def list_cameras() -> List[Dict]:
        """ค้นหากล้องที่มีในระบบ (index 0-9)"""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append({
                    'index': i,
                    'name': f'Camera {i}'
                })
                cap.release()
        logger.info(f"Found {len(cameras)} cameras: {[c['index'] for c in cameras]}")
        return cameras

    def close(self):
        """ปิดกล้องและ release resources"""
        with self.lock:
            self.running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                logger.info("Camera closed")
