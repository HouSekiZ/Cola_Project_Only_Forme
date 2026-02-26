"""
Base renderer interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseRenderer(ABC):
    """Base class สำหรับ renderer ทุกประเภท"""

    @abstractmethod
    def render(self, frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        """
        วาด overlay ลงบน frame

        Args:
            frame: original frame
            detection: result from detector.detect()

        Returns:
            frame with overlay drawn
        """
        pass