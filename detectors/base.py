"""
Base detector interface - all detectors must implement this
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseDetector(ABC):
    """Base class สำหรับ detector ทุกประเภท"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = True

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        ตรวจจับจาก frame

        Returns:
            {
                'detected': bool,
                'confidence': float,
                'data': Any,      # detector-specific data
                'alarm': bool     # ควรแจ้งเตือนหรือไม่
            }
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset internal state"""
        pass

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
