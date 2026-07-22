import sys
import os
import cv2
import logging
import numpy as np
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

# Add Raspberry Pi system python package path for virtual environments (venv)
for sys_path in ["/usr/lib/python3/dist-packages", "/usr/lib/python3.11/dist-packages", "/usr/lib/python3.13/dist-packages"]:
    if os.path.exists(sys_path) and sys_path not in sys.path:
        sys.path.append(sys_path)

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False

class CameraWrapper:
    """
    Unified Camera Manager for Raspberry Pi 5 & Desktop setups.
    Automatically prioritizes Picamera2 for CSI Cameras (e.g. Raspberry Pi Camera Module 3 IMX708)
    and falls back to OpenCV VideoCapture (V4L2) for USB Webcams or RTSP streams.
    """

    def __init__(self, source: Union[int, str] = 0):
        self.source = source
        self.picam2 = None
        self.cap = None
        self.is_picam = False
        self._open()

    def _open(self):
        # Try Picamera2 for numerical CSI camera sources
        is_num = isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit())
        if PICAMERA2_AVAILABLE and is_num:
            try:
                logger.info("Attempting Picamera2 initialization for Raspberry Pi CSI Camera Module...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
                self.picam2.configure(config)
                self.picam2.start()
                self.is_picam = True
                logger.info("Picamera2 started successfully.")
                return
            except Exception as e:
                logger.warning(f"Picamera2 init failed ({e}). Falling back to OpenCV VideoCapture.")
                if self.picam2:
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                self.picam2 = None
                self.is_picam = False

        # OpenCV VideoCapture fallback for USB webcams & streams
        idx = int(self.source) if is_num else self.source
        logger.info(f"Opening OpenCV VideoCapture source '{idx}'...")
        if isinstance(idx, int):
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx)
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.cap = cv2.VideoCapture(idx)

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def is_opened(self) -> bool:
        if self.is_picam and self.picam2:
            return True
        return bool(self.cap and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.is_picam and self.picam2:
            try:
                frame_rgb = self.picam2.capture_array()
                if frame_rgb is None or frame_rgb.size == 0:
                    return False, None
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                return True, frame_bgr
            except Exception as e:
                logger.error(f"Picamera2 capture error: {e}")
                return False, None

        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    return False, None
                return True, frame
            except Exception as e:
                logger.error(f"OpenCV VideoCapture error: {e}")
                return False, None

        return False, None

    def release(self):
        if self.is_picam and self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                logger.warning(f"Error stopping Picamera2: {e}")
            self.picam2 = None
            self.is_picam = False

        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing OpenCV VideoCapture: {e}")
            self.cap = None
