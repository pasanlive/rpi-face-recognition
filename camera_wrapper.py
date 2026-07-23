import sys
import os
import cv2
import time
import logging
import threading
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
    Supports:
      - Picamera2 for Raspberry Pi CSI Camera Modules (IMX708)
      - OpenCV VideoCapture (V4L2) for USB Webcams
      - RTSP / RTMP / HTTP / File video streams with background thread reading
    """

    def __init__(self, source: Union[int, str] = 0, width: Optional[int] = None, height: Optional[int] = None):
        from config import Config
        self.source = source
        self.width = width if width is not None else Config.CAMERA_WIDTH
        self.height = height if height is not None else Config.CAMERA_HEIGHT
        self.picam2 = None
        self.cap = None
        self.is_picam = False
        self.is_rtsp = False
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self._open()

    def _open(self):
        # Determine numerical or string URL source
        is_num = isinstance(self.source, int) or (isinstance(self.source, str) and str(self.source).isdigit())
        
        if is_num:
            self.source = int(self.source)

        # Detect RTSP or Network video streams
        if isinstance(self.source, str) and (
            self.source.startswith(("rtsp://", "rtmps://", "rtmp://", "http://", "https://")) or
            os.path.isfile(self.source)
        ):
            self.is_rtsp = True

        # For RTSP network streams, configure OpenCV FFMPEG TCP transport options to eliminate packet loss artifacts
        if isinstance(self.source, str) and self.source.startswith(("rtsp://", "rtmps://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|allowed_media_types;video"

        # Try Picamera2 for numerical CSI camera sources
        if PICAMERA2_AVAILABLE and is_num:
            try:
                logger.info(f"Attempting Picamera2 initialization for RPi CSI Camera ({self.width}x{self.height})...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(main={"size": (self.width, self.height)})
                self.picam2.configure(config)
                self.picam2.start()
                self.is_picam = True
                logger.info(f"Picamera2 started successfully at {self.width}x{self.height}.")
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

        # OpenCV VideoCapture fallback for USB webcams & RTSP streams
        logger.info(f"Opening OpenCV VideoCapture source '{self.source}' ({self.width}x{self.height})...")
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            # RTSP stream or video file
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)

        if self.cap and self.cap.isOpened():
            # Minimize buffer delay for network RTSP streams
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Start background thread to continually pull frames and eliminate buffer lag
            self.running = True
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info("OpenCV VideoCapture background frame reader started.")

    def _reader_loop(self):
        """Background thread continuously pulling frames to prevent buffer backlog."""
        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
                with self.lock:
                    self.latest_frame = frame
            except Exception as e:
                logger.error(f"Error in camera background reader loop: {e}")
                time.sleep(0.05)

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

        if self.running:
            with self.lock:
                if self.latest_frame is not None:
                    return True, self.latest_frame.copy()
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
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None

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

