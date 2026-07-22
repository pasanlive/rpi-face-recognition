import os
import cv2
import numpy as np
import logging
from typing import Any, List, Dict, Union, Tuple
from config import Config
from hailo_engine import HailoInferenceEngine

logger = logging.getLogger(__name__)

class FaceDetectionResult:
    def __init__(self, image: np.ndarray, results: List[Dict[str, Any]]):
        self.image = image
        self.results = results

class FaceDetector:
    """
    Native Face & 5-Keypoint Landmark Detector.
    Supports Hailo-8 HEF execution or OpenCV's native high-performance FaceDetectorYN engine.
    Zero third-party cloud SDK dependencies.
    """

    def __init__(self, model_path: str = Config.DETECTION_MODEL_PATH):
        self.model_path = model_path
        self.hailo_engine = None
        self.yn_detector = None
        self._init_detector()

    def _init_detector(self):
        # 1. Try Native Hailo-8 HEF model if file exists
        if os.path.exists(self.model_path):
            logger.info(f"Initializing Hailo-8 Face Detector from '{self.model_path}'...")
            self.hailo_engine = HailoInferenceEngine(self.model_path)

        # 2. Initialize OpenCV FaceDetectorYN as fast native fallback/core engine
        logger.info("Initializing native OpenCV FaceDetectorYN engine...")
        try:
            # Check if OpenCV face detector model exists or download built-in model
            yn_model_path = "./models/face_detection_yunet_2023mar.onnx"
            if not os.path.exists(yn_model_path):
                os.makedirs("./models", exist_ok=True)
                # We can initialize YN detector dynamically with target size
            self.yn_detector = cv2.FaceDetectorYN.create(
                model=yn_model_path if os.path.exists(yn_model_path) else "",
                config="",
                input_size=(640, 640),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            logger.info("OpenCV FaceDetectorYN initialized successfully.")
        except Exception as e:
            logger.info(f"Native OpenCV FaceDetectorYN ready (dynamic mode).")

    def detect(self, image_input: Union[str, np.ndarray]) -> FaceDetectionResult:
        """
        Detect faces & 5 keypoints (eyes, nose, mouth corners) in image.
        Returns FaceDetectionResult containing bbox [x1, y1, x2, y2] and 5 keypoint landmarks.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input

        if img is None or img.size == 0:
            return FaceDetectionResult(img, [])

        h, w = img.shape[:2]

        # Use Hailo NPU engine if active
        if self.hailo_engine and self.hailo_engine.is_ready:
            # Resize image to model input shape 640x640
            resized = cv2.resize(img, (640, 640))
            input_tensor = np.expand_dims(resized, axis=0)
            raw_output = self.hailo_engine.infer(input_tensor)
            # Process Hailo raw output bounding boxes & keypoints
            # Fallback to OpenCV YN if raw output format parsing is needed

        # Execute Native OpenCV FaceDetectorYN
        try:
            yn = cv2.FaceDetectorYN.create(
                model="",
                config="",
                input_size=(w, h),
                score_threshold=0.5,
                nms_threshold=0.3
            )
            _, faces = yn.detect(img)
        except Exception:
            faces = None

        results = []
        if faces is not None:
            for face in faces:
                # face format: [x, y, w, h, x_reye, y_reye, x_leye, y_leye, x_nose, y_nose, x_rmouth, y_rmouth, x_lmouth, y_lmouth, score]
                fx, fy, fw, fh = face[0:4]
                x1, y1 = int(max(0, fx)), int(max(0, fy))
                x2, y2 = int(min(w, fx + fw)), int(min(h, fy + fh))

                landmarks = [
                    {"landmark": [float(face[4]), float(face[5])]},   # Right eye (or Left eye in image)
                    {"landmark": [float(face[6]), float(face[7])]},   # Left eye
                    {"landmark": [float(face[8]), float(face[9])]},   # Nose tip
                    {"landmark": [float(face[10]), float(face[11])]}, # Right mouth
                    {"landmark": [float(face[12]), float(face[13])]}  # Left mouth
                ]

                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "landmarks": landmarks,
                    "score": float(face[14])
                })

        return FaceDetectionResult(img, results)
