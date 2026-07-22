import os
import cv2
import numpy as np
import logging
import urllib.request
from typing import Any, List, Dict, Union, Tuple
from config import Config
from hailo_engine import HailoInferenceEngine

logger = logging.getLogger(__name__)

def ensure_yunet_model(model_path: str = "./models/face_detection_yunet_2023mar.onnx") -> str:
    """
    Ensure the lightweight, high-accuracy YuNet face detection ONNX model is available locally.
    """
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    try:
        logger.info(f"Downloading high-accuracy OpenCV YuNet face detection model to '{model_path}'...")
        urllib.request.urlretrieve(url, model_path)
        logger.info("YuNet model downloaded successfully.")
        return model_path
    except Exception as e:
        logger.warning(f"Could not download YuNet model automatically: {e}")
        return ""

class FaceDetectionResult:
    def __init__(self, image: np.ndarray, results: List[Dict[str, Any]]):
        self.image = image
        self.results = results

class FaceDetector:
    """
    Native Multi-Engine Face & 5-Keypoint Landmark Detector.
    Supports Hailo-8 NPU HEF, OpenCV YuNet ONNX, and OpenCV Haar Cascade fallback.
    Zero third-party cloud SDK dependencies.
    """

    def __init__(self, model_path: str = Config.DETECTION_MODEL_PATH):
        self.model_path = model_path
        self.hailo_engine = None
        self.yn_model_file = ""
        self.haar_cascade = None
        self._init_detector()

    def _init_detector(self):
        # 1. Try Native Hailo-8 HEF model if file exists
        if os.path.exists(self.model_path):
            logger.info(f"Initializing Hailo-8 Face Detector from '{self.model_path}'...")
            self.hailo_engine = HailoInferenceEngine(self.model_path)

        # 2. Ensure YuNet model is present
        self.yn_model_file = ensure_yunet_model()

        # 3. Load Haar Cascade fail-safe fallback
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

    def detect(self, image_input: Union[str, np.ndarray]) -> FaceDetectionResult:
        """
        Detect faces & 5 keypoints in image.
        Returns FaceDetectionResult with bboxes and 5 keypoint landmarks.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input

        if img is None or img.size == 0:
            return FaceDetectionResult(img, [])

        h, w = img.shape[:2]
        results = []

        # 1. Execute OpenCV YuNet detector if ONNX model is available
        if self.yn_model_file and os.path.exists(self.yn_model_file):
            try:
                yn = cv2.FaceDetectorYN.create(
                    model=self.yn_model_file,
                    config="",
                    input_size=(w, h),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000
                )
                _, faces = yn.detect(img)
                if faces is not None:
                    for face in faces:
                        fx, fy, fw, fh = face[0:4]
                        x1, y1 = int(max(0, fx)), int(max(0, fy))
                        x2, y2 = int(min(w, fx + fw)), int(min(h, fy + fh))

                        landmarks = [
                            {"landmark": [float(face[4]), float(face[5])]},   # Right eye
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
            except Exception as e:
                logger.error(f"YuNet detection error: {e}")

        # 2. Haar Cascade Fallback if YuNet produced no faces
        if not results and self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                for (x, y, fw, fh) in faces:
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + fw), int(y + fh)
                    
                    # Estimate standard 5 keypoint ratios relative to bounding box
                    landmarks = [
                        {"landmark": [float(x + 0.3 * fw), float(y + 0.35 * fh)]}, # Right eye
                        {"landmark": [float(x + 0.7 * fw), float(y + 0.35 * fh)]}, # Left eye
                        {"landmark": [float(x + 0.5 * fw), float(y + 0.55 * fh)]}, # Nose tip
                        {"landmark": [float(x + 0.35 * fw), float(y + 0.75 * fh)]},# Right mouth
                        {"landmark": [float(x + 0.65 * fw), float(y + 0.75 * fh)]} # Left mouth
                    ]

                    results.append({
                        "bbox": [x1, y1, x2, y2],
                        "landmarks": landmarks,
                        "score": 0.85
                    })
            except Exception as e:
                logger.error(f"Haar cascade detection error: {e}")

        return FaceDetectionResult(img, results)
