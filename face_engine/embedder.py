import os
import cv2
import numpy as np
import logging
import urllib.request
from typing import Any, List
from config import Config
from hailo_engine import HailoInferenceEngine

logger = logging.getLogger(__name__)

def ensure_sface_model(model_path: str = "./models/face_recognition_sface_2021dec.onnx") -> str:
    """
    Ensure the official high-accuracy SFace face embedding ONNX model is available locally.
    """
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    try:
        logger.info(f"Downloading OpenCV SFace face recognition model to '{model_path}'...")
        urllib.request.urlretrieve(url, model_path)
        logger.info("SFace model downloaded successfully.")
        return model_path
    except Exception as e:
        logger.warning(f"Could not download SFace model automatically: {e}")
        return ""

class FaceEmbedder:
    """
    Native Face Feature Embedding Extractor.
    Supports Hailo-8 NPU ArcFace HEF and OpenCV SFace ONNX engine.
    Zero third-party cloud SDK dependencies.
    """

    def __init__(self, model_path: str = Config.EMBEDDING_MODEL_PATH):
        self.model_path = model_path
        self.hailo_engine = None
        self.sf_recognizer = None
        self._init_embedder()

    def _init_embedder(self):
        # 1. Try Native Hailo-8 HEF model if file exists
        if os.path.exists(self.model_path):
            logger.info(f"Initializing Hailo-8 Face Embedder from '{self.model_path}'...")
            self.hailo_engine = HailoInferenceEngine(self.model_path)

        # 2. Native OpenCV SFace model
        sface_file = ensure_sface_model()
        if sface_file and os.path.exists(sface_file):
            try:
                self.sf_recognizer = cv2.FaceRecognizerSF.create(sface_file, "")
                logger.info("OpenCV FaceRecognizerSF initialized successfully.")
            except Exception as e:
                logger.warning(f"FaceRecognizerSF init error: {e}")

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract normalized feature embedding vector (512-D) for an aligned 112x112 face.
        """
        if aligned_face is None or aligned_face.size == 0:
            return np.zeros(Config.VECTOR_DIM, dtype=np.float32)

        # 1. Use Hailo-8 NPU HEF model if available
        if self.hailo_engine and self.hailo_engine.is_ready:
            input_face = cv2.resize(aligned_face, (112, 112))
            input_tensor = np.expand_dims(input_face, axis=0)
            raw_out = self.hailo_engine.infer(input_tensor)
            if raw_out is not None:
                first_key = list(raw_out.keys())[0]
                vec = raw_out[first_key].flatten().astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                if len(vec) == Config.VECTOR_DIM:
                    return vec

        # 2. Native OpenCV SFace feature extraction
        if self.sf_recognizer:
            try:
                aligned_112 = cv2.resize(aligned_face, (112, 112))
                feature = self.sf_recognizer.feature(aligned_112)
                vec = feature.flatten().astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                # Pad to 512 dimensions for LanceDB schema
                if len(vec) < Config.VECTOR_DIM:
                    vec = np.pad(vec, (0, Config.VECTOR_DIM - len(vec)))
                return vec[:Config.VECTOR_DIM]
            except Exception as e:
                logger.error(f"SFace feature extraction error: {e}")

        # Fail-safe color histogram embedding
        hist = cv2.calcHist([aligned_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        vec = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        if len(vec) < Config.VECTOR_DIM:
            vec = np.pad(vec, (0, Config.VECTOR_DIM - len(vec)))
        return vec[:Config.VECTOR_DIM]

    def extract_batch(self, aligned_faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract feature embeddings for a batch of aligned faces.
        """
        embeddings = []
        for face in aligned_faces:
            emb = self.extract_embedding(face)
            embeddings.append(emb)
        return embeddings
