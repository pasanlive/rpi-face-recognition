import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

from config import Config
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.alignment import align_and_crop
from database.manager import FaceDatabaseManager
from behavior_engine.manager import BehaviorEngineManager
from activity_logger.logger import ActivityLogger

logger = logging.getLogger(__name__)

class FaceRecognitionPipeline:
    """
    End-to-End Real-Time Face & Behavioral Recognition Pipeline on Raspberry Pi 5 with Hailo-8.
    Stages:
        1. Face & Keypoint Landmark Detection (SCRFD / YuNet)
        2. Affine Alignment to ArcFace 112x112 layout
        3. Feature Embedding Extraction (ArcFace 512-D / Hailo-8 NPU)
        4. LanceDB Vector Search & Identity Classification
        5. Facial & Body Behavior Analysis (3D Head Pose, EAR/Drowsiness, MAR, Posture, Loitering)
        6. Activity Event Logging & Snapshot Capture
        7. Visual HUD Overlay Annotation
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        db_manager: Optional[FaceDatabaseManager] = None,
        activity_logger: Optional[ActivityLogger] = None,
        get_camera_index_callable=None
    ):
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        self.db_manager = db_manager or FaceDatabaseManager()
        self.activity_logger = activity_logger or ActivityLogger()
        self.behavior_manager = BehaviorEngineManager(activity_logger=self.activity_logger, get_camera_index_callable=get_camera_index_callable)

    def process_frame(
        self,
        frame: np.ndarray,
        threshold: float = Config.SIMILARITY_THRESHOLD
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single image frame (BGR format) and return the annotated frame along with detection metadata.
        """
        if frame is None or frame.size == 0:
            return frame, []

        detection_results = self.detector.detect(frame)

        if not hasattr(detection_results, "results") or not detection_results.results:
            annotated_frame, _ = self.behavior_manager.process_frame(frame, [])
            return annotated_frame, []

        aligned_faces = []
        bboxes = []
        landmarks_list = []

        for face in detection_results.results:
            bbox = [int(v) for v in face["bbox"]]
            landmarks = face["landmarks"]
            bboxes.append(bbox)
            landmarks_list.append(landmarks)

        identities_and_scores = []
        if getattr(Config, "ENABLE_FACE_RECOGNITION", True):
            aligned_faces = []
            for landmarks in landmarks_list:
                aligned_face, _ = align_and_crop(frame, [lm["landmark"] for lm in landmarks], image_size=Config.INPUT_FACE_SIZE)
                aligned_faces.append(aligned_face)

            embeddings = self.embedder.extract_batch(aligned_faces)
            identities_and_scores = self.db_manager.identify_batch(embeddings, threshold=threshold)
        else:
            identities_and_scores = [("Person", 1.0) for _ in bboxes]

        recognition_results = []
        for bbox, landmarks, (name, score) in zip(bboxes, landmarks_list, identities_and_scores):
            recognition_results.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "identity": name,
                "score": score
            })

        # Process Facial Behavior, Body Motion Tracking, HUD overlays, and Activity Logging
        annotated_frame, telemetry = self.behavior_manager.process_frame(frame, recognition_results)

        return annotated_frame, telemetry
