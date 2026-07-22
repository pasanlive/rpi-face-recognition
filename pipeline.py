import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

from config import Config
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.alignment import align_and_crop
from database.manager import FaceDatabaseManager

logger = logging.getLogger(__name__)

class FaceRecognitionPipeline:
    """
    End-to-End Real-Time Face Recognition Pipeline on Raspberry Pi 5 with Hailo-8.
    Stages:
        1. Face & Keypoint Landmark Detection (SCRFD / RetinaFace)
        2. Affine Alignment to ArcFace 112x112 layout
        3. Feature Embedding Extraction (ArcFace MobileFaceNet 512-D)
        4. LanceDB Vector Search & Identity Classification
        5. Visual Overlay Annotation
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        db_manager: Optional[FaceDatabaseManager] = None
    ):
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        self.db_manager = db_manager or FaceDatabaseManager()

    def process_frame(
        self,
        frame: np.ndarray,
        threshold: float = Config.SIMILARITY_THRESHOLD
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single image frame (BGR format) and return the annotated frame along with detection metadata.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]:
                annotated_frame: OpenCV BGR image with drawn bounding boxes, landmarks, labels, and scores.
                detections: List of dicts containing bbox, landmarks, name, score.
        """
        annotated_frame = frame.copy()
        detection_results = self.detector.detect(frame)

        if not hasattr(detection_results, "results") or not detection_results.results:
            return annotated_frame, []

        aligned_faces = []
        bboxes = []
        landmarks_list = []

        for face in detection_results.results:
            bbox = [int(v) for v in face["bbox"]]
            landmarks = [lm["landmark"] for lm in face["landmarks"]]
            bboxes.append(bbox)
            landmarks_list.append(landmarks)

            aligned_face, _ = align_and_crop(frame, landmarks, image_size=Config.INPUT_FACE_SIZE)
            aligned_faces.append(aligned_face)

        # Batch embed aligned faces on Hailo-8
        embeddings = self.embedder.extract_batch(aligned_faces)

        # Batch identify embeddings in LanceDB
        identities_and_scores = self.db_manager.identify_batch(embeddings, threshold=threshold)

        metadata = []
        for bbox, landmarks, (name, score) in zip(bboxes, landmarks_list, identities_and_scores):
            x1, y1, x2, y2 = bbox
            metadata.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "name": name,
                "score": score
            })

            # Color scheme: Green for recognized known person, Red/Orange for Unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw landmarks (eyes, nose, mouth corners)
            for lm in landmarks:
                lx, ly = int(lm[0]), int(lm[1])
                cv2.circle(annotated_frame, (lx, ly), 3, (255, 255, 0), -1)

            # Draw Label background banner
            label_str = f"{name} ({int(score * 100)}%)" if name != "Unknown" else "Unknown"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)

            banner_y1 = max(0, y1 - text_h - 10)
            banner_y2 = max(y1, text_h + 10)
            cv2.rectangle(
                annotated_frame,
                (x1, banner_y1),
                (x1 + text_w + 10, y1),
                color,
                -1
            )

            # Put text label
            cv2.putText(
                annotated_frame,
                label_str,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated_frame, metadata
