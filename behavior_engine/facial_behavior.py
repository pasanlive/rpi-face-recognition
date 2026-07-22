import cv2
import numpy as np
import math
import time
import logging
from typing import Dict, Any, Tuple, List, Optional
from config import Config

logger = logging.getLogger(__name__)

class FacialBehaviorAnalyzer:
    """
    Analyzes 3D Head Pose (Pitch, Yaw, Roll), Eye Aspect Ratio (EAR for Drowsiness),
    Mouth Aspect Ratio (MAR for Yawning), and Real-Time Attention Score.
    """

    def __init__(self):
        # Standard 3D Facial Model Points for 5-point solvePnP (in mm)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (-225.0, 170.0, -135.0),     # Left eye
            (225.0, 170.0, -135.0),      # Right eye
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right Mouth corner
        ], dtype=np.float64)

        self.drowsy_start_time: Optional[float] = None
        self.blink_count = 0
        self.last_eye_open = True

    def _estimate_head_pose(
        self,
        landmarks_5pt: List[List[float]],
        image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float, str]:
        """
        Estimate Pitch, Yaw, Roll angles in degrees using cv2.solvePnP (SQPNP/EPNP).
        landmarks_5pt format: [RightEye, LeftEye, Nose, RightMouth, LeftMouth]
        """
        h, w = image_shape[:2]
        
        image_points_2d = np.array([
            landmarks_5pt[2], # Nose tip
            landmarks_5pt[1], # Left eye
            landmarks_5pt[0], # Right eye
            landmarks_5pt[4], # Left mouth
            landmarks_5pt[3]  # Right mouth
        ], dtype=np.float64)

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success = False
        rotation_vector = None
        translation_vector = None

        # SQPNP and EPNP work for 4+ and 5 points
        for flag in [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]:
            try:
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points_3d,
                    image_points_2d,
                    camera_matrix,
                    dist_coeffs,
                    flags=flag
                )
                if success:
                    break
            except Exception:
                continue

        if not success or rotation_vector is None:
            # Simple 2D geometric fallback for Yaw & Pitch
            right_eye, left_eye, nose = landmarks_5pt[0], landmarks_5pt[1], landmarks_5pt[2]
            dx = left_eye[0] - right_eye[0]
            dy = left_eye[1] - right_eye[1]
            roll = math.degrees(math.atan2(dy, dx if dx != 0 else 1e-5))

            eye_center_x = (right_eye[0] + left_eye[0]) / 2.0
            eye_center_y = (right_eye[1] + left_eye[1]) / 2.0
            yaw = (nose[0] - eye_center_x) * 0.5
            pitch = (nose[1] - eye_center_y) * 0.5
        else:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

            pitch = float(euler_angles[0][0])
            yaw = float(euler_angles[1][0])
            roll = float(euler_angles[2][0])

        # Categorize direction
        direction = "Center"
        if yaw > 15:
            direction = "Looking Right"
        elif yaw < -15:
            direction = "Looking Left"
        elif pitch > 15:
            direction = "Looking Down"
        elif pitch < -15:
            direction = "Looking Up"

        return pitch, yaw, roll, direction

    def _compute_ear(self, landmarks_5pt: List[List[float]], bbox: List[int]) -> float:
        """
        Compute Eye Aspect Ratio (EAR) metric from eye keypoint geometry.
        """
        right_eye = np.array(landmarks_5pt[0])
        left_eye = np.array(landmarks_5pt[1])
        eye_dist = np.linalg.norm(right_eye - left_eye)
        bbox_width = max(1.0, float(bbox[2] - bbox[0]))
        
        ear_ratio = eye_dist / bbox_width
        return float(ear_ratio)

    def _compute_mar(self, landmarks_5pt: List[List[float]]) -> float:
        """
        Compute Mouth Aspect Ratio (MAR) metric from mouth keypoints vs nose distance.
        """
        nose = np.array(landmarks_5pt[2])
        right_mouth = np.array(landmarks_5pt[3])
        left_mouth = np.array(landmarks_5pt[4])
        
        mouth_width = np.linalg.norm(right_mouth - left_mouth)
        mouth_center = (right_mouth + left_mouth) / 2.0
        vertical_dist = np.linalg.norm(nose - mouth_center)

        if mouth_width == 0:
            return 0.0
        return float(vertical_dist / mouth_width)

    def analyze_face(
        self,
        image_shape: Tuple[int, int],
        landmarks_5pt: List[List[float]],
        bbox: List[int]
    ) -> Dict[str, Any]:
        """
        Analyze facial telemetry: Head Pose, Drowsiness, Yawning, and Attention Score.
        """
        pitch, yaw, roll, gaze_dir = self._estimate_head_pose(landmarks_5pt, image_shape)
        ear = self._compute_ear(landmarks_5pt, bbox)
        mar = self._compute_mar(landmarks_5pt)

        now = time.time()
        is_drowsy = False
        if ear < Config.DROWSINESS_EAR_THRESHOLD:
            if self.drowsy_start_time is None:
                self.drowsy_start_time = now
            elif (now - self.drowsy_start_time) >= Config.DROWSINESS_TIME_SEC:
                is_drowsy = True
            
            if self.last_eye_open:
                self.blink_count += 1
                self.last_eye_open = False
        else:
            self.drowsy_start_time = None
            self.last_eye_open = True

        is_yawning = mar > Config.YAWN_MAR_THRESHOLD

        pose_penalty = min(50.0, (abs(yaw) + abs(pitch)) * 1.5)
        drowsy_penalty = 50.0 if is_drowsy else (25.0 if ear < Config.DROWSINESS_EAR_THRESHOLD else 0.0)
        attention_score = max(0.0, min(100.0, 100.0 - pose_penalty - drowsy_penalty))

        return {
            "pitch": round(pitch, 1),
            "yaw": round(yaw, 1),
            "roll": round(roll, 1),
            "gaze_direction": gaze_dir,
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "is_drowsy": is_drowsy,
            "is_yawning": is_yawning,
            "attention_score": round(attention_score, 1),
            "blink_count": self.blink_count
        }
