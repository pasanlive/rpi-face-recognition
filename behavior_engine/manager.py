import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from config import Config
from .facial_behavior import FacialBehaviorAnalyzer
from .pose_behavior import PoseBehaviorAnalyzer
from activity_logger.logger import ActivityLogger

logger = logging.getLogger(__name__)

class BehaviorEngineManager:
    """
    Unified Manager coordinating Facial Behavior Analysis, Body Pose & Motion Tracking,
    Activity Logging, and Visual HUD Annotation Rendering.
    """

    def __init__(self, activity_logger: Optional[ActivityLogger] = None, get_camera_index_callable=None):
        """Initialize BehaviorEngineManager.

        Args:
            activity_logger: Optional ActivityLogger instance.
            get_camera_index_callable: Callable that returns the current active camera index.
                If None, defaults to a function returning 0.
        """
        self.facial_analyzer = FacialBehaviorAnalyzer()
        self.pose_analyzer = PoseBehaviorAnalyzer(get_camera_index_callable)
        self.activity_logger = activity_logger or ActivityLogger()

    def process_frame(
        self,
        frame: np.ndarray,
        recognition_results: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame for facial & pose behaviors, trigger activity logs, and render visual HUD overlays.
        """
        if frame is None or frame.size == 0:
            return frame, []

        annotated_frame = frame.copy()
        telemetry_results = []
        h, w = frame.shape[:2]

        # Draw Security Zone Polygon boundary
        if Config.ENABLE_POSE_BEHAVIOR:
            # Ensure we have the latest polygon for the current camera
            self.pose_analyzer._load_security_zone()
            zone_pts = self.pose_analyzer.security_zone_polygon.reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [zone_pts], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(annotated_frame, "SECURITY ZONE", (110, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        for res in recognition_results:
            bbox = res["bbox"]
            identity = res.get("identity", "Unknown")
            landmarks = res.get("landmarks", [])
            x1, y1, x2, y2 = bbox

            beh_data = {}

            # 1. Facial Behavior Analysis
            if Config.ENABLE_FACIAL_BEHAVIOR and len(landmarks) >= 5:
                landmarks_5pt = [lm["landmark"] for lm in landmarks]
                facial_telemetry = self.facial_analyzer.analyze_face((h, w), landmarks_5pt, bbox)
                beh_data.update(facial_telemetry)

            # 2. Body Pose & Motion Analysis
            if Config.ENABLE_POSE_BEHAVIOR:
                pose_telemetry = self.pose_analyzer.analyze_pose_and_motion(bbox)
                beh_data.update(pose_telemetry)

            telemetry_results.append({
                "identity": identity,
                "bbox": bbox,
                "behavior": beh_data
            })

            # 3. Trigger Activity Logging for notable events
            face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if beh_data.get("is_drowsy", False):
                self.activity_logger.log_event(identity, "DROWSINESS DETECTED", beh_data, face_crop)
            elif beh_data.get("is_loitering", False):
                self.activity_logger.log_event(identity, "LOITERING DETECTED", beh_data, face_crop)
            elif beh_data.get("is_intrusion", False):
                self.activity_logger.log_event(identity, "ZONE INTRUSION", beh_data, face_crop)
            elif identity != "Unknown":
                self.activity_logger.log_event(identity, "RECOGNIZED PERSON", beh_data, face_crop)

            # 4. Render Visual HUD Overlay
            self._render_hud(annotated_frame, bbox, identity, beh_data)

        return annotated_frame, telemetry_results

    def _render_hud(
        self,
        frame: np.ndarray,
        bbox: List[int],
        identity: str,
        beh: Dict[str, Any]
    ):
        """
        Render sleek visual HUD overlay metrics onto frame.
        """
        x1, y1, x2, y2 = bbox
        
        # Color coding: Green for Recognized, Red for Alerts, Yellow for Unknown
        if beh.get("is_drowsy") or beh.get("is_loitering") or beh.get("is_intrusion"):
            box_color = (0, 0, 255) # Red for alert
        elif identity != "Unknown":
            box_color = (0, 255, 0) # Green for recognized
        else:
            box_color = (0, 255, 255) # Yellow

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # HUD Panel Text Lines
        lines = [f"ID: {identity}"]
        
        if "attention_score" in beh:
            lines.append(f"Focus: {beh['attention_score']}% ({beh.get('gaze_direction', 'Center')})")
        if "posture" in beh:
            lines.append(f"Pose: {beh['posture']}")
        if "dwell_time_sec" in beh:
            lines.append(f"Dwell: {beh['dwell_time_sec']}s")

        # Alerts
        if beh.get("is_drowsy"):
            lines.append("! ALERT: DROWSY !")
        if beh.get("is_yawning"):
            lines.append("! YAWNING DETECTED !")
        if beh.get("is_loitering"):
            lines.append("! ALERT: LOITERING !")
        if beh.get("is_intrusion"):
            lines.append("! INTRUSION DETECTED !")

        # Render panel box above/below bbox
        panel_y = max(20, y1 - (len(lines) * 18) - 5)
        for i, line_text in enumerate(lines):
            ty = panel_y + (i * 18)
            # Text background outline
            cv2.putText(frame, line_text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            # Text foreground
            txt_color = (0, 0, 255) if "!" in line_text else (255, 255, 255)
            cv2.putText(frame, line_text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_color, 1, cv2.LINE_AA)
