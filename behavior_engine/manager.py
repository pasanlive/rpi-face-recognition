import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from config import Config
from .facial_behavior import FacialBehaviorAnalyzer
from .pose_behavior import PoseBehaviorAnalyzer
from activity_logger.logger import ActivityLogger

logger = logging.getLogger(__name__)

def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    try:
        hex_str = str(hex_str).lstrip('#')
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return (b, g, r)
    except Exception:
        pass
    return (0, 255, 255)

class BehaviorEngineManager:
    """
    Unified Manager coordinating Facial Behavior Analysis, Body Pose & Motion Tracking,
    Multi-Zone Room Management, Activity Logging, and Visual HUD Annotation Rendering.
    """

    def __init__(self, activity_logger: Optional[ActivityLogger] = None, get_camera_index_callable=None):
        self.facial_analyzer = FacialBehaviorAnalyzer()
        self.pose_analyzer = PoseBehaviorAnalyzer(get_camera_index_callable)
        self.activity_logger = activity_logger or ActivityLogger()

    def process_frame(
        self,
        frame: np.ndarray,
        recognition_results: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if frame is None or frame.size == 0:
            return frame, []

        annotated_frame = frame.copy()
        telemetry_results = []
        h, w = frame.shape[:2]

        # Draw Multi-Zone Security Polygons and Room Labels
        if Config.ENABLE_POSE_BEHAVIOR:
            self.pose_analyzer._reload_security_zone_if_needed()
            multi_zones = getattr(self.pose_analyzer, "multi_zones", [])
            for zone in multi_zones:
                poly = zone.get("polygon", [])
                name = zone.get("name", "Zone")
                sec_level = zone.get("security_level", "Medium")
                color_hex = zone.get("color", "#10b981")
                bgr = hex_to_bgr(color_hex)

                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    # Draw polygon border
                    cv2.polylines(annotated_frame, [pts], isClosed=True, color=bgr, thickness=2, lineType=cv2.LINE_AA)
                    
                    # Fill semi-transparent overlay
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [pts], color=bgr)
                    cv2.addWeighted(overlay, 0.12, annotated_frame, 0.88, 0, annotated_frame)

                    # Label text at first polygon vertex
                    lx, ly = poly[0][0] + 8, poly[0][1] + 24
                    lx = min(w - 150, max(10, lx))
                    ly = min(h - 20, max(25, ly))
                    label_str = f"[{name}] ({sec_level})"
                    cv2.putText(annotated_frame, label_str, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(annotated_frame, label_str, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 1, cv2.LINE_AA)

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

            # 2. Body Pose & Motion Analysis & Cross-Zone Transition Tracking
            if Config.ENABLE_POSE_BEHAVIOR:
                pose_telemetry = self.pose_analyzer.analyze_pose_and_motion(bbox)
                beh_data.update(pose_telemetry)

                # Check Cross-Zone Movement Transition (e.g. Room 1 -> Room 2)
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                multi_zones = getattr(self.pose_analyzer, "multi_zones", [])
                transition_evt = self.pose_analyzer.zone_transition_tracker.update_person_position(
                    person_name=identity,
                    point=centroid,
                    zones=multi_zones,
                    activity_logger=self.activity_logger,
                    frame_crop=face_crop
                )
                if transition_evt:
                    beh_data["cross_zone_event"] = transition_evt

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
