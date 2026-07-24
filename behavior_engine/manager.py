import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from config import Config
from .facial_behavior import FacialBehaviorAnalyzer
from .pose_behavior import PoseBehaviorAnalyzer
from .security_zone import get_scaled_zone_polygon
from .object_detector import COCOObjectDetector
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
    Multi-Zone Room Management, Object & Item Carrying Tracker, Activity Logging, and Visual HUD Rendering.
    """

    def __init__(self, activity_logger: Optional[ActivityLogger] = None, get_camera_index_callable=None):
        self.facial_analyzer = FacialBehaviorAnalyzer()
        self.pose_analyzer = PoseBehaviorAnalyzer(get_camera_index_callable)
        self.object_detector = COCOObjectDetector()
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

        # 1. Execute COCO Object Detection
        detected_objects = []
        if getattr(Config, "ENABLE_OBJECT_DETECTION", True):
            detected_objects = self.object_detector.detect_objects(frame)
            for obj in detected_objects:
                ox1, oy1, ox2, oy2 = obj["box"]
                olabel = obj["label"].capitalize()
                # Draw cyan bounding box for detected items
                cv2.rectangle(annotated_frame, (ox1, oy1), (ox2, oy2), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Item: {olabel}", (ox1, max(15, oy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        # 2. Draw Multi-Zone Security Polygons and Room Labels
        if Config.ENABLE_POSE_BEHAVIOR:
            self.pose_analyzer._reload_security_zone_if_needed()
            multi_zones = getattr(self.pose_analyzer, "multi_zones", [])
            for zone in multi_zones:
                name = zone.get("name", "Zone")
                sec_level = zone.get("security_level", "Medium")
                color_hex = zone.get("color", "#10b981")
                bgr = hex_to_bgr(color_hex)

                poly_pts = get_scaled_zone_polygon(zone, frame_w=w, frame_h=h)

                if len(poly_pts) >= 3:
                    pts = poly_pts.reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], isClosed=True, color=bgr, thickness=2, lineType=cv2.LINE_AA)
                    
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [pts], color=bgr)
                    cv2.addWeighted(overlay, 0.12, annotated_frame, 0.88, 0, annotated_frame)

                    lx, ly = poly_pts[0][0] + 8, poly_pts[0][1] + 24
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
            fw = max(1, x2 - x1)
            fh = max(1, y2 - y1)

            beh_data = {}

            # 3. Facial Behavior Analysis
            if Config.ENABLE_FACIAL_BEHAVIOR and len(landmarks) >= 5:
                landmarks_5pt = [lm["landmark"] for lm in landmarks]
                facial_telemetry = self.facial_analyzer.analyze_face((h, w), landmarks_5pt, bbox)
                beh_data.update(facial_telemetry)

            # 4. Body Pose & Motion Analysis & Carried Item Tracking
            if Config.ENABLE_POSE_BEHAVIOR:
                pose_telemetry = self.pose_analyzer.analyze_pose_and_motion(bbox)
                beh_data.update(pose_telemetry)

                # Estimate body bounding box for carrying spatial association
                body_bbox = (
                    max(0, x1 - int(fw * 0.5)),
                    y1,
                    min(w, x2 + int(fw * 0.5)),
                    min(h, y1 + int(fh * 4.5))
                )
                carried_items = self.pose_analyzer.object_tracker.associate_items_to_person(body_bbox, detected_objects)
                beh_data["carried_items"] = carried_items

                # Check Cross-Zone Movement Transition & Carried Item Transfer
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                multi_zones = getattr(self.pose_analyzer, "multi_zones", [])
                transition_evt = self.pose_analyzer.zone_transition_tracker.update_person_position(
                    person_name=identity,
                    point=centroid,
                    zones=multi_zones,
                    activity_logger=self.activity_logger,
                    frame_crop=face_crop,
                    frame_w=w,
                    frame_h=h
                )
                if transition_evt:
                    beh_data["cross_zone_event"] = transition_evt
                    # Trigger Item Transfer Event if carrying items
                    if carried_items and identity != "Unknown":
                        self.pose_analyzer.object_tracker.process_zone_transition(
                            person_name=identity,
                            from_zone_name=transition_evt.get("from_zone", "Outside"),
                            to_zone_name=transition_evt.get("to_zone", "Room"),
                            carried_items=carried_items,
                            activity_logger=self.activity_logger,
                            frame_crop=face_crop
                        )

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
        if "carried_items" in beh and beh["carried_items"]:
            items_str = ", ".join([it.capitalize() for it in beh["carried_items"]])
            lines.append(f"Carried: [{items_str}]")
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
