import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from config import Config

from .security_zone import load_security_zone, load_multi_zones, find_zone_for_point, ZoneTransitionTracker
from .object_tracker import ObjectCarryingTracker

logger = logging.getLogger(__name__)

class TrackedPerson:
    def __init__(self, track_id: int, bbox: List[int], start_time: float):
        self.track_id = track_id
        self.bbox = bbox
        self.start_time = start_time
        self.last_seen = start_time

    @property
    def centroid(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @property
    def dwell_time(self) -> float:
        return time.time() - self.start_time

class PoseBehaviorAnalyzer:
    """
    Analyzes Body Posture (Standing, Sitting, Fall Detection), Loitering Dwell Time,
    Multi-Zone Security Intrusion, and Cross-Zone Movements.
    """

    def __init__(self, get_camera_index_callable=None):
        self.next_track_id = 1
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self._get_camera_index = get_camera_index_callable or (lambda: 0)
        self.zone_transition_tracker = ZoneTransitionTracker()
        self.object_tracker = ObjectCarryingTracker()
        self._load_security_zone()

    def _load_security_zone(self):
        cam_idx = self._get_camera_index()
        self.security_zone_polygon = load_security_zone(cam_idx)
        self.multi_zones = load_multi_zones(cam_idx)

    def _reload_security_zone_if_needed(self):
        """Reload polygon if the active camera index has changed.
        Called before each check to ensure the polygon matches the current camera.
        """
        self._load_security_zone()

    def _match_track(self, centroid: Tuple[int, int], max_dist: float = 80.0) -> Optional[int]:
        """
        Match centroid to existing track using Euclidean distance.
        """
        best_id = None
        best_dist = max_dist
        for track_id, person in self.tracked_persons.items():
            dist = np.hypot(centroid[0] - person.centroid[0], centroid[1] - person.centroid[1])
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def classify_posture(self, bbox: List[int]) -> str:
        """
        Classify human posture state from bounding box aspect ratio and dimensions.
        """
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = h / float(w)

        if aspect_ratio < 0.75:
            return "Lying Down / Fall Detected"
        elif aspect_ratio < 1.3:
            return "Sitting"
        else:
            return "Standing"

    def is_inside_security_zone(self, centroid: Tuple[int, int]) -> bool:
        """Check if centroid point is inside the current security zone polygon.
        The polygon is reloaded each call to reflect any changes for the active camera.
        """
        # Ensure we have the latest polygon for the current camera
        self._reload_security_zone_if_needed()
        res = cv2.pointPolygonTest(self.security_zone_polygon, (float(centroid[0]), float(centroid[1])), False)
        return res >= 0

    def analyze_pose_and_motion(self, bbox: List[int]) -> Dict[str, Any]:
        """
        Analyze posture, loitering dwell time, and security zone status for a person bounding box.
        """
        now = time.time()
        x1, y1, x2, y2 = bbox
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Match or assign track ID
        matched_id = self._match_track(centroid)
        if matched_id is None:
            track_id = self.next_track_id
            self.next_track_id += 1
            person = TrackedPerson(track_id, bbox, now)
            self.tracked_persons[track_id] = person
        else:
            track_id = matched_id
            person = self.tracked_persons[track_id]
            person.bbox = bbox
            person.last_seen = now

        # Clean up stale tracks (not seen for > 3 seconds)
        stale_ids = [tid for tid, p in self.tracked_persons.items() if (now - p.last_seen) > 3.0]
        for tid in stale_ids:
            del self.tracked_persons[tid]

        posture = self.classify_posture(bbox)
        dwell_time = round(person.dwell_time, 1)
        is_loitering = dwell_time >= Config.LOITERING_TIME_LIMIT_SEC
        is_intrusion = self.is_inside_security_zone(centroid)

        return {
            "track_id": track_id,
            "posture": posture,
            "dwell_time_sec": dwell_time,
            "is_loitering": is_loitering,
            "is_intrusion": is_intrusion
        }
