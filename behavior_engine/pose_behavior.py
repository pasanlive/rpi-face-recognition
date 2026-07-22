import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from config import Config

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
    and Virtual Security Zone Intrusion.
    """

    def __init__(self):
        self.next_track_id = 1
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        # Default Security Zone: Polygon covering center-bottom screen region
        self.security_zone_polygon = np.array([
            [100, 200], [540, 200], [540, 460], [100, 460]
        ], dtype=np.int32)

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
        """
        Check if centroid point is inside virtual security zone polygon.
        """
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
