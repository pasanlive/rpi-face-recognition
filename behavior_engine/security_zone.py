import cv2
import json
import time
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Union, Optional
import lancedb

from config import Config
from database.schema import SecurityZoneSchema, MultiZoneSchema

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_WIDTH = Config.CAMERA_WIDTH
DEFAULT_VIDEO_HEIGHT = Config.CAMERA_HEIGHT

_SECURITY_ZONE_TABLE = "security_zones"
_MULTI_ZONE_TABLE = "multi_security_zones"

def _get_db():
    return lancedb.connect(uri=Config.DB_URI)

def _ensure_multi_table():
    db = _get_db()
    try:
        existing = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
        if _MULTI_ZONE_TABLE in existing:
            return db.open_table(_MULTI_ZONE_TABLE)
        return db.create_table(_MULTI_ZONE_TABLE, schema=MultiZoneSchema, exist_ok=True)
    except Exception as e:
        logger.warning(f"Error opening multi zone table: {e}")
        return db.open_table(_MULTI_ZONE_TABLE)

def _normalize_cam_id(camera_index: Union[int, str]) -> int:
    if isinstance(camera_index, int):
        return camera_index
    if isinstance(camera_index, str) and str(camera_index).isdigit():
        return int(camera_index)
    return abs(hash(str(camera_index))) % (2**31 - 1)

def get_default_multi_zones() -> List[Dict[str, Any]]:
    w, h = DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT
    return [
        {
            "id": "zone_room1",
            "name": "Room 1",
            "security_level": "Medium",
            "color": "#10b981",
            "polygon": [[0, 0], [w // 2, 0], [w // 2, h], [0, h]]
        },
        {
            "id": "zone_room2",
            "name": "Room 2",
            "security_level": "High",
            "color": "#ef4444",
            "polygon": [[w // 2, 0], [w, 0], [w, h], [w // 2, h]]
        }
    ]

def load_multi_zones(camera_index: Union[int, str]) -> List[Dict[str, Any]]:
    """Load all configured security zones for a camera."""
    cam_id = _normalize_cam_id(camera_index)
    table = _ensure_multi_table()
    try:
        df = table.to_pandas()
        row = df[df["camera_index"] == cam_id]
        if not row.empty:
            zones_str = row.iloc[0]["zones_json"]
            zones = json.loads(zones_str)
            if isinstance(zones, list) and len(zones) > 0:
                return zones
    except Exception as e:
        logger.error(f"Error loading multi-zones: {e}")
    
    # Return defaults & save
    default_zones = get_default_multi_zones()
    save_multi_zones(camera_index, default_zones)
    return default_zones

def save_multi_zones(camera_index: Union[int, str], zones: List[Dict[str, Any]]):
    """Save multiple named security zones for a camera."""
    cam_id = _normalize_cam_id(camera_index)
    table = _ensure_multi_table()
    try:
        table.delete(f"camera_index = {cam_id}")
    except Exception:
        pass
    
    zones_json = json.dumps(zones)
    record = {"camera_index": cam_id, "zones_json": zones_json}
    table.add([record])
    logger.info(f"Saved {len(zones)} security zones for camera '{camera_index}' (id={cam_id}).")

def load_security_zone(camera_index: Union[int, str]) -> np.ndarray:
    """Backwards compatible: return primary polygon as numpy array."""
    zones = load_multi_zones(camera_index)
    if zones and "polygon" in zones[0]:
        return np.array(zones[0]["polygon"], dtype=np.int32)
    w, h = DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT
    return np.array([[0, 0], [w // 2, 0], [w // 2, h], [0, h]], dtype=np.int32)

def save_security_zone(camera_index: Union[int, str], polygon: List[List[int]]):
    """Backwards compatible: update primary polygon."""
    zones = load_multi_zones(camera_index)
    if not zones:
        zones = get_default_multi_zones()
    zones[0]["polygon"] = polygon
    save_multi_zones(camera_index, zones)

def validate_polygon(polygon: List[List[int]]) -> bool:
    """Basic validation: must have at least 3 points and be a list of [x, y] ints."""
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    for pt in polygon:
        if not (isinstance(pt, list) or isinstance(pt, tuple)) or len(pt) != 2:
            return False
        if not all(isinstance(coord, int) for coord in pt):
            return False
    return True

def find_zone_for_point(point: Tuple[int, int], zones: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Check which security zone polygon contains the given (x, y) point."""
    x, y = point
    for zone in zones:
        poly = zone.get("polygon", [])
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32)
            dist = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
            if dist >= 0:
                return zone
    return None

class ZoneTransitionTracker:
    """Tracks state transitions of people moving between named zones."""
    def __init__(self, transition_cooldown_sec: float = 3.0):
        self.person_current_zone: Dict[str, str] = {}  # person_name -> zone_id
        self.last_transition_time: Dict[str, float] = {}
        self.cooldown_sec = transition_cooldown_sec

    def update_person_position(
        self,
        person_name: str,
        point: Tuple[int, int],
        zones: List[Dict[str, Any]],
        activity_logger: Optional[Any] = None,
        frame_crop: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check person's point against active zones.
        If person moves from Zone A to Zone B, log CROSS_ZONE_TRANSITION event.
        """
        if not person_name or person_name == "Unknown":
            return None

        current_zone = find_zone_for_point(point, zones)
        curr_zone_id = current_zone["id"] if current_zone else "OUTSIDE"
        curr_zone_name = current_zone["name"] if current_zone else "Unrestricted Area"

        prev_zone_id = self.person_current_zone.get(person_name, "OUTSIDE")

        if prev_zone_id != curr_zone_id:
            now = time.time()
            last_t = self.last_transition_time.get(person_name, 0)
            if (now - last_t) >= self.cooldown_sec:
                self.last_transition_time[person_name] = now
                self.person_current_zone[person_name] = curr_zone_id

                prev_zone_name = "Outside"
                for z in zones:
                    if z["id"] == prev_zone_id:
                        prev_zone_name = z["name"]

                sec_level = current_zone.get("security_level", "Medium") if current_zone else "Low"

                event_type = "CROSS_ZONE_TRANSITION"
                details = {
                    "from_zone": prev_zone_name,
                    "to_zone": curr_zone_name,
                    "security_level": sec_level,
                    "message": f"{person_name} walked from [{prev_zone_name}] to [{curr_zone_name}] (Security: {sec_level})"
                }

                logger.info(f"Transition Detected: {person_name} moved from '{prev_zone_name}' to '{curr_zone_name}'")

                if activity_logger:
                    activity_logger.log_event(
                        person_name=person_name,
                        event_type=event_type,
                        behavior_data=details,
                        frame_crop=frame_crop
                    )
                return details

        self.person_current_zone[person_name] = curr_zone_id
        return None
