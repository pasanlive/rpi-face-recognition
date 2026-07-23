import numpy as np
import logging
from typing import List, Tuple, Union
import lancedb
from config import Config
from database.schema import SecurityZoneSchema

logger = logging.getLogger(__name__)

# Constants for default video size (used when generating default polygon)
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480

_SECURITY_ZONE_TABLE = "security_zones"

def _get_db():
    return lancedb.connect(uri=Config.DB_URI)

def _ensure_table():
    db = _get_db()
    try:
        return db.open_table(_SECURITY_ZONE_TABLE)
    except Exception:
        logger.info(f"Creating security zone table '{_SECURITY_ZONE_TABLE}'")
        return db.create_table(_SECURITY_ZONE_TABLE, schema=SecurityZoneSchema, exist_ok=True)

def _flatten_polygon(polygon: List[List[int]]) -> List[int]:
    return [coord for point in polygon for coord in point]

def _unflatten_polygon(flat: List[int]) -> List[List[int]]:
    return [[flat[i], flat[i+1]] for i in range(0, len(flat), 2)]

def _normalize_cam_id(camera_index: Union[int, str]) -> int:
    if isinstance(camera_index, int):
        return camera_index
    if isinstance(camera_index, str) and str(camera_index).isdigit():
        return int(camera_index)
    return abs(hash(str(camera_index))) % (2**31 - 1)

def get_default_polygon() -> List[List[int]]:
    w, h = DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT
    # Left half of the frame
    return [[0, 0], [w // 2, 0], [w // 2, h], [0, h]]

def load_security_zone(camera_index: Union[int, str]) -> np.ndarray:
    """Load the security zone polygon for a given camera index or RTSP source.
    Returns an (N,2) int32 numpy array. If no entry exists, creates a default one.
    """
    cam_id = _normalize_cam_id(camera_index)
    table = _ensure_table()
    try:
        df = table.to_pandas()
        row = df[df["camera_index"] == cam_id]
        if not row.empty:
            flat = row.iloc[0]["polygon"]
            polygon = _unflatten_polygon(flat)
            return np.array(polygon, dtype=np.int32)
    except Exception as e:
        logger.error(f"Error loading security zone: {e}")
    # If not found, create default
    default_polygon = get_default_polygon()
    save_security_zone(camera_index, default_polygon)
    return np.array(default_polygon, dtype=np.int32)

def save_security_zone(camera_index: Union[int, str], polygon: List[List[int]]):
    """Persist a security zone polygon for a camera.
    Overwrites any existing entry.
    """
    cam_id = _normalize_cam_id(camera_index)
    table = _ensure_table()
    # Delete existing entry for this camera if present
    try:
        table.delete(f"camera_index = {cam_id}")
    except Exception as e:
        logger.debug(f"No previous security zone to delete for camera {cam_id}: {e}")
    flat = _flatten_polygon(polygon)
    record = {"camera_index": cam_id, "polygon": flat}
    table.add([record])
    logger.info(f"Saved security zone for camera index/source '{camera_index}' (id={cam_id})")

def validate_polygon(polygon: List[List[int]]) -> bool:
    """Basic validation: must have at least 3 points and be a list of [x, y] ints.
    More complex checks (self‑intersection) can be added later.
    """
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    for pt in polygon:
        if not (isinstance(pt, list) or isinstance(pt, tuple)) or len(pt) != 2:
            return False
        if not all(isinstance(coord, int) for coord in pt):
            return False
    return True
