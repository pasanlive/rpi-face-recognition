import uuid
import datetime
import numpy as np
from typing import List, Dict, Any
from lancedb.pydantic import LanceModel, Vector
from config import Config

class FaceRecognitionSchema(LanceModel):
    id: str
    vector: Vector(Config.VECTOR_DIM)
    entity_name: str
    created_at: str

    @classmethod
    def prepare_record(cls, embedding: np.ndarray, entity_name: str) -> "FaceRecognitionSchema":
        """
        Convert a 512-D numpy embedding array and entity identity name to a FaceRecognitionSchema instance.
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return cls(
            id=str(uuid.uuid4()),
            vector=np.array(embedding, dtype=np.float32),
            entity_name=entity_name,
            created_at=now_str
        )

# Schema for per-camera security zones
class SecurityZoneSchema(LanceModel):
    camera_index: int  # primary key
    polygon: List[int]  # flattened list [x1, y1, x2, y2, ...]

class MultiZoneSchema(LanceModel):
    camera_index: int
    zones_json: str  # JSON serialized list of multi-zone objects [{id, name, security_level, color, polygon}]
