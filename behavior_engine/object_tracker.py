import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Set

logger = logging.getLogger(__name__)

# COCO object classes relevant to items carried by people
TRACKED_ITEM_CLASSES = {
    "backpack", "handbag", "suitcase", "laptop", "cell phone",
    "bottle", "book", "box", "umbrella"
}

def box_overlap(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculate IoU / overlap ratio of box_a inside box_b."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    return inter_area / float(a_area) if a_area > 0 else 0.0

class ObjectCarryingTracker:
    """
    Tracks objects carried by recognized people and logs cross-zone item transfer events.
    """
    def __init__(self, item_cooldown_sec: float = 4.0):
        self.person_items: Dict[str, Set[str]] = {}  # person_name -> set of item names
        self.last_transfer_log: Dict[str, float] = {}
        self.cooldown_sec = item_cooldown_sec

    def associate_items_to_person(
        self,
        person_box: Tuple[int, int, int, int],
        detected_objects: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Given a person bounding box and a list of detected objects:
        [{"label": "laptop", "box": (x1, y1, x2, y2), "confidence": 0.85}, ...]
        Returns list of object labels carried by this person.
        """
        carried = []
        # Expand person box slightly (20%) to capture bags/laptops held adjacent to body
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        expanded_box = (
            max(0, px1 - int(pw * 0.15)),
            max(0, py1 - int(ph * 0.1)),
            px2 + int(pw * 0.15),
            py2 + int(ph * 0.1)
        )

        for obj in detected_objects:
            label = obj.get("label", "").lower()
            box = obj.get("box")
            if label in TRACKED_ITEM_CLASSES and box:
                overlap = box_overlap(box, expanded_box)
                if overlap > 0.3:  # At least 30% of object inside expanded person box
                    carried.append(label)

        return list(set(carried))

    def process_zone_transition(
        self,
        person_name: str,
        from_zone_name: str,
        to_zone_name: str,
        carried_items: List[str],
        activity_logger: Optional[Any] = None,
        frame_crop: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        When a person transitions between zones, check carried items and log item transfer events.
        """
        if not person_name or person_name == "Unknown" or not carried_items:
            return []

        events = []
        now = time.time()

        for item in carried_items:
            transfer_key = f"{person_name}:{item}:{from_zone_name}->{to_zone_name}"
            last_t = self.last_transfer_log.get(transfer_key, 0)
            if (now - last_t) >= self.cooldown_sec:
                self.last_transfer_log[transfer_key] = now

                event_type = "ITEM_TRANSFERRED_BETWEEN_ZONES"
                details = {
                    "person_name": person_name,
                    "item": item,
                    "from_zone": from_zone_name,
                    "to_zone": to_zone_name,
                    "message": f"{person_name} carried [{item.capitalize()}] from [{from_zone_name}] to [{to_zone_name}]"
                }

                logger.info(f"Item Transfer Detected: {details['message']}")

                if activity_logger:
                    activity_logger.log_event(
                        person_name=person_name,
                        event_type=event_type,
                        behavior_data=details,
                        frame_crop=frame_crop
                    )
                events.append(details)

        return events
