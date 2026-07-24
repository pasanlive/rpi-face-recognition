import os
import cv2
import uuid
import time
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import lancedb
import pyarrow as pa

from config import Config

logger = logging.getLogger(__name__)

# Define PyArrow Schema for Activity Logs in LanceDB
ActivityLogPyArrowSchema = pa.schema([
    ("log_id", pa.string()),
    ("timestamp", pa.string()),
    ("person_name", pa.string()),
    ("event_type", pa.string()),
    ("snapshot_filename", pa.string()),
    ("behavior_json", pa.string())
])

class ActivityLogger:
    """
    Manages persistent logging of detection and behavioral events to LanceDB & local snapshot files.
    Includes automated cooldown throttling to prevent duplicate logs on every frame.
    """

    def __init__(self, db_uri: str = Config.DB_URI, snapshot_dir: str = Config.SNAPSHOT_DIR, on_log_callback=None):
        self.db_uri = db_uri
        self.snapshot_dir = snapshot_dir
        self.table_name = Config.LOG_TABLE_NAME
        self.last_log_times: Dict[str, float] = {}
        self.cooldown_sec = Config.LOG_COOLDOWN_SEC
        self.on_log_callback = on_log_callback

        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.db_uri, exist_ok=True)

        self.db = lancedb.connect(self.db_uri)
        self.table = self._init_table()

    def _init_table(self):
        try:
            existing_tables = self.db.list_tables() if hasattr(self.db, "list_tables") else self.db.table_names()
            if self.table_name in existing_tables:
                logger.info(f"Opening existing activity log table '{self.table_name}'.")
                return self.db.open_table(self.table_name)
            else:
                logger.info(f"Creating new activity log table '{self.table_name}'.")
                return self.db.create_table(self.table_name, schema=ActivityLogPyArrowSchema)
        except Exception as e:
            logger.warning(f"Error initializing activity log table: {e}. Opening table directly...")
            return self.db.open_table(self.table_name)

    def log_event(
        self,
        person_name: str,
        event_type: str,
        behavior_data: Dict[str, Any],
        frame_crop: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Log an activity event if outside cooldown period.
        Saves snapshot thumbnail image and stores structured log entry in LanceDB.
        """
        if not Config.ENABLE_ACTIVITY_LOGGING:
            return None

        # Check cooldown throttle key
        throttle_key = f"{person_name}:{event_type}"
        now = time.time()
        if throttle_key in self.last_log_times and (now - self.last_log_times[throttle_key]) < self.cooldown_sec:
            return None

        self.last_log_times[throttle_key] = now

        log_id = str(uuid.uuid4())
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot_filename = ""

        # Save snapshot image if provided
        if frame_crop is not None and frame_crop.size > 0:
            try:
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_id[:8]}.jpg"
                filepath = os.path.join(self.snapshot_dir, filename)
                # Resize snapshot for optimal storage size (e.g. max 320x320)
                h, w = frame_crop.shape[:2]
                if w > 320 or h > 320:
                    scale = 320.0 / max(h, w)
                    frame_crop = cv2.resize(frame_crop, (int(w * scale), int(h * scale)))
                cv2.imwrite(filepath, frame_crop)
                snapshot_filename = filename
            except Exception as e:
                logger.error(f"Error saving activity log snapshot image: {e}")

        log_entry = {
            "log_id": log_id,
            "timestamp": timestamp_str,
            "person_name": person_name,
            "event_type": event_type,
            "snapshot_filename": snapshot_filename,
            "behavior_json": json.dumps(behavior_data)
        }

        try:
            self.table.add([log_entry])
            logger.info(f"Activity Logged [{event_type}]: Person='{person_name}' at {timestamp_str}")
            if self.on_log_callback and callable(self.on_log_callback):
                try:
                    self.on_log_callback(log_entry)
                except Exception as cb_err:
                    logger.error(f"Error in on_log_callback: {cb_err}")
            return log_entry
        except Exception as e:
            logger.error(f"Failed to write activity log entry to LanceDB: {e}")
            return None

    def get_logs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Fetch recent activity log entries sorted by timestamp descending.
        """
        try:
            if self.table.count_rows() == 0:
                return []

            df = self.table.to_pandas()
            if df.empty:
                return []

            # Sort by timestamp descending
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values(by="timestamp_dt", ascending=False).drop(columns=["timestamp_dt"])

            # Paginate
            paginated_df = df.iloc[offset : offset + limit]
            
            logs = []
            for _, row in paginated_df.iterrows():
                beh_data = {}
                try:
                    beh_data = json.loads(row["behavior_json"])
                except Exception:
                    pass

                logs.append({
                    "log_id": str(row["log_id"]),
                    "timestamp": str(row["timestamp"]),
                    "person_name": str(row["person_name"]),
                    "event_type": str(row["event_type"]),
                    "snapshot_filename": str(row["snapshot_filename"]),
                    "behavior_data": beh_data
                })
            return logs
        except Exception as e:
            logger.error(f"Error reading activity logs from LanceDB: {e}")
            return []

    def clear_logs(self) -> bool:
        """
        Clear all activity logs from LanceDB table and delete snapshot files.
        """
        try:
            # Delete snapshot image files
            if os.path.exists(self.snapshot_dir):
                for f in os.listdir(self.snapshot_dir):
                    fp = os.path.join(self.snapshot_dir, f)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass

            # Drop and recreate LanceDB table
            self.db.drop_table(self.table_name)
            self.table = self.db.create_table(self.table_name, schema=ActivityLogPyArrowSchema)
            self.last_log_times.clear()
            logger.info("Activity logs and snapshots cleared successfully.")
            return True
        except Exception as e:
            logger.error(f"Error clearing activity logs: {e}")
            return False

    def export_csv(self) -> str:
        """
        Export activity logs to CSV formatted string.
        """
        logs = self.get_logs(limit=10000)
        if not logs:
            return "log_id,timestamp,person_name,event_type,snapshot_filename,behavior_data\n"
        
        flat_logs = []
        for item in logs:
            flat_logs.append({
                "log_id": item["log_id"],
                "timestamp": item["timestamp"],
                "person_name": item["person_name"],
                "event_type": item["event_type"],
                "snapshot_filename": item["snapshot_filename"],
                "behavior_data": json.dumps(item["behavior_data"])
            })
        df = pd.DataFrame(flat_logs)
        return df.to_csv(index=False)
