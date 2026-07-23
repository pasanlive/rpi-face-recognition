import os

class Config:
    # Hardware target: 'hailo8' for Raspberry Pi 5 AI Hat+ 26 TOPS, or 'cpu'
    TARGET_DEVICE = os.getenv("FACE_APP_TARGET", "hailo8")
    
    # Model Paths
    MODEL_DIR = os.getenv("FACE_APP_MODEL_DIR", "./models")
    DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, "scrfd_10g_hailo8.hef")
    EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "arcface_mobilefacenet_hailo8.hef")

    # Vector Database Settings (LanceDB)
    DB_URI = os.getenv("FACE_APP_DB_URI", "./face_database")
    TABLE_NAME = "face"
    LOG_TABLE_NAME = "activity_log"
    VECTOR_DIM = 512
    METRIC_TYPE = "cosine"
    SIMILARITY_THRESHOLD = float(os.getenv("FACE_APP_THRESHOLD", "0.36"))

    # Face Alignment Settings
    INPUT_FACE_SIZE = 112

    # Camera & Video Source (numerical index like 0, 1 or RTSP URL like "rtsp://user:pass@192.168.1.100:554/stream")
    _cam_env = os.getenv("FACE_APP_CAM_INDEX", "0")
    DEFAULT_CAMERA_SOURCE = int(_cam_env) if _cam_env.isdigit() else _cam_env
    CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
    CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))

    # Behavioral Detection Settings
    ENABLE_FACIAL_BEHAVIOR = os.getenv("ENABLE_FACIAL_BEHAVIOR", "true").lower() == "true"
    ENABLE_POSE_BEHAVIOR = os.getenv("ENABLE_POSE_BEHAVIOR", "true").lower() == "true"
    ENABLE_ACTIVITY_LOGGING = os.getenv("ENABLE_ACTIVITY_LOGGING", "true").lower() == "true"

    # Behavioral Thresholds
    DROWSINESS_EAR_THRESHOLD = float(os.getenv("DROWSINESS_EAR_THRESHOLD", "0.22"))
    DROWSINESS_TIME_SEC = float(os.getenv("DROWSINESS_TIME_SEC", "1.5"))
    YAWN_MAR_THRESHOLD = float(os.getenv("YAWN_MAR_THRESHOLD", "0.55"))
    LOITERING_TIME_LIMIT_SEC = float(os.getenv("LOITERING_TIME_LIMIT_SEC", "10.0"))

    # Activity Logging & Snapshots
    SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "./activity_logs/snapshots")
    LOG_COOLDOWN_SEC = float(os.getenv("LOG_COOLDOWN_SEC", "5.0"))
