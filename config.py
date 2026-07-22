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
    VECTOR_DIM = 512
    METRIC_TYPE = "cosine"
    SIMILARITY_THRESHOLD = float(os.getenv("FACE_APP_THRESHOLD", "0.36"))

    # Face Alignment Settings
    INPUT_FACE_SIZE = 112

    # Camera & Video Source
    DEFAULT_CAMERA_SOURCE = int(os.getenv("FACE_APP_CAM_INDEX", "0"))
