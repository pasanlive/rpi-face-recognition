import os

class Config:
    # Hardware target: 'hailo8' for 26 TOPS AI Hat+, 'hailo8l' for 13 TOPS AI Hat, 'cpu' or 'cloud'
    TARGET_DEVICE = os.getenv("FACE_APP_TARGET", "hailo8")
    INFERENCE_HOST = os.getenv("FACE_APP_HOST", "@local") # Use @cloud for cloud fallback
    # Auto-detect local model directories to run offline without cloud token requirement
    _candidate_zoos = [
        os.getenv("FACE_APP_ZOO_URL", ""),
        "./models",
        "../models",
        "../hailo_examples/models",
        os.path.expanduser("~/Documents/rpi-face-recognition/models"),
        os.path.expanduser("~/Documents/hailo_examples/models"),
        os.path.expanduser("~/hailo_examples/models"),
        os.path.expanduser("~/models")
    ]
    
    ZOO_URL = "degirum/models_hailort"
    for _candidate in _candidate_zoos:
        if _candidate and not _candidate.startswith("degirum") and os.path.exists(_candidate):
            ZOO_URL = _candidate
            break

    TOKEN = os.getenv("FACE_APP_TOKEN", os.getenv("DEGIRUM_CLOUD_TOKEN", ""))

    # Model Selection (defaults tailored for Hailo-8 26 TOPS)
    DETECTION_MODELS = {
        "scrfd_10g": f"scrfd_10g--640x640_quant_hailort_{TARGET_DEVICE}_1",
        "scrfd_2.5g": f"scrfd_2.5g--640x640_quant_hailort_{TARGET_DEVICE}_1",
        "scrfd_500m": f"scrfd_500m--640x640_quant_hailort_{TARGET_DEVICE}_1",
        "retinaface": f"retinaface_mobilenet--736x1280_quant_hailort_{TARGET_DEVICE}_1",
        "yolov8n": f"yolov8n_relu6_widerface_kpts--640x640_quant_hailort_{TARGET_DEVICE}_1"
    }
    
    DEFAULT_DETECTION_MODEL = DETECTION_MODELS["scrfd_10g"]
    EMBEDDING_MODEL = f"arcface_mobilefacenet--112x112_quant_hailort_{TARGET_DEVICE}_1"

    # Database Settings
    DB_URI = os.getenv("FACE_APP_DB_URI", "./face_database")
    TABLE_NAME = "face"
    VECTOR_DIM = 512
    METRIC_TYPE = "cosine"
    SIMILARITY_THRESHOLD = float(os.getenv("FACE_APP_THRESHOLD", "0.35"))

    # Alignment Settings
    INPUT_FACE_SIZE = 112

    # Camera & Video Stream
    DEFAULT_CAMERA_SOURCE = int(os.getenv("FACE_APP_CAM_INDEX", "0"))
