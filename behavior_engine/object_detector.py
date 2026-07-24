import os
import cv2
import numpy as np
import logging
import urllib.request
from typing import List, Dict, Any, Tuple
from config import Config
from hailo_engine import HailoInferenceEngine, HAILO_PLATFORM_AVAILABLE

logger = logging.getLogger(__name__)

# Standard COCO 80 Class Names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Tracked item labels relevant for security carrying detection
TARGET_ITEM_LABELS = {
    "backpack", "handbag", "suitcase", "laptop", "cell phone",
    "bottle", "book", "umbrella", "box", "chair", "tv"
}

def ensure_coco_model(model_dir: str = "./models") -> str:
    """
    Ensure OpenCV Zoo NanoDet ONNX object detection model is available locally for CPU fallback mode.
    Auto-downloads if missing.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "object_detection_nanodet_2022nov.onnx")

    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        return model_path

    url = "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
    try:
        logger.info(f"Downloading high-speed OpenCV NanoDet ONNX COCO model to '{model_path}'...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(model_path, 'wb') as f:
            f.write(resp.read())
        logger.info("NanoDet ONNX COCO model downloaded successfully.")
        return model_path
    except Exception as e:
        logger.warning(f"Could not download NanoDet ONNX model: {e}")
        return ""

def ensure_hailo_hef_model(hef_path: str = "./models/yolov8n.hef") -> str:
    """
    Ensure Hailo-8 YOLOv8n HEF model is available locally.
    Auto-downloads from Hailo Model Zoo S3 bucket if missing.
    """
    if os.path.exists(hef_path) and os.path.getsize(hef_path) > 1000000:
        return hef_path

    os.makedirs(os.path.dirname(hef_path), exist_ok=True)
    url = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/yolov8n.hef"
    try:
        logger.info(f"Downloading pre-compiled Hailo-8 YOLOv8n HEF model to '{hef_path}'...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(hef_path, 'wb') as f:
            f.write(resp.read())
        logger.info("Hailo-8 YOLOv8n HEF model downloaded successfully.")
        return hef_path
    except Exception as e:
        logger.warning(f"Could not download Hailo HEF model: {e}")
        return ""

class COCOObjectDetector:
    """
    High-Performance COCO Object & Item Carrying Detector.
    Supports native Hailo-8 NPU HEF hardware acceleration (26 TOPS RPi AI Hat+),
    with seamless fallback to OpenCV NanoDet ONNX / CPU.
    """

    def __init__(self, model_dir: str = Config.MODEL_DIR, hef_path: str = Config.OBJECT_DETECTION_HEF_PATH):
        self.model_dir = model_dir
        self.hef_path = ensure_hailo_hef_model(hef_path)
        self.hailo_engine = None
        self.net = None
        self.model_loaded = False

        # 1. Try Hailo-8 NPU HEF model first
        if HAILO_PLATFORM_AVAILABLE and os.path.exists(self.hef_path):
            try:
                logger.info(f"Initializing Hailo-8 NPU Object Detector from HEF '{self.hef_path}'...")
                self.hailo_engine = HailoInferenceEngine(self.hef_path)
                if self.hailo_engine.is_ready:
                    logger.info("⚡ Hailo-8 NPU Object Detector active. Running object detection on 26 TOPS AI Hat+ hardware.")
                    return
            except Exception as e:
                logger.warning(f"Could not load Hailo HEF model: {e}")

        # 2. Fallback to OpenCV NanoDet ONNX CPU model ONLY if Hailo is absent
        self.onnx_path = ensure_coco_model(self.model_dir)
        self._init_onnx_model()

    def _init_onnx_model(self):
        if self.onnx_path and os.path.exists(self.onnx_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.onnx_path)
                self.model_loaded = True
                logger.info(f"Initialized OpenCV NanoDet ONNX COCO Object Detector from '{self.onnx_path}'.")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX COCO model: {e}")

        logger.info("COCO Detector running with adaptive salient item detection fallback.")

    def detect_objects(self, frame: np.ndarray, confidence_threshold: float = 0.30) -> List[Dict[str, Any]]:
        """
        Detect items in frame.
        Returns list of objects: [{"label": "backpack", "box": (x1, y1, x2, y2), "confidence": 0.85}, ...]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detected = []

        # 1. Execute on Hailo-8 NPU if ready (100% NPU Hardware Execution)
        if self.hailo_engine and self.hailo_engine.is_ready:
            try:
                input_blob = cv2.resize(frame, (640, 640))
                input_blob = np.expand_dims(input_blob, axis=0)
                results = self.hailo_engine.infer(input_blob)
                if results:
                    output_tensor = list(results.values())[0]
                    if len(output_tensor.shape) >= 2:
                        for item in output_tensor[0]:
                            if len(item) >= 6:
                                y1_n, x1_n, y2_n, x2_n, score, cls_id = item[:6]
                                if score >= confidence_threshold:
                                    lbl = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else "object"
                                    if lbl in TARGET_ITEM_LABELS or lbl != "person":
                                        bx1 = int(x1_n * w)
                                        by1 = int(y1_n * h)
                                        bx2 = int(x2_n * w)
                                        by2 = int(y2_n * h)
                                        detected.append({
                                            "label": lbl,
                                            "box": (bx1, by1, bx2, by2),
                                            "confidence": round(float(score), 2)
                                        })
                return detected
            except Exception as e:
                logger.error(f"Hailo-8 NPU object detection error: {e}")

        # 2. Fallback to OpenCV NanoDet ONNX CPU Engine
        if self.model_loaded and self.net is not None:
            try:
                blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), (103.53, 116.28, 123.675), swapRB=True)
                self.net.setInput(blob)
                outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

                if len(outs) == 6:
                    # Dynamically match class score tensors and distance regression tensors by shape
                    cls_tensors = [o[0] for o in outs if o.ndim == 3 and o.shape[2] == 80]
                    dis_tensors = [o[0] for o in outs if o.ndim == 3 and o.shape[2] == 32]

                    # Sort tensors by anchor count descending (2704 -> 676 -> 169)
                    cls_tensors.sort(key=lambda t: t.shape[0], reverse=True)
                    dis_tensors.sort(key=lambda t: t.shape[0], reverse=True)

                    strides = [8, 16, 32]
                    proj = np.arange(8, dtype=np.float32)

                    boxes = []
                    confidences = []
                    class_ids = []

                    scale_x = w / 416.0
                    scale_y = h / 416.0

                    for i, (cls_pred, dis_pred) in enumerate(zip(cls_tensors, dis_tensors)):
                        stride = strides[i] if i < len(strides) else 32
                        n_anchors = cls_pred.shape[0]

                        feat_w = int(np.sqrt(n_anchors))
                        feat_h = n_anchors // feat_w if feat_w > 0 else 1

                        grid_y, grid_x = np.mgrid[0:feat_h, 0:feat_w]
                        grid_x = (grid_x.flatten() + 0.5) * stride
                        grid_y = (grid_y.flatten() + 0.5) * stride

                        exp_cls = np.exp(cls_pred - np.max(cls_pred, axis=1, keepdims=True))
                        cls_scores = exp_cls / np.sum(exp_cls, axis=1, keepdims=True)

                        max_cls = np.argmax(cls_scores, axis=1)
                        max_scores = np.max(cls_scores, axis=1)

                        mask = max_scores >= confidence_threshold
                        for idx in np.where(mask)[0]:
                            if idx >= len(dis_pred) or idx >= len(grid_x):
                                continue
                            score = float(max_scores[idx])
                            cls_id = int(max_cls[idx])
                            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "object"

                            if label in TARGET_ITEM_LABELS or label != "person":
                                cx, cy = grid_x[idx], grid_y[idx]

                                dis = dis_pred[idx].reshape(4, 8)
                                exp_dis = np.exp(dis - np.max(dis, axis=1, keepdims=True))
                                prob_dis = exp_dis / np.sum(exp_dis, axis=1, keepdims=True)
                                dist = np.dot(prob_dis, proj) * stride

                                x1 = int((cx - dist[0]) * scale_x)
                                y1 = int((cy - dist[1]) * scale_y)
                                x2 = int((cx + dist[2]) * scale_x)
                                y2 = int((cy + dist[3]) * scale_y)

                                bx = max(0, x1)
                                by = max(0, y1)
                                bw = min(w - bx, x2 - x1)
                                bh = min(h - by, y2 - y1)

                                if bw > 15 and bh > 15:
                                    boxes.append([bx, by, bw, bh])
                                    confidences.append(score)
                                    class_ids.append(cls_id)

                    if len(boxes) > 0:
                        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.45)
                        if len(indices) > 0:
                            flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                            for idx in flat_indices:
                                bx, by, bw, bh = boxes[idx]
                                lbl = COCO_CLASSES[class_ids[idx]]
                                detected.append({
                                    "label": lbl,
                                    "box": (bx, by, bx + bw, by + bh),
                                    "confidence": round(confidences[idx], 2)
                                })
                            return detected

            except Exception as e:
                logger.error(f"Error running NanoDet ONNX Object Detector: {e}", exc_info=True)

        # Salient edge/contour region detector fallback for carried items
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2500 < area < (w * h * 0.25):
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect = bw / float(bh)
                    if 0.3 <= aspect <= 3.0:
                        detected.append({
                            "label": "object",
                            "box": (bx, by, bx + bw, by + bh),
                            "confidence": 0.50
                        })
        except Exception:
            pass

        return detected
