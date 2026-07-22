import os
import cv2
import time
import logging
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

from config import Config
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.alignment import align_and_crop
from database.manager import FaceDatabaseManager
from pipeline import FaceRecognitionPipeline

logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__, template_folder="templates")
    
    # Initialize Core Components lazily or at start
    logger.info("Initializing Face Recognition Pipeline for Web UI...")
    db_manager = FaceDatabaseManager()
    
    # Try initializing detector & embedder
    pipeline = None
    try:
        pipeline = FaceRecognitionPipeline(db_manager=db_manager)
    except Exception as e:
        logger.warning(f"Could not initialize Hailo hardware pipeline immediately: {e}. App starting in manager mode.")

    # Global State
    current_camera_index = Config.DEFAULT_CAMERA_SOURCE
    current_threshold = Config.SIMILARITY_THRESHOLD
    camera_cap = None

    def get_camera():
        nonlocal camera_cap
        if camera_cap is None or not camera_cap.isOpened():
            camera_cap = cv2.VideoCapture(current_camera_index)
        return camera_cap

    def generate_frames():
        nonlocal camera_cap, pipeline, current_threshold
        cap = get_camera()

        while True:
            success, frame = cap.read()
            if not success:
                # If frame read fails, generate a fallback black frame with warning
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Camera Stream Unavailable", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.5)
                continue

            # Run face recognition pipeline if initialized
            if pipeline:
                try:
                    annotated_frame, _ = pipeline.process_frame(frame, threshold=current_threshold)
                except Exception as err:
                    logger.error(f"Pipeline error: {err}")
                    annotated_frame = frame
            else:
                annotated_frame = frame

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03) # ~30 fps cap

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/identities', methods=['GET'])
    def list_identities():
        entities = db_manager.list_entities()
        return jsonify({"success": True, "identities": entities})

    @app.route('/api/identities/<entity_name>', methods=['DELETE'])
    def delete_identity(entity_name):
        success = db_manager.delete_entity(entity_name)
        return jsonify({"success": success, "entity_name": entity_name})

    @app.route('/api/enroll', methods=['POST'])
    def enroll_face():
        identity = request.form.get('identity', '').strip()
        if not identity:
            return jsonify({"success": False, "error": "Identity name is required"}), 400

        # Option A: Uploaded file
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if pipeline:
                    detected = pipeline.detector.detect(img)
                    if len(detected.results) != 1:
                        return jsonify({
                            "success": False,
                            "error": f"Image must contain exactly 1 face. Found {len(detected.results)} faces."
                        }), 400

                    res = detected.results[0]
                    landmarks = [lm["landmark"] for lm in res["landmarks"]]
                    aligned_img, _ = align_and_crop(img, landmarks, image_size=Config.INPUT_FACE_SIZE)
                    embedding = pipeline.embedder.extract_embedding(aligned_img)
                    rec_id = db_manager.add_face(embedding, identity)
                    return jsonify({"success": True, "record_id": rec_id, "identity": identity})

        # Option B: Capture current camera frame
        cap = get_camera()
        success, frame = cap.read()
        if success and pipeline:
            detected = pipeline.detector.detect(frame)
            if len(detected.results) != 1:
                return jsonify({
                    "success": False,
                    "error": f"Camera capture must contain exactly 1 face. Found {len(detected.results)} faces."
                }), 400

            res = detected.results[0]
            landmarks = [lm["landmark"] for lm in res["landmarks"]]
            aligned_img, _ = align_and_crop(frame, landmarks, image_size=Config.INPUT_FACE_SIZE)
            embedding = pipeline.embedder.extract_embedding(aligned_img)
            rec_id = db_manager.add_face(embedding, identity)
            return jsonify({"success": True, "record_id": rec_id, "identity": identity})

        return jsonify({"success": False, "error": "Enrollment failed"}), 500

    @app.route('/api/settings', methods=['GET', 'POST'])
    def settings():
        nonlocal current_threshold, current_camera_index, camera_cap
        if request.method == 'POST':
            data = request.json or {}
            if 'threshold' in data:
                current_threshold = float(data['threshold'])
            if 'camera_index' in data:
                new_idx = int(data['camera_index'])
                if new_idx != current_camera_index:
                    current_camera_index = new_idx
                    if camera_cap:
                        camera_cap.release()
                        camera_cap = None
            return jsonify({"success": True, "threshold": current_threshold, "camera_index": current_camera_index})
        
        return jsonify({
            "success": True,
            "threshold": current_threshold,
            "camera_index": current_camera_index,
            "target_device": Config.TARGET_DEVICE,
            "detection_model": Config.DEFAULT_DETECTION_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL
        })

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)
