import os
import sys
import cv2
import time
import logging
import numpy as np
# Ensure project root is in PYTHONPATH for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from behavior_engine import security_zone

from config import Config
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.alignment import align_and_crop
from database.manager import FaceDatabaseManager
from pipeline import FaceRecognitionPipeline
from activity_logger.logger import ActivityLogger
from camera_wrapper import CameraWrapper
from system_metrics import get_system_metrics

try:
    from flask_socketio import SocketIO, emit
    socketio = SocketIO()
    HAS_SOCKETIO = True
except ImportError:
    socketio = None
    HAS_SOCKETIO = False

logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__, template_folder="templates")
    if HAS_SOCKETIO and socketio is not None:
        socketio.init_app(
            app,
            cors_allowed_origins="*",
            async_mode="threading"
        )
        app.socketio = socketio
    
    # Initialize Core Components
    logger.info("Initializing Face & Behavioral Recognition Pipeline for Web UI...")
    db_manager = FaceDatabaseManager()
    activity_logger = ActivityLogger()

    # Instant WebSocket Activity Log Event Emitter
    if socketio:
        def notify_websocket_log_event(log_entry):
            try:
                beh_data = {}
                b_json = log_entry.get("behavior_json")
                if isinstance(b_json, str) and b_json:
                    try:
                        beh_data = json.loads(b_json)
                    except Exception:
                        pass
                else:
                    beh_data = log_entry.get("behavior_data", {})

                payload = {
                    "log_id": str(log_entry.get("log_id", "")),
                    "timestamp": str(log_entry.get("timestamp", "")),
                    "person_name": str(log_entry.get("person_name", "")),
                    "event_type": str(log_entry.get("event_type", "")),
                    "snapshot_filename": str(log_entry.get("snapshot_filename", "")),
                    "behavior_data": beh_data
                }
                socketio.emit("new_activity_event", payload)
            except Exception as err:
                logger.error(f"Error broadcasting WebSocket activity log: {err}")

        activity_logger.on_log_callback = notify_websocket_log_event

        # Background thread pushing system metrics every 1.5s
        import threading
        def _stats_emitter_loop():
            while True:
                time.sleep(1.5)
                try:
                    metrics = get_system_metrics()
                    metrics["target_device"] = Config.TARGET_DEVICE
                    socketio.emit("system_stats", metrics)
                except Exception as e:
                    logger.debug(f"Error emitting stats: {e}")

        stats_thread = threading.Thread(target=_stats_emitter_loop, daemon=True)
        stats_thread.start()
    
    # Global State
    current_camera_index = Config.DEFAULT_CAMERA_SOURCE
    current_threshold = Config.SIMILARITY_THRESHOLD
    camera_cap = None
    
    pipeline = None
    # Define a getter for the current camera index to be used by the pipeline and behavior analyzer
    def get_current_camera_index():
        return current_camera_index
    try:
        pipeline = FaceRecognitionPipeline(
            db_manager=db_manager,
            activity_logger=activity_logger,
            get_camera_index_callable=get_current_camera_index
        )
    except Exception as e:
        logger.warning(f"Could not initialize pipeline immediately: {e}. App starting in manager mode.")

    def get_camera():
        nonlocal camera_cap
        if camera_cap is None or not camera_cap.is_opened():
            logger.info(f"Opening camera source '{current_camera_index}'...")
            camera_cap = CameraWrapper(current_camera_index)
        return camera_cap

    def generate_frames():
        nonlocal camera_cap, pipeline, current_threshold

        while True:
            try:
                cap = get_camera()
                if not cap or not cap.is_opened():
                    success = False
                    frame = None
                else:
                    success, frame = cap.read()
            except (cv2.error, Exception) as e:
                logger.error(f"Error reading camera frame: {e}")
                if camera_cap:
                    camera_cap.release()
                    camera_cap = None
                success = False
                frame = None

            if not success or frame is None or frame.size == 0:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Camera Stream Unavailable", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.5)
                continue

            # Run face recognition & behavioral pipeline
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

    @app.route('/stream')
    def stream_page():
        return render_template('stream.html')

    @app.route('/logs')
    def logs_page():
        return render_template('logs.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/identities', methods=['GET'])
    def list_identities():
        try:
            entities = db_manager.list_entities()
            return jsonify({"success": True, "identities": entities})
        except Exception as e:
            logger.error(f"Error in /api/identities: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/identities/<entity_name>', methods=['DELETE'])
    def delete_identity(entity_name):
        try:
            success = db_manager.delete_entity(entity_name)
            return jsonify({"success": success, "entity_name": entity_name})
        except Exception as e:
            logger.error(f"Error in DELETE /api/identities: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/enroll', methods=['POST'])
    def enroll_face():
        try:
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
            if cap and cap.is_opened():
                success, frame = cap.read()
                if success and frame is not None and pipeline:
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

            return jsonify({"success": False, "error": "Enrollment failed. Camera frame unavailable or pipeline not initialized."}), 400
        except Exception as e:
            logger.error(f"Error in /api/enroll: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/activity_logs', methods=['GET', 'DELETE'])
    def handle_activity_logs():
        try:
            if request.method == 'DELETE':
                success = activity_logger.clear_logs()
                return jsonify({"success": success})

            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))
            logs = activity_logger.get_logs(limit=limit, offset=offset)
            return jsonify({"success": True, "logs": logs})
        except Exception as e:
            logger.error(f"Error in /api/activity_logs: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/activity_logs/export', methods=['GET'])
    def export_activity_logs():
        try:
            csv_data = activity_logger.export_csv()
            return Response(
                csv_data,
                mimetype="text/csv",
                headers={"Content-disposition": "attachment; filename=activity_logs.csv"}
            )
        except Exception as e:
            logger.error(f"Error exporting activity logs: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/activity_logs/snapshots/<filename>')
    def serve_snapshot(filename):
        try:
            return send_from_directory(os.path.abspath(Config.SNAPSHOT_DIR), filename)
        except Exception as e:
            logger.error(f"Error serving snapshot file '{filename}': {e}")
            return jsonify({"error": "File not found"}), 404



    @app.route('/api/system_stats')
    def system_stats():
        try:
            metrics = get_system_metrics()
            metrics["success"] = True
            metrics["target_device"] = Config.TARGET_DEVICE
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"Error fetching system stats: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/camera_snapshot')
    def camera_snapshot():
        nonlocal camera_cap
        try:
            cap = get_camera()
            if cap and cap.is_opened():
                success, frame = cap.read()
                if success and frame is not None and frame.size > 0:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    return Response(buffer.tobytes(), mimetype='image/jpeg')
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
        blank = np.zeros((Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank, "Live Snapshot Unavailable", (int(Config.CAMERA_WIDTH * 0.25), int(Config.CAMERA_HEIGHT * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', blank)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    # Multi-Zone Security API
    @app.route('/api/security_zone', methods=['GET', 'POST'], strict_slashes=False)
    @app.route('/api/security_zone/<path:camera_id>', methods=['GET', 'POST'], strict_slashes=False)
    def handle_security_zone(camera_id=None):
        cam = camera_id if camera_id is not None else current_camera_index
        if request.method == 'GET':
            try:
                zones = security_zone.load_multi_zones(cam)
                primary_poly = zones[0]["polygon"] if zones else []
                return jsonify({
                    "success": True,
                    "zones": zones,
                    "polygon": primary_poly,
                    "points": primary_poly,
                    "camera_id": str(cam),
                    "camera_width": Config.CAMERA_WIDTH,
                    "camera_height": Config.CAMERA_HEIGHT
                })
            except Exception as e:
                logger.error(f"Error loading security zone: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500
        elif request.method == 'POST':
            try:
                data = request.json or {}
                if "zones" in data and isinstance(data["zones"], list):
                    zones = data["zones"]
                    security_zone.save_multi_zones(cam, zones)
                    return jsonify({"success": True, "message": "Multi-zones saved successfully", "zones": zones})
                
                polygon = data.get('polygon', data.get('points'))
                if not polygon:
                    return jsonify({"success": False, "error": "Polygon or zones data missing"}), 400
                if not security_zone.validate_polygon(polygon):
                    return jsonify({"success": False, "error": "Invalid polygon format"}), 400
                security_zone.save_security_zone(cam, polygon)
                return jsonify({"success": True, "message": "Security zone saved", "camera_id": str(cam)})
            except Exception as e:
                logger.error(f"Error saving security zone: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

    # Settings API (Supports local indices e.g. 0, 1 or RTSP URLs)
    @app.route('/api/settings', methods=['GET', 'POST'])
    def settings():
        nonlocal current_threshold, current_camera_index, camera_cap
        try:
            if request.method == 'POST':
                data = request.json or {}
                if 'threshold' in data:
                    current_threshold = float(data['threshold'])
                if 'enable_face_recognition' in data:
                    Config.ENABLE_FACE_RECOGNITION = bool(data['enable_face_recognition'])
                if 'enable_behavioral_detection' in data:
                    b_val = bool(data['enable_behavioral_detection'])
                    Config.ENABLE_POSE_BEHAVIOR = b_val
                    Config.ENABLE_FACIAL_BEHAVIOR = b_val
                if 'enable_facial_behavior' in data:
                    Config.ENABLE_FACIAL_BEHAVIOR = bool(data['enable_facial_behavior'])
                if 'enable_pose_behavior' in data:
                    Config.ENABLE_POSE_BEHAVIOR = bool(data['enable_pose_behavior'])
                if 'enable_object_detection' in data:
                    Config.ENABLE_OBJECT_DETECTION = bool(data['enable_object_detection'])

                if 'camera_source' in data or 'camera_index' in data:
                    raw_source = data.get('camera_source', data.get('camera_index'))
                    if isinstance(raw_source, str) and raw_source.isdigit():
                        new_source = int(raw_source)
                    elif isinstance(raw_source, (int, str)):
                        new_source = raw_source
                    else:
                        new_source = current_camera_index

                    if new_source != current_camera_index:
                        logger.info(f"Switching camera source from '{current_camera_index}' to '{new_source}'...")
                        current_camera_index = new_source
                        if camera_cap:
                            camera_cap.release()
                            camera_cap = None

                return jsonify({
                    "success": True,
                    "threshold": current_threshold,
                    "camera_index": current_camera_index,
                    "camera_source": current_camera_index,
                    "enable_face_recognition": Config.ENABLE_FACE_RECOGNITION,
                    "enable_behavioral_detection": Config.ENABLE_POSE_BEHAVIOR,
                    "enable_facial_behavior": Config.ENABLE_FACIAL_BEHAVIOR,
                    "enable_pose_behavior": Config.ENABLE_POSE_BEHAVIOR,
                    "enable_object_detection": Config.ENABLE_OBJECT_DETECTION
                })

            return jsonify({
                "success": True,
                "threshold": current_threshold,
                "camera_index": current_camera_index,
                "camera_source": current_camera_index,
                "camera_width": Config.CAMERA_WIDTH,
                "camera_height": Config.CAMERA_HEIGHT,
                "target_device": Config.TARGET_DEVICE,
                "enable_face_recognition": Config.ENABLE_FACE_RECOGNITION,
                "enable_behavioral_detection": Config.ENABLE_POSE_BEHAVIOR,
                "enable_facial_behavior": Config.ENABLE_FACIAL_BEHAVIOR,
                "enable_pose_behavior": Config.ENABLE_POSE_BEHAVIOR,
                "enable_object_detection": Config.ENABLE_OBJECT_DETECTION,
                "enable_activity_logging": Config.ENABLE_ACTIVITY_LOGGING
            })
        except Exception as e:
            logger.error(f"Error in /api/settings: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
