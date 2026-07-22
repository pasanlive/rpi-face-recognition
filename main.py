import argparse
import sys
import os
import cv2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from config import Config
from database.manager import FaceDatabaseManager
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from pipeline import FaceRecognitionPipeline

from camera_wrapper import CameraWrapper

def cmd_run(args):
    """
    Run real-time face recognition stream.
    """
    source = args.source
    if source.isdigit():
        source = int(source)

    logger.info(f"Starting Face Recognition Pipeline on video source '{source}'...")
    db_manager = FaceDatabaseManager()
    pipeline = FaceRecognitionPipeline(db_manager=db_manager)

    cap = CameraWrapper(source)
    if not cap.is_opened():
        logger.error(f"Cannot open video source: {source}")
        return

    window_name = "Raspberry Pi 5 - Hailo-8 Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.info("End of video stream or failed to read frame.")
                break

            annotated_frame, metadata = pipeline.process_frame(frame, threshold=args.threshold)
            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User quit stream.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def cmd_enroll(args):
    """
    Enroll a single image or an entire folder of images into the LanceDB vector database.
    """
    input_path = args.path
    db_manager = FaceDatabaseManager()
    detector = FaceDetector()
    embedder = FaceEmbedder()

    if os.path.isdir(input_path):
        logger.info(f"Enrolling images from directory '{input_path}'...")
        stats = db_manager.populate_from_directory(input_path, detector, embedder)
        logger.info(f"Enrollment finished. Summary: {stats}")
    elif os.path.isfile(input_path):
        identity = args.name
        if not identity:
            # Fallback to filename prefix if name not explicitly passed
            identity = os.path.splitext(os.path.basename(input_path))[0].split('_')[0].capitalize()
        
        logger.info(f"Enrolling image '{input_path}' for identity '{identity}'...")
        img = cv2.imread(input_path)
        if img is None:
            logger.error(f"Could not read image file: {input_path}")
            return

        detected = detector.detect(img)
        if len(detected.results) != 1:
            logger.error(f"Image must contain exactly 1 face. Found {len(detected.results)} faces.")
            return

        from face_engine.alignment import align_and_crop
        res = detected.results[0]
        landmarks = [lm["landmark"] for lm in res["landmarks"]]
        aligned_img, _ = align_and_crop(img, landmarks, image_size=Config.INPUT_FACE_SIZE)
        embedding = embedder.extract_embedding(aligned_img)

        rec_id = db_manager.add_face(embedding, identity)
        logger.info(f"Enrolled successfully! Identity: '{identity}' (Record ID: {rec_id})")

def cmd_list(args):
    """
    List enrolled identities in LanceDB.
    """
    db_manager = FaceDatabaseManager()
    entities = db_manager.list_entities()
    print("\n--- Enrolled Identities in Vector Database ---")
    if not entities:
        print("No identities currently enrolled.")
    else:
        for idx, item in enumerate(entities, 1):
            print(f"{idx}. Name: {item['entity_name']:<20} Vectors: {item['count']}")
    print("----------------------------------------------\n")

def cmd_delete(args):
    """
    Delete identity from LanceDB.
    """
    db_manager = FaceDatabaseManager()
    name = args.name
    success = db_manager.delete_entity(name)
    if success:
        logger.info(f"Deleted records for identity '{name}'.")
    else:
        logger.error(f"Failed to delete records for identity '{name}'.")

def cmd_web(args):
    """
    Launch Flask Web UI Dashboard.
    """
    from webui.app import create_app
    app = create_app()
    logger.info(f"Launching Web UI Dashboard at http://{args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=False)

def cmd_set_token(args):
    """
    Save DeGirum token locally to user config directory.
    """
    token_str = args.token_value.strip()
    if not token_str:
        logger.error("Token value cannot be empty.")
        return

    os.environ["DEGIRUM_CLOUD_TOKEN"] = token_str
    try:
        import degirum as dg
        if hasattr(dg, "set_token"):
            dg.set_token(token_str)
        logger.info(f"Successfully registered DeGirum token on system ({token_str[:4]}...{token_str[-4:] if len(token_str) > 8 else ''}).")
    except Exception as e:
        logger.error(f"Failed to set degirum token: {e}")

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 (AI Hat+ 26 TOPS) Face Recognition App")
    parser.add_argument("--token", default=Config.TOKEN, help="DeGirum cloud token")
    parser.add_argument("--zoo-url", default=Config.ZOO_URL, help="Model zoo URL or local directory path")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: set-token
    parser_set_token = subparsers.add_parser("set-token", help="Save DeGirum cloud token to system license file")
    parser_set_token.add_argument("token_value", help="Your free DeGirum token from https://cs.degirum.com")

    # Subcommand: run
    parser_run = subparsers.add_parser("run", help="Run live camera/video face recognition")
    parser_run.add_argument("--source", default=str(Config.DEFAULT_CAMERA_SOURCE), help="Camera index or video filepath")
    parser_run.add_argument("--threshold", type=float, default=Config.SIMILARITY_THRESHOLD, help="Cosine similarity threshold")

    # Subcommand: enroll
    parser_enroll = subparsers.add_parser("enroll", help="Enroll image or directory of images")
    parser_enroll.add_argument("--path", required=True, help="Path to image file or directory")
    parser_enroll.add_argument("--name", default="", help="Person identity name (for single image enrollment)")

    # Subcommand: list
    parser_list = subparsers.add_parser("list", help="List enrolled identities in database")

    # Subcommand: delete
    parser_delete = subparsers.add_parser("delete", help="Delete identity from database")
    parser_delete.add_argument("--name", required=True, help="Identity name to delete")

    # Subcommand: web
    parser_web = subparsers.add_parser("web", help="Launch Web UI Dashboard")
    parser_web.add_argument("--host", default="0.0.0.0", help="Host IP to bind web server")
    parser_web.add_argument("--port", type=int, default=5000, help="Port to bind web server")

    args = parser.parse_args()

    if args.token:
        Config.TOKEN = args.token
    if args.zoo_url:
        Config.ZOO_URL = args.zoo_url

    if args.command == "set-token":
        cmd_set_token(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
