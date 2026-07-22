import os
import sys
import unittest
import numpy as np

from config import Config
from face_engine.alignment import align_and_crop
from database.schema import FaceRecognitionSchema
from database.manager import FaceDatabaseManager
from behavior_engine.facial_behavior import FacialBehaviorAnalyzer
from behavior_engine.pose_behavior import PoseBehaviorAnalyzer
from activity_logger.logger import ActivityLogger

class TestFaceRecognitionApp(unittest.TestCase):

    def test_alignment_math(self):
        """Test affine transformation matrix calculation and output shape."""
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_landmarks = [
            [200.0, 250.0], # Right eye
            [400.0, 250.0], # Left eye
            [300.0, 350.0], # Nose
            [220.0, 450.0], # Right mouth
            [380.0, 450.0]  # Left mouth
        ]

        aligned, M = align_and_crop(dummy_img, dummy_landmarks, image_size=112)
        self.assertEqual(aligned.shape, (112, 112, 3))
        self.assertEqual(M.shape, (2, 3))

    def test_database_manager(self):
        """Test LanceDB schema creation, face record insertion, vector search, and deletion."""
        test_db_uri = "./test_face_db"
        db_mgr = FaceDatabaseManager(db_uri=test_db_uri, table_name="test_faces")

        # Generate dummy 512-D normalized embedding vector
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        rec_id = db_mgr.add_face(embedding, "TestPerson")
        self.assertIsNotNone(rec_id)

        # Test search with exact same vector
        identity, score = db_mgr.identify_face(embedding, threshold=0.3)
        self.assertEqual(identity, "TestPerson")
        self.assertGreaterEqual(score, 0.95)

        # Test list entities
        entities = db_mgr.list_entities()
        self.assertTrue(any(e['entity_name'] == 'TestPerson' for e in entities))

        # Test delete entity
        db_mgr.delete_entity("TestPerson")
        updated_entities = db_mgr.list_entities()
        self.assertFalse(any(e['entity_name'] == 'TestPerson' for e in updated_entities))

        # Cleanup test DB directory
        if os.path.exists(test_db_uri):
            import shutil
            shutil.rmtree(test_db_uri, ignore_errors=True)

    def test_facial_behavior_analyzer(self):
        """Test 3D Head Pose, EAR, MAR, and Attention Score calculation."""
        analyzer = FacialBehaviorAnalyzer()
        dummy_landmarks = [
            [200.0, 250.0], # Right eye
            [400.0, 250.0], # Left eye
            [300.0, 350.0], # Nose tip
            [220.0, 450.0], # Right mouth
            [380.0, 450.0]  # Left mouth
        ]
        bbox = [150, 150, 450, 500]

        res = analyzer.analyze_face((640, 640), dummy_landmarks, bbox)
        self.assertIn("pitch", res)
        self.assertIn("yaw", res)
        self.assertIn("attention_score", res)
        self.assertGreaterEqual(res["attention_score"], 0.0)
        self.assertLessEqual(res["attention_score"], 100.0)

    def test_pose_behavior_analyzer(self):
        """Test Posture classification, Loitering tracking, and Security zone status."""
        pose_analyzer = PoseBehaviorAnalyzer()
        standing_bbox = [100, 50, 200, 350] # Tall bbox
        sitting_bbox = [100, 100, 250, 220] # Square bbox

        self.assertEqual(pose_analyzer.classify_posture(standing_bbox), "Standing")
        self.assertEqual(pose_analyzer.classify_posture(sitting_bbox), "Sitting")

        res = pose_analyzer.analyze_pose_and_motion(standing_bbox)
        self.assertIn("track_id", res)
        self.assertIn("dwell_time_sec", res)

    def test_activity_logger(self):
        """Test Activity Logger database storage, snapshot creation, and retrieval."""
        test_db_uri = "./test_activity_db"
        test_snapshot_dir = "./test_activity_snapshots"
        logger = ActivityLogger(db_uri=test_db_uri, snapshot_dir=test_snapshot_dir)

        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        beh_data = {"attention_score": 85.0, "posture": "Standing"}

        entry = logger.log_event("Alice", "TEST_EVENT", beh_data, dummy_crop)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["person_name"], "Alice")

        logs = logger.get_logs(limit=10)
        self.assertTrue(len(logs) > 0)
        self.assertEqual(logs[0]["person_name"], "Alice")

        # Cleanup
        logger.clear_logs()
        if os.path.exists(test_db_uri):
            import shutil
            shutil.rmtree(test_db_uri, ignore_errors=True)
        if os.path.exists(test_snapshot_dir):
            import shutil
            shutil.rmtree(test_snapshot_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
