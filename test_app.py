import os
import sys
import unittest
import numpy as np

from config import Config
from face_engine.alignment import align_and_crop
from database.schema import FaceRecognitionSchema
from database.manager import FaceDatabaseManager

class TestFaceRecognitionApp(unittest.TestCase):

    def test_alignment_math(self):
        """Test affine transformation matrix calculation and output shape."""
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_landmarks = [
            (200.0, 250.0), # Left eye
            (400.0, 250.0), # Right eye
            (300.0, 350.0), # Nose
            (220.0, 450.0), # Left mouth
            (380.0, 450.0)  # Right mouth
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

        # Test search with random different vector
        random_emb = np.random.randn(512).astype(np.float32)
        random_emb = random_emb / np.linalg.norm(random_emb)
        diff_identity, diff_score = db_mgr.identify_face(random_emb, threshold=0.85)
        self.assertEqual(diff_identity, "Unknown")

        # Test list entities
        entities = db_mgr.list_entities()
        self.assertTrue(any(e['entity_name'] == 'TestPerson' for e in entities))

        # Test delete entity
        db_mgr.delete_entity("TestPerson")
        updated_entities = db_mgr.list_entities()
        self.assertFalse(any(e['entity_name'] == 'TestPerson' for e in updated_entities))

        # Cleanup test DB directory
        import shutil
        if os.path.exists(test_db_uri):
            shutil.rmtree(test_db_uri)

if __name__ == '__main__':
    unittest.main()
