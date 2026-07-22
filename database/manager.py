import lancedb
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from config import Config
from database.schema import FaceRecognitionSchema
from face_engine.alignment import align_and_crop

logger = logging.getLogger(__name__)

class FaceDatabaseManager:
    """
    Manages LanceDB vector database operations for storing, searching, and managing face embeddings.
    """

    def __init__(self, db_uri: str = Config.DB_URI, table_name: str = Config.TABLE_NAME):
        self.db_uri = db_uri
        self.table_name = table_name
        self.db = lancedb.connect(uri=self.db_uri)
        self.table = self._init_table()

    def _init_table(self):
        try:
            return self.db.open_table(self.table_name)
        except Exception:
            logger.info(f"Table '{self.table_name}' not found. Creating new LanceDB table at {self.db_uri}...")
            try:
                return self.db.create_table(self.table_name, schema=FaceRecognitionSchema, exist_ok=True)
            except (TypeError, Exception):
                try:
                    return self.db.create_table(self.table_name, schema=FaceRecognitionSchema)
                except Exception:
                    return self.db.open_table(self.table_name)

    def add_face(self, embedding: np.ndarray, entity_name: str) -> str:
        """
        Add a single face embedding for a person to the database.
        """
        record = FaceRecognitionSchema.prepare_record(embedding, entity_name)
        self.table.add(data=[record])
        logger.info(f"Successfully added record for identity '{entity_name}' (ID: {record.id}).")
        return record.id

    def identify_face(
        self,
        embedding: np.ndarray,
        threshold: float = Config.SIMILARITY_THRESHOLD,
        top_k: int = 1
    ) -> Tuple[str, float]:
        """
        Query database for the nearest face embedding.

        Returns:
            Tuple[str, float]: (identity_name, similarity_score)
            If distance metric score is below threshold, identity_name returns "Unknown".
        """
        if self.table.count_rows() == 0:
            return "Unknown", 0.0

        search_result = (
            self.table.search(embedding, vector_column_name="vector")
            .metric(Config.METRIC_TYPE)
            .limit(top_k)
            .to_list()
        )

        if not search_result:
            return "Unknown", 0.0

        top_match = search_result[0]
        # Cosine distance to similarity: similarity = 1 - distance
        distance = top_match["_distance"]
        similarity = round(1.0 - float(distance), 3)

        if similarity >= threshold:
            return top_match["entity_name"], similarity
        else:
            return "Unknown", similarity

    def identify_batch(
        self,
        embeddings: List[np.ndarray],
        threshold: float = Config.SIMILARITY_THRESHOLD
    ) -> List[Tuple[str, float]]:
        """
        Identify a batch of face embeddings.
        """
        results = []
        for emb in embeddings:
            res = self.identify_face(emb, threshold=threshold)
            results.append(res)
        return results

    def populate_from_directory(
        self,
        input_directory: str,
        detector: Any,
        embedder: Any
    ) -> Dict[str, int]:
        """
        Scan directory for face images (e.g. Alice_1.jpg, Bob_2.png), detect & align face,
        extract embedding, and populate the LanceDB table.
        """
        path = Path(input_directory)
        if not path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_directory}")

        valid_exts = {".png", ".jpg", ".jpeg"}
        image_files = [f for f in path.rglob("*") if f.suffix.lower() in valid_exts]

        stats = {}
        for img_file in image_files:
            # Identity derived from filename before underscore/number (e.g. "Alice_1.jpg" -> "Alice")
            identity = img_file.stem.split("_")[0].strip().capitalize()
            try:
                detected = detector.detect(str(img_file))
                num_faces = len(detected.results)

                if num_faces == 0:
                    logger.warning(f"Skipping {img_file.name}: No face detected.")
                    continue
                elif num_faces > 1:
                    logger.warning(f"Skipping {img_file.name}: Multiple faces detected ({num_faces}). Use single-face image for clean enrollment.")
                    continue

                res = detected.results[0]
                landmarks = [lm["landmark"] for lm in res["landmarks"]]
                aligned_img, _ = align_and_crop(detected.image, landmarks, image_size=Config.INPUT_FACE_SIZE)
                embedding = embedder.extract_embedding(aligned_img)

                self.add_face(embedding, identity)
                stats[identity] = stats.get(identity, 0) + 1

            except Exception as e:
                logger.error(f"Error processing {img_file.name}: {e}")

        logger.info(f"Database population completed. Added {sum(stats.values())} embeddings for {len(stats)} individuals.")
        return stats

    def list_entities(self) -> List[Dict[str, Any]]:
        """
        List all enrolled individuals and record counts.
        """
        if self.table.count_rows() == 0:
            return []

        df = self.table.to_pandas()
        grouped = df.groupby("entity_name").size().reset_index(name="count")
        entities = grouped.to_dict(orient="records")
        return entities

    def delete_entity(self, entity_name: str) -> bool:
        """
        Delete all face embeddings for the given identity entity_name.
        """
        try:
            filter_query = f"entity_name = '{entity_name}'"
            self.table.delete(filter_query)
            logger.info(f"Deleted records for entity '{entity_name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete entity '{entity_name}': {e}")
            return False

    def clear_all(self):
        """
        Clear all database entries.
        """
        try:
            self.db.drop_table(self.table_name, ignore_missing=True)
        except TypeError:
            try:
                self.db.drop_table(self.table_name)
            except Exception:
                pass
        self.table = self._init_table()
