import os
try:
    import degirum as dg
    DEGIRUM_AVAILABLE = True
except ImportError:
    dg = None
    DEGIRUM_AVAILABLE = False

import numpy as np
import logging
from typing import Any, List, Union
from config import Config

logger = logging.getLogger(__name__)

class FaceEmbedder:
    """
    Wrapper for loading and executing ArcFace embedding extraction via DeGirum PySDK on Hailo-8.
    """

    def __init__(
        self,
        model_name: str = Config.EMBEDDING_MODEL,
        inference_host_address: str = Config.INFERENCE_HOST,
        zoo_url: str = Config.ZOO_URL,
        token: str = Config.TOKEN
    ):
        self.model_name = model_name
        self.inference_host_address = inference_host_address
        self.zoo_url = zoo_url
        self.token = token
        self.model = None
        self._load_model()

    def _load_model(self):
        if not DEGIRUM_AVAILABLE:
            logger.warning("DeGirum PySDK is not installed. FaceEmbedder running in offline/mock mode.")
            return

        if self.token:
            os.environ["DEGIRUM_CLOUD_TOKEN"] = self.token
            try:
                if hasattr(dg, "set_token"):
                    dg.set_token(self.token)
            except Exception:
                pass

        possible_local_paths = [
            self.zoo_url,
            "./models",
            "../models",
            "../hailo_examples/models",
            os.path.expanduser("~/hailo_examples/models"),
            os.path.expanduser("~/Documents/hailo_examples/models")
        ]
        zoo_url = self.zoo_url
        if zoo_url.startswith("degirum"):
            for candidate in possible_local_paths:
                if not candidate.startswith("degirum") and os.path.exists(candidate):
                    logger.info(f"Found local model zoo directory at '{candidate}'. Using local models.")
                    zoo_url = candidate
                    break

        logger.info(f"Loading Face Embedding model '{self.model_name}' on '{self.inference_host_address}' (zoo: '{zoo_url}')...")
        try:
            kwargs = {
                "model_name": self.model_name,
                "inference_host_address": self.inference_host_address,
                "zoo_url": zoo_url,
            }
            if self.token:
                kwargs["token"] = self.token

            self.model = dg.load_model(**kwargs)
            logger.info("Face Embedding model loaded successfully.")
        except Exception as e:
            err_msg = str(e)
            if "Authorization failed" in err_msg or "Token is not installed" in err_msg or "connect to server hub.degirum.com" in err_msg:
                logger.error(
                    f"\n{'='*60}\n"
                    f"DeGirum PySDK License Token Missing for '{self.model_name}'.\n\n"
                    f"TO FIX THIS:\n"
                    f"1) Register your free DeGirum cloud token on this system:\n"
                    f"   export DEGIRUM_CLOUD_TOKEN='your_token_here'\n"
                    f"   OR run: python3 main.py set-token 'your_token_here'\n\n"
                    f"Get a free token at: https://cs.degirum.com\n"
                    f"{'='*60}\n"
                )
            else:
                logger.error(f"Failed to load Face Embedding model '{self.model_name}': {e}")
            raise e

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract 512-D feature embedding vector for a single aligned face.
        """
        result = self.model(aligned_face)
        embedding = np.array(result.results[0]["data"][0], dtype=np.float32)
        return embedding

    def extract_batch(self, aligned_faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract feature embeddings for a batch of aligned faces.
        """
        if not aligned_faces:
            return []

        embeddings = []
        for res in self.model.predict_batch(aligned_faces):
            emb = np.array(res.results[0]["data"][0], dtype=np.float32)
            embeddings.append(emb)
        return embeddings
