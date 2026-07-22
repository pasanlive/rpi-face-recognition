try:
    import degirum as dg
    DEGIRUM_AVAILABLE = True
except ImportError:
    dg = None
    DEGIRUM_AVAILABLE = False

import logging
from typing import Any, Union
from config import Config

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Wrapper for loading and executing Face Detection models via DeGirum PySDK on Hailo-8.
    """

    def __init__(
        self,
        model_name: str = Config.DEFAULT_DETECTION_MODEL,
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
            logger.warning("DeGirum PySDK is not installed. FaceDetector running in offline/mock mode.")
            return

        zoo_url = self.zoo_url
        if zoo_url.startswith("degirum") and os.path.exists("./models"):
            logger.info("Found local './models' directory. Attempting local model loading...")
            zoo_url = "./models"

        logger.info(f"Loading Face Detector model '{self.model_name}' on '{self.inference_host_address}' (zoo: '{zoo_url}')...")
        try:
            kwargs = {
                "model_name": self.model_name,
                "inference_host_address": self.inference_host_address,
                "zoo_url": zoo_url,
                "overlay_color": (0, 255, 0)
            }
            if self.token:
                kwargs["token"] = self.token

            self.model = dg.load_model(**kwargs)
            self.model.overlay_show_probabilities = False
            logger.info("Face Detector model loaded successfully.")
        except Exception as e:
            err_msg = str(e)
            if "Authorization failed" in err_msg or "connect to server hub.degirum.com" in err_msg:
                logger.error(
                    f"\n{'='*60}\n"
                    f"DeGirum AI Hub Authorization Failed when loading '{self.model_name}'.\n\n"
                    f"TO FIX THIS:\n"
                    f"Option A (Cloud Zoo): Provide your free DeGirum token:\n"
                    f"   export FACE_APP_TOKEN='your_token_here'\n"
                    f"   (Get free token at https://cs.degirum.com)\n\n"
                    f"Option B (Local Zoo): Download models to a local directory and set:\n"
                    f"   export FACE_APP_ZOO_URL='/path/to/models'\n"
                    f"{'='*60}\n"
                )
            else:
                logger.error(f"Failed to load Face Detector model '{self.model_name}': {e}")
            raise e

    def detect(self, image_source: Union[str, Any]):
        """
        Run face detection on an image path, numpy array, or PIL image.
        Returns DeGirum inference result containing bbox and keypoint landmarks.
        """
        return self.model(image_source)

    def detect_batch(self, image_sources):
        """
        Run batch inference on multiple images.
        """
        return self.model.predict_batch(image_sources)
