import os
import logging
import numpy as np
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType
    )
    HAILO_PLATFORM_AVAILABLE = True
except ImportError:
    HAILO_PLATFORM_AVAILABLE = False


class HailoInferenceEngine:
    """
    Native HailoRT Python Inference Engine for running .hef models directly on Hailo-8 NPU.
    Zero third-party cloud dependencies or external licensing required.
    """

    def __init__(self, hef_path: str):
        self.hef_path = hef_path
        self.hef = None
        self.target = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.is_ready = False

        if not HAILO_PLATFORM_AVAILABLE:
            logger.warning(f"hailo_platform SDK is not installed in Python env. Engine '{hef_path}' running in fallback mode.")
            return

        if not os.path.exists(hef_path):
            logger.warning(f"HEF model file not found at '{hef_path}'. Engine running in fallback mode.")
            return

        self._init_hailo()

    def _init_hailo(self):
        try:
            logger.info(f"Loading native Hailo HEF model '{self.hef_path}'...")
            self.hef = HEF(self.hef_path)
            self.target = VDevice()
            configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.target.configure(self.hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()

            self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
            self.is_ready = True
            logger.info(f"Hailo HEF model '{self.hef_path}' configured successfully on Hailo-8 NPU.")
        except Exception as e:
            logger.error(f"Failed to initialize Hailo HEF model '{self.hef_path}': {e}")
            self.is_ready = False

    def infer(self, input_data: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Execute synchronous inference on Hailo-8 NPU.
        """
        if not self.is_ready:
            return None

        try:
            with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
                input_vstream_info = self.hef.get_input_vstream_infos()[0]
                input_dict = {input_vstream_info.name: input_data.astype(np.float32)}

                with self.network_group.activate(self.network_group_params):
                    results = infer_pipeline.infer(input_dict)
                    return results
        except Exception as e:
            logger.error(f"HailoRT inference error: {e}")
            return None
