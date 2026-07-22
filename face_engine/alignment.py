import numpy as np
import cv2
from typing import List, Tuple

# Standard ArcFace reference keypoints for 112x112 image alignment
ARCFACE_REF_KEYPOINTS = np.array(
    [
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041],  # Right mouth corner
    ],
    dtype=np.float32,
)

def align_and_crop(img: np.ndarray, landmarks: List[Tuple[float, float]], image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align and crop face from full image using extracted keypoint landmarks.

    Args:
        img (np.ndarray): Original input image (BGR numpy array).
        landmarks (List[Tuple[float, float]]): List of 5 keypoints [(x, y), ...] for eyes, nose, mouth.
        image_size (int): Target aligned image size (typically 112 or 128).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (aligned_cropped_face_image, affine_transformation_matrix)
    """
    if len(landmarks) != 5:
        raise ValueError(f"Expected 5 keypoint landmarks, received {len(landmarks)}")

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    else:
        raise ValueError(f"image_size must be a multiple of 112 or 128, got {image_size}")

    dst = ARCFACE_REF_KEYPOINTS * ratio
    dst[:, 0] += diff_x

    # Estimate similarity transformation matrix
    M, _ = cv2.estimateAffinePartial2D(np.array(landmarks, dtype=np.float32), dst, ransacReprojThreshold=1000)
    if M is None:
        # Fallback if matrix estimation fails
        aligned_img = cv2.resize(img, (image_size, image_size))
        return aligned_img, np.eye(2, 3)

    aligned_img = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return aligned_img, M
