"""
Image preprocessing utilities for biometric verification.

Includes resizing, masking, and image preparation functions.
"""
import os
from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import iris
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..results import ImageData, ImageType


def prepare_image_data(
    image_path: str, 
    resize_target: Optional[Tuple[int, int]] = None, 
    keep_aspect: bool = True
) -> ImageData:
    """Load and prepare image data with optional resizing.
    
    Args:
        image_path: Path to image file
        resize_target: Optional (width, height) tuple for resizing
        keep_aspect: Whether to maintain aspect ratio when resizing
        
    Returns:
        ImageData object with original and processed images
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Determine image type
    if len(img.shape) == 2:
        image_type = ImageType.GRAYSCALE
    elif img.shape[2] == 3:
        image_type = ImageType.COLOR
    elif img.shape[2] == 4:
        image_type = ImageType.COLOR
    else:
        image_type = ImageType.GRAYSCALE
        
    # Resize if requested
    processed = img
    if resize_target:
        processed = resize_image(img, resize_target, keep_aspect)
        
    return ImageData(
        original=img,
        processed=processed,
        image_type=image_type,
        filename=os.path.basename(image_path)
    )


def resize_image(
    img: np.ndarray, 
    target_size: Tuple[int, int] = (640, 480), 
    keep_aspect: bool = False
) -> np.ndarray:
    """Resize an image either to a fixed size or while keeping aspect ratio.
    
    Args:
        img: Input image as numpy array
        target_size: (width, height) for target size
        keep_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image as numpy array
    """
    if keep_aspect:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    else:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def create_iris_mask(img: np.ndarray, exclude_pupil: bool = False) -> np.ndarray:
    """Create segmentation mask for iris region.
    
    Args:
        img: Input image (will be converted to grayscale if needed)
        exclude_pupil: If True, masks only iris without pupil
        
    Returns:
        Binary mask (0/1) where 1 represents iris region
    """
    iris_pipeline = iris.IRISPipeline()

    # Ensure image is in GRAYSCALE
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run the pipeline
    output = iris_pipeline(
        iris.IRImage(img_data=img, image_id="image_id", eye_side="left")
    )

    # Get segmentation map object
    segmap_obj = iris_pipeline.call_trace['segmentation']

    # Extract the softmax predictions
    preds = segmap_obj.predictions

    if exclude_pupil:
        # Iris only (exclude pupil - class 2)
        iris_probs = preds[:, :, 1]  # Class 1: iris
        pupil_probs = preds[:, :, 2]  # Class 2: pupil
        iris_mask = ((iris_probs > 0.5) & (pupil_probs <= 0.3)).astype(np.uint8)
    else:
        # Iris including pupil
        iris_probs = preds[:, :, 1]
        iris_mask = (iris_probs > 0.5).astype(np.uint8)

    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_OPEN, kernel)
    iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_CLOSE, kernel)

    return iris_mask


def create_hand_mask(img: np.ndarray) -> np.ndarray:
    """Create segmentation mask for hand region using skin color detection.
    
    Args:
        img: Input image (will be converted to BGR if needed)
        
    Returns:
        Binary mask (0/1) where 1 represents hand region
    """
    # Ensure image is in BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Improved skin range
    lower = np.array([0, 140, 85], dtype=np.uint8)
    upper = np.array([255, 180, 138], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    # Optional: clean up noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.medianBlur(mask, 5)

    # Remove dark shadows
    shadow = (Y < 60).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow))

    # Fill holes (e.g., fingernails)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # Keep only the largest blob (the hand)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    return mask


def create_face_mask(img: np.ndarray, model_path: str = 'face_segmentation/selfie_multiclass_256x256.tflite') -> np.ndarray:
    """Create segmentation mask for face region using MediaPipe.
    
    The model provides these class IDs:
    - 0: background
    - 1: hair
    - 2: body-skin
    - 3: face-skin
    - 4: clothes
    - 5: accessories
    
    Args:
        img: Input image (RGB/BGR)
        model_path: Path to TFLite segmentation model
        
    Returns:
        Binary mask (0/1) where 1 represents foreground (everything except background)
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True
    )

    # Original image size (for resizing output)
    orig_h, orig_w = img.shape[:2]

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img
        )

        result = segmenter.segment(mp_image)
        category_mask = result.category_mask.numpy_view()  # H × W with class IDs

        # Combine everything except background into 1 class
        final_mask = (category_mask != 0).astype(np.uint8)

        # Resize to original image size
        final_mask = cv2.resize(
            final_mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        return final_mask  # 0/1 binary mask
