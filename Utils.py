import os
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
from VerificationResult import ImageData, ImageType
import cv2

# This could be used in the future to prepare image data consistently
def prepare_image_data(image_path: str, resize_target: Optional[Tuple[int, int]] = None, 
                          keep_aspect: bool = True) -> ImageData:
        """Load and prepare image data with optional resizing"""
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Determine image type
        # Probably dont want to detect this, bacause some methods require grayscale even for color images
        if len(img.shape) == 2:
            image_type = ImageType.GRAYSCALE
        elif img.shape[2] == 3:
            image_type = ImageType.RGB
        elif img.shape[2] == 4:
            image_type = ImageType.RGBA
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

def resize_image(img, target_size=(640, 480), keep_aspect=False):
    """
    Resize an image either to a fixed size or while keeping aspect ratio.
    
    Args:
        img (np.ndarray): Input image.
        target_size (tuple): (width, height) if keep_aspect=False.
        keep_aspect (bool): Whether to maintain aspect ratio.
        
    Returns:
        np.ndarray: Resized image.
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