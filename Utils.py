import os
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
from VerificationResult import ImageData, ImageType
import cv2
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import iris

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
    
def create_iris_mask(img: np.ndarray, exclude_pupil: bool = False):
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

def create_hand_mask(img):
    #img = cv2.bilateralFilter(img, 9, 75, 75)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Improved skin range
    lower = np.array([0, 140, 85], dtype=np.uint8)
    upper = np.array([255, 180, 138], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)


        # optional: clean up noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.medianBlur(mask, 5)

    # remove dark shadows
    shadow = (Y < 60).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow))

    # fill holes (e.g., fingernails)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # keep only the largest blob (the hand)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8) * 255

    return mask


# Usage examples:
if __name__ == "__main__":
    #img_path = 'data/iris/same/5/Iris_20220817_125828_Left.bmp'
    img_path1 = 'data/hand/different/5/Hand_0000541.jpg'
    img_path2 = 'data/hand/different/5/Hand_0000723.jpg'

    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Iris including pupil
    #iris_with_pupil = create_iris_mask(img, exclude_pupil=False)
    
    # Iris excluding pupil  
    #iris_without_pupil = create_iris_mask(img, exclude_pupil=True)

    # Hand mask
    hand_mask1 = create_hand_mask(img1)
    hand_mask2 = create_hand_mask(img2)
    
    # Display results
    plt.figure(figsize=(10, 5))
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(iris_with_pupil, cmap='gray')
    # plt.title("Iris with Pupil")
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(iris_without_pupil, cmap='gray')
    # plt.title("Iris without Pupil")

    plt.subplot(1, 2, 1)
    plt.imshow(hand_mask1, cmap='gray')
    plt.title("Hand Mask 1")

    plt.subplot(1, 2, 2)
    plt.imshow(hand_mask2, cmap='gray')
    plt.title("Hand Mask 2")
    
    plt.show()