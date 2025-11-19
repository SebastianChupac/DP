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
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    # ensure image is in BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


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
    mask = (labels == largest).astype(np.uint8)

    return mask

def create_face_mask(img):

    # The model provides these class IDs:
    # 0 - background
    # 1 - hair
    # 2 - body-skin
    # 3 - face-skin
    # 4 - clothes
    # 5 - accessories

    base_options = python.BaseOptions(
        model_asset_path='face_segmentation/selfie_multiclass_256x256.tflite'
    )
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

        # -------------------------------
        # Stored separately, but not used directly
        # -------------------------------
        background_mask = (category_mask == 0).astype(np.uint8)
        hair_mask        = (category_mask == 1).astype(np.uint8)
        body_skin_mask   = (category_mask == 2).astype(np.uint8)
        face_skin_mask   = (category_mask == 3).astype(np.uint8)
        clothes_mask     = (category_mask == 4).astype(np.uint8)
        accessory_mask   = (category_mask == 5).astype(np.uint8)

        # -------------------------------
        # Combine everything except background into 1 class
        # -------------------------------
        final_mask = (category_mask != 0).astype(np.uint8)

        # -------------------------------
        # Resize to original image size
        # -------------------------------
        final_mask = cv2.resize(
            final_mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        return final_mask  # 0/1 binary mask




# Usage examples:
if __name__ == "__main__":
    image = cv2.imread('data/face/different/5/009_01_01_200_14_crop_128.png')
    mask = create_face_mask(image)
    print(f'Segmentation mask:')
    plt.figure(figsize=(14, 8))
    plt.imshow(mask, cmap='gray')
    plt.title("Segmentation Mask")
    plt.axis("off")
    plt.show()
    #img_path = 'data/iris/same/5/Iris_20220817_125828_Left.bmp'
    #img_path1 = 'data/hand/different/5/Hand_0000541.jpg'
    #img_path2 = 'data/hand/different/5/Hand_0000723.jpg'

    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #img1 = cv2.imread(img_path1)
    #img2 = cv2.imread(img_path2)

    # Iris including pupil
    #iris_with_pupil = create_iris_mask(img, exclude_pupil=False)
    
    # Iris excluding pupil  
    #iris_without_pupil = create_iris_mask(img, exclude_pupil=True)

    # Hand mask
    #hand_mask1 = create_hand_mask(img1)
    #hand_mask2 = create_hand_mask(img2)
    
    # Display results
    #plt.figure(figsize=(10, 5))
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(iris_with_pupil, cmap='gray')
    # plt.title("Iris with Pupil")
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(iris_without_pupil, cmap='gray')
    # plt.title("Iris without Pupil")

    # plt.subplot(1, 2, 1)
    # plt.imshow(hand_mask1, cmap='gray')
    # plt.title("Hand Mask 1")

    # plt.subplot(1, 2, 2)
    # plt.imshow(hand_mask2, cmap='gray')
    # plt.title("Hand Mask 2")
    
    # plt.show()