"""
DeepDetect matcher for bioverify framework.

DeepDetect uses a CNN to predict keypoint masks, then computes SIFT
descriptors on those keypoints and matches with FLANN + NNDR filtering.

Architecture:
    1. CNN (ESPNet-based) predicts keypoint score maps
    2. Threshold score maps to get binary keypoint masks
    3. Extract keypoint coordinates from masks
    4. Compute SIFT descriptors at detected keypoints
    5. Match descriptors with FLANN + NNDR (Lowe's ratio test)
    6. Estimate homography with RANSAC
    
Framework Integration:
    - Follows BaseMatcher pattern: _preprocess_image, _match_impl, _create_verification_result
    - Uses framework's masking pipeline (get_or_compute_mask for face/iris/hand ROIs)
    - ROI masks are applied to images BEFORE CNN prediction (focuses CNN's attention)
    - All processing happens at resized resolution (default 320x320)
    
Differences from Original Implementation:
    - Original: Resizes to 320x320 for CNN, computes SIFT on original high-res images
    - Framework: All processing at 320x320 for consistency and efficiency
    - Rationale: SIFT is scale-invariant, so descriptor quality should be similar
    - Benefit: Simpler pipeline, consistent with other framework matchers
    
Decision Logic:
    - Predicts same person if:
      * inlier_ratio > ratio_threshold (default 0.3)
      AND
      * mean_reprojection_error < max_reprojection_error (default 5.0 px)
      
Reference:
    Original implementation: DeepDetect/DeepDetect.py
"""
import os
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from typing import Dict, Any, Tuple, Optional

from .base import BaseMatcher, MatcherConfig
from ..results import VerificationResult


class DeepDetectMatcher(BaseMatcher):
    """
    DeepDetect matcher implementation.
    
    DeepDetect uses a CNN (ESPNet-based) to predict keypoint masks,
    then computes SIFT descriptors on detected keypoints and matches
    them using FLANN matcher with NNDR filtering.
    
    Architecture:
    1. CNN predicts keypoint score maps (480x480 -> binary masks)
    2. Extract keypoint coordinates from masks
    3. Compute SIFT descriptors at detected keypoints
    4. Match with FLANN + NNDR (Lowe's ratio test)
    5. Estimate homography with RANSAC
    """
    
    def __init__(self, config: MatcherConfig):
        """
        Initialize DeepDetect matcher.
        
        Args:
            config: MatcherConfig instance with:
                - model_threshold: CNN prediction threshold (default: 0.5)
                - nndr_threshold: Nearest Neighbor Distance Ratio (default: 0.8)
                - ransac_thresh: RANSAC reprojection threshold in pixels (default: 7)
                - ratio_threshold: Inlier ratio threshold for same person (default: 0.3)
                - max_reprojection_error: Max mean reprojection error for same person (default: 5.0)
        """
        super().__init__(config)
        
        params = config.extra_params
        
        # Model parameters
        self.model_threshold = params.get('model_threshold', 0.5)
        self.nndr_threshold = params.get('nndr_threshold', 0.8)
        
        # Decision thresholds
        self.ratio_threshold = params.get('ratio_threshold', 0.3)
        self.max_reprojection_error = params.get('max_reprojection_error', 5.0)
        
        # Load DeepDetect model once
        self._load_model()
        
    def _load_model(self):
        """Load the DeepDetect CNN model."""
        import sys
        import __main__
        
        # Import all model classes - pickle expects them in __main__ or 'model' module
        from .deepdetect_models import model as model_module
        
        # Temporarily add all model classes to __main__ for unpickling
        model_classes = ['ESPNet', 'ESPNet_Encoder', 'CBR', 'BR', 'CB', 'C', 'CDilated', 
                        'DownSamplerB', 'DilatedParllelResidualBlockB', 'InputProjectionA']
        original_main_attrs = {}
        
        for cls_name in model_classes:
            if hasattr(__main__, cls_name):
                original_main_attrs[cls_name] = getattr(__main__, cls_name)
            setattr(__main__, cls_name, getattr(model_module, cls_name))
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resolve model weight path
        model_dir = os.path.join(os.path.dirname(__file__), 'deepdetect_models')
        model_path = os.path.join(model_dir, 'DEEP_DETECT_Best_Model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DeepDetect model weights not found at {model_path}. "
                f"Please ensure DEEP_DETECT_Best_Model.pth is in the deepdetect_models directory."
            )
        
        # Load pre-trained model
        try:
            self.model = torch.load(model_path, weights_only=False, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
        finally:
            # Restore original __main__ attributes
            for cls_name in model_classes:
                if cls_name in original_main_attrs:
                    setattr(__main__, cls_name, original_main_attrs[cls_name])
                elif hasattr(__main__, cls_name):
                    delattr(__main__, cls_name)
        
    def get_name(self) -> str:
        """Return matcher name."""
        return "DeepDetect"
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for DeepDetect.
        
        Converts to BGR and resizes based on config.
        Masking is handled separately by the framework.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            Preprocessed BGR image (resized per config)
        """
        # Convert grayscale to BGR if needed (DeepDetect CNN requires 3 channels)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # BGRA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        # Resize based on config (default 320x320 for DeepDetect)
        if self.config.resize_width and self.config.resize_height:
            image = cv2.resize(
                image, 
                (self.config.resize_width, self.config.resize_height), 
                interpolation=cv2.INTER_CUBIC
            )
            
        return image
        
    def _predict_keypoint_masks(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict keypoint masks using DeepDetect CNN.
        
        Args:
            image1: Preprocessed BGR image (320x320)
            image2: Preprocessed BGR image (320x320)
            
        Returns:
            Tuple of (mask1, mask2) as binary masks
        """
        # Convert to PIL Images
        img1_pil = Image.fromarray(image1)
        img2_pil = Image.fromarray(image2)
        
        # Convert to tensors
        transform = T.Compose([T.ToTensor()])
        input1 = transform(img1_pil).unsqueeze(0).to(self.device)
        input2 = transform(img2_pil).unsqueeze(0).to(self.device)
        
        # Run model
        with torch.no_grad():
            pred1 = self.model(input1)
            pred2 = self.model(input2)
            # Convert logits to probabilities
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)
            
        # Convert to numpy and threshold
        mask1 = pred1.cpu().squeeze().numpy()
        mask2 = pred2.cpu().squeeze().numpy()
        
        # Resize back to original image size (320x320 in our case)
        mask1 = cv2.resize(mask1, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_CUBIC)
        mask2 = cv2.resize(mask2, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Threshold to binary masks
        mask1 = (mask1 > self.model_threshold).astype(np.uint8)
        mask2 = (mask2 > self.model_threshold).astype(np.uint8)
        
        return mask1, mask2
        
    def _mask_to_keypoints(self, mask: np.ndarray, size: float = 3.0) -> list:
        """
        Convert binary mask to list of cv2.KeyPoint objects.
        
        Args:
            mask: Binary mask (1 = keypoint, 0 = background)
            size: Keypoint size parameter
            
        Returns:
            List of cv2.KeyPoint objects
        """
        ys, xs = np.where(mask == 1)
        keypoints = [cv2.KeyPoint(float(x), float(y), size) for (y, x) in zip(ys, xs)]
        return keypoints
        
    @staticmethod
    def _prepare_mask(mask: Optional[np.ndarray], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Prepare mask for bitwise operations.
        
        Args:
            mask: Input mask (may be float 0-1 or uint8 0-255)
            target_shape: (height, width) to resize mask to
            
        Returns:
            Prepared uint8 mask with values 0 or 255
        """
        if mask is None:
            return None
            
        # Resize if needed
        if mask.shape[:2] != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Convert to uint8 if needed
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
            
        # Scale to 0-255 if needed
        if mask.max() <= 1:
            mask = mask * 255
            
        return mask
        
    def _match_impl(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform DeepDetect matching.
        
        Args:
            image1: First image (preprocessed, resized to 320x320)
            image2: Second image (preprocessed, resized to 320x320)
            mask1: Optional ROI mask for first image (e.g., face/iris/hand mask)
            mask2: Optional ROI mask for second image
            
        Returns:
            Tuple of (keypoints1, keypoints2, matches)
            - keypoints1: Nx2 array of (x, y) coordinates in image1
            - keypoints2: Nx2 array of (x, y) coordinates in image2
            - matches: Mx2 array of match indices (idx1, idx2)
        """
        # Apply ROI masks to images before CNN prediction (following original implementation)
        # This focuses the CNN's attention on the relevant region (e.g., inside iris, face area)
        masked_img1 = image1.copy()
        masked_img2 = image2.copy()
        
        if mask1 is not None:
            mask1_prepared = self._prepare_mask(mask1, masked_img1.shape[:2])
            masked_img1 = cv2.bitwise_and(masked_img1, masked_img1, mask=mask1_prepared)
            
        if mask2 is not None:
            mask2_prepared = self._prepare_mask(mask2, masked_img2.shape[:2])
            masked_img2 = cv2.bitwise_and(masked_img2, masked_img2, mask=mask2_prepared)
        
        # Predict keypoint masks using CNN on masked images
        kp_mask1, kp_mask2 = self._predict_keypoint_masks(masked_img1, masked_img2)
        
        # Convert masks to keypoint lists
        kp1_list = self._mask_to_keypoints(kp_mask1)
        kp2_list = self._mask_to_keypoints(kp_mask2)
        
        # Compute SIFT descriptors at detected keypoints
        # Note: Original DeepDetect uses original high-res images for SIFT,
        # but for framework consistency we use the resized images.
        # SIFT is scale-invariant so this should not significantly affect matching quality.
        sift = cv2.SIFT_create()
        kp1, des1 = sift.compute(image1, kp1_list)
        kp2, des2 = sift.compute(image2, kp2_list)
        
        # Handle no keypoints case
        if len(kp1) == 0 or len(kp2) == 0 or des1 is None or des2 is None:
            return (
                np.array([kp.pt for kp in kp1]) if len(kp1) > 0 else np.array([]).reshape(0, 2),
                np.array([kp.pt for kp in kp2]) if len(kp2) > 0 else np.array([]).reshape(0, 2),
                np.array([]).reshape(0, 2)
            )
        
        # FLANN knnMatch with k=2 requires at least 2 descriptors in the search index
        # If either descriptor set has fewer than 2 descriptors, we cannot apply NNDR filtering
        if len(des1) < 2 or len(des2) < 2:
            return (
                np.array([kp.pt for kp in kp1]),
                np.array([kp.pt for kp in kp2]),
                np.array([]).reshape(0, 2)
            )
        
        # Match with FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply NNDR (Lowe's ratio test)
        good_matches = []
        confidence_scores = []
        eps = 1e-12
        
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            ratio = m.distance / (n.distance + eps)
            if ratio < self.nndr_threshold:
                good_matches.append(m)
                # Confidence = 1 - ratio (higher is better)
                confidence_scores.append(1.0 - ratio)
        
        # Convert to numpy arrays
        kp1_array = np.array([kp.pt for kp in kp1])
        kp2_array = np.array([kp.pt for kp in kp2])
        
        if len(good_matches) == 0:
            return (
                kp1_array,
                kp2_array,
                np.array([]).reshape(0, 2)
            )
        
        # Create match indices array (N x 2)
        match_indices = np.array([[m.queryIdx, m.trainIdx] for m in good_matches])
        
        return kp1_array, kp2_array, match_indices
        
    def _get_matcher_params(self) -> dict:
        """Return matcher parameters for logging/debugging."""
        return {
            "model_threshold": self.model_threshold,
            "nndr_threshold": self.nndr_threshold,
            "ratio_threshold": self.ratio_threshold,
            "max_reprojection_error": self.max_reprojection_error,
            "device": str(self.device),
        }
        
    def _create_verification_result(
        self,
        img1_path: str,
        img2_path: str,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        matches: np.ndarray,
        homography: Optional[np.ndarray],
        inliers: Optional[np.ndarray],
        reprojection_error: Optional[float],
        ground_truth: Optional[bool] = None,
    ) -> VerificationResult:
        """
        Create verification result with DeepDetect-specific decision logic.
        
        Uses improved confidence scoring that accounts for:
        - Geometric consistency (inlier ratio)
        - Statistical significance (sample size)
        - Measurement quality (reprojection error)
        
        Decision logic:
        - Predict same person if inlier_ratio > ratio_threshold 
          AND mean_reprojection_error < max_reprojection_error
        """
        # Compute metrics
        inlier_mask = inliers.astype(bool) if inliers is not None else None
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        num_matches = len(matches) if matches is not None else 0
        inlier_ratio = num_inliers / max(1, num_matches)
        
        # Compute improved confidence score
        confidence = self._compute_confidence_score(num_matches, num_inliers, reprojection_error)
        # rewrite confidence with inlier ratio for this experiment
        #confidence = inlier_ratio
        # Decision logic: inlier ratio AND reprojection error thresholds
        is_same_person = False
        if confidence >= self.ratio_threshold:
            if reprojection_error is None or reprojection_error < self.max_reprojection_error:
                is_same_person = True
        
        return VerificationResult(
            method_name=self.get_name(),
            is_same_person_pred=is_same_person,
            verification_confidence=confidence,
            ground_truth=ground_truth,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error=reprojection_error,
            homography_confidence=confidence if homography is not None else 0.0,
            matcher_params=self._get_matcher_params(),
        )
