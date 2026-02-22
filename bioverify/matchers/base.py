"""
Base matcher interface for biometric verification.

This module provides the abstract base class and configuration dataclass
for all matcher implementations.

Design Decisions:
- Input: Image file paths (str) - each matcher handles preprocessing internally
- Output: VerificationResult - unified format for all matchers
- Config: YAML-based with runtime overrides via from_dict()
- Models: Loaded once in __init__() for efficiency
- Masking: Flexible strategy - check cache first, compute if needed
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - guard for missing/broken torch installs
    torch = None

from ..results import VerificationResult, VisualizationResult, ImageData, ImageType, Keypoint, Match
from ..utils.preprocessing import (
    resize_image,
    create_iris_mask,
    create_hand_mask,
    create_face_mask,
)

PUBLIC_DATASET_ROOT = "C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset"


@dataclass
class MatcherConfig:
    """
    Configuration for a matcher.
    
    Attributes:
        resize_width: Target width for image resizing (None = no resize)
        resize_height: Target height for image resizing (None = no resize)
        ransac_thresh: RANSAC inlier threshold in pixels
        ransac_max_iters: Maximum RANSAC iterations
        min_matches: Minimum matches required for valid result
        use_masking: Whether to apply modality-specific masking
        mask_cache_dir: Directory for cached masks (None = compute on-the-fly)
        device: Device for torch models ('cuda' or 'cpu')
        extra_params: Method-specific parameters
    """
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    ransac_thresh: float = 3.0
    ransac_max_iters: int = 10000
    min_matches: int = 4
    use_masking: bool = False
    mask_cache_dir: Optional[str] = None
    device: str = "cuda"
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MatcherConfig":
        """
        Create MatcherConfig from dictionary (e.g., loaded from YAML).
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            MatcherConfig instance
        """
        # Extract known fields
        known_fields = {
            "resize_width", "resize_height", "ransac_thresh", 
            "ransac_max_iters", "min_matches", "use_masking",
            "mask_cache_dir", "device"
        }
        
        main_params = {k: v for k, v in config_dict.items() if k in known_fields}
        extra_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        
        if extra_params:
            main_params["extra_params"] = extra_params
            
        return cls(**main_params)


class BaseMatcher(ABC):
    """
    Abstract base class for all matchers.
    
    Subclasses must implement:
    - _match_impl(): Core matching logic
    - get_name(): Matcher name for logging/results
    - _create_verification_result(): Create lightweight result for experiments
    
    Subclasses can optionally override:
    - _create_visualization_result(): Create rich result for visualization
    - _preprocess_image(): Custom preprocessing beyond resize
    """
    
    def __init__(self, config: MatcherConfig):
        """
        Initialize matcher.
        
        Args:
            config: Matcher configuration
        """
        self.config = config
        
    @abstractmethod
    def get_name(self) -> str:
        """
        Get matcher name.
        
        Returns:
            Name string (e.g., "SIFT", "SuperGlue")
        """
        pass
    
    @abstractmethod
    def _match_impl(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform matching on preprocessed images.
        
        Args:
            img1: First image (grayscale or BGR)
            img2: Second image (grayscale or BGR)
            mask1: Optional mask for first image
            mask2: Optional mask for second image
            
        Returns:
            Tuple of (keypoints1, keypoints2, matches):
            - keypoints1: Nx2 array of (x, y) coordinates in img1
            - keypoints2: Nx2 array of (x, y) coordinates in img2
            - matches: Mx2 array of match indices (idx1, idx2)
        """
        pass
    
    @abstractmethod
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
        Create lightweight VerificationResult for experiment tracking.
        
        This result does NOT store images or raw keypoints to save memory
        during batch processing. Subclasses implement their own logic to
        determine is_match and confidence based on the matching metrics.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            keypoints1: Keypoints in img1 (Nx2 array)
            keypoints2: Keypoints in img2 (Nx2 array)
            matches: Match indices (Mx2 array)
            homography: Estimated homography matrix (or None)
            inliers: Boolean mask of inlier matches (or None)
            reprojection_error: Mean reprojection error (or None)
            
        Returns:
            Lightweight VerificationResult (no images/keypoints)
        """
        pass
    
    def _create_visualization_result(
        self,
        img1_path: str,
        img2_path: str,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        matches: np.ndarray,
        homography: Optional[np.ndarray],
        inliers: Optional[np.ndarray],
        reprojection_error: Optional[float],
        decision: Optional[VerificationResult] = None,
        modality: Optional[str] = None,
    ) -> VisualizationResult:
        """
        Create rich VisualizationResult for debugging and visualization.
        
        Default implementation stores all matching artifacts (images, keypoints,
        matches, descriptors, masks). Subclasses can override for custom behavior.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            img1: First image array
            img2: Second image array
            mask1: Optional mask for first image
            mask2: Optional mask for second image
            keypoints1: Keypoints in img1 (Nx2 array)
            keypoints2: Keypoints in img2 (Nx2 array)
            matches: Match indices (Mx2 array)
            homography: Estimated homography matrix (or None)
            inliers: Boolean mask of inlier matches (or None)
            reprojection_error: Mean reprojection error (or None)
            
        Returns:
            Rich VisualizationResult with all artifacts
        """
        # Compute metrics
        inlier_mask = inliers.astype(bool) if inliers is not None else None
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        num_matches = len(matches) if matches is not None else 0
        inlier_ratio = num_inliers / max(1, num_matches)
        
        return VisualizationResult(
            method_name=self.get_name(),
            modality=modality,
            image1=ImageData(
                original=img1,
                processed=img1,
                image_type=ImageType.GRAYSCALE,
                mask=mask1,
                filename=Path(img1_path).name,
            ) if img1 is not None else None,
            image2=ImageData(
                original=img2,
                processed=img2,
                image_type=ImageType.GRAYSCALE,
                mask=mask2,
                filename=Path(img2_path).name,
            ) if img2 is not None else None,
            keypoints1=[Keypoint(x=float(kp[0]), y=float(kp[1])) for kp in keypoints1] if len(keypoints1) > 0 else [],
            keypoints2=[Keypoint(x=float(kp[0]), y=float(kp[1])) for kp in keypoints2] if len(keypoints2) > 0 else [],
            matches=[Match(kp1_idx=int(m[0]), kp2_idx=int(m[1]), distance=0.0, is_inlier=bool(inlier_mask[i]) if inlier_mask is not None and i < len(inlier_mask) else None) for i, m in enumerate(matches)] if matches is not None and len(matches) > 0 else [],
            homography=homography,
            homography_confidence=inlier_ratio if homography is not None else 0.0,
            inlier_mask=inlier_mask,
            is_same_person_pred=decision.is_same_person_pred if decision else None,
            verification_confidence=decision.verification_confidence if decision else 0.0,
            ground_truth=decision.ground_truth if decision else None,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error=reprojection_error,
            matcher_params=self._get_matcher_params(),
        )
    
    def match(
        self,
        img1_path: str,
        img2_path: str,
        modality: Optional[str] = None,
        visualize: bool = False,
        ground_truth: Optional[bool] = None,
    ) -> Optional[VerificationResult]:
        """
        Main entry point for matching two images.
        
        Orchestrates the full matching pipeline:
        1. Load images
        2. Preprocess (resize, etc.)
        3. Get or compute masks
        4. Call subclass _match_impl()
        5. Estimate homography
        6. Create result (lightweight for experiments or rich for visualization)
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            modality: Modality hint for masking ('iris', 'face', 'hand', 'fingervein')
            visualize: If True, returns VisualizationResult; if False, returns VerificationResult
            
        Returns:
            VerificationResult (lightweight) or VisualizationResult (rich) depending on visualize flag
        """
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # Preprocess
        img1 = self._preprocess_image(img1)
        img2 = self._preprocess_image(img2)
        
        # Get masks if needed
        mask1 = None
        mask2 = None
        if self.config.use_masking and modality:
            mask1 = self._get_or_compute_mask(img1_path, img1, modality)
            mask2 = self._get_or_compute_mask(img2_path, img2, modality)
        
        # Perform matching
        keypoints1, keypoints2, matches = self._match_impl(img1, img2, mask1, mask2)
        
        # Estimate homography if we have enough matches
        homography = None
        inliers = None
        reprojection_error = None
        
        if len(matches) >= self.config.min_matches:
            # Extract matched points
            pts1 = keypoints1[matches[:, 0]]
            pts2 = keypoints2[matches[:, 1]]
            
            # Estimate homography
            homography, inliers = self._estimate_homography(pts1, pts2)
            
            if homography is not None:
                reprojection_error = self._compute_reprojection_error(
                    pts1[inliers], pts2[inliers], homography
                )
        
        # Create result based on visualize flag
        verification_result = self._create_verification_result(
            img1_path=img1_path,
            img2_path=img2_path,
            keypoints1=keypoints1,
            keypoints2=keypoints2,
            matches=matches,
            homography=homography,
            inliers=inliers,
            reprojection_error=reprojection_error,
            ground_truth=ground_truth,
        )
        verification_result.modality = modality

        if visualize:
            return self._create_visualization_result(
                img1_path=img1_path,
                img2_path=img2_path,
                img1=img1,
                img2=img2,
                mask1=mask1,
                mask2=mask2,
                keypoints1=keypoints1,
                keypoints2=keypoints2,
                matches=matches,
                homography=homography,
                inliers=inliers,
                reprojection_error=reprojection_error,
                decision=verification_result,
                modality=modality,
            )
        return verification_result
    
    def _get_matcher_params(self) -> Dict[str, Any]:
        """
        Get matcher parameters for result tracking.
        
        Default implementation returns extra_params from config.
        Subclasses can override to include additional parameters.
        
        Returns:
            Dictionary of matcher parameters
        """
        return dict(self.config.extra_params)
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        Load image from path.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Loaded image as numpy array (BGR format)
            
        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image can't be read
        """
        if not Path(img_path).exists():
            img_path = Path(PUBLIC_DATASET_ROOT) / img_path
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        return img
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image (resize, etc.).
        
        Default implementation handles resizing based on config.
        Subclasses can override for additional preprocessing.
        
        Args:
            img: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize if configured
        if self.config.resize_width and self.config.resize_height:
            keep_aspect = bool(
                self.config.extra_params.get(
                    "resize_keep_aspect",
                    self.config.extra_params.get("keep_aspect", False),
                )
            )
            img = resize_image(
                img,
                (self.config.resize_width, self.config.resize_height),
                keep_aspect=keep_aspect,
            )
        
        return img
    
    def _get_or_compute_mask(
        self,
        img_path: str,
        img: np.ndarray,
        modality: str,
    ) -> Optional[np.ndarray]:
        """
        Get mask from cache or compute it.
        
        Strategy:
        1. If mask_cache_dir is set, check for cached mask
        2. If not found, compute mask
        3. Optionally save to cache
        
        Args:
            img_path: Path to image (used for cache lookup)
            img: Image array
            modality: Modality type ('iris', 'face', 'hand', 'fingervein')
            
        Returns:
            Binary mask or None if computation fails
        """
        mask = None
        
        # Try loading from cache
        if self.config.mask_cache_dir:
            cache_dir = Path(self.config.mask_cache_dir)
            # Create cache filename based on image path
            img_name = Path(img_path).stem
            mask_path = cache_dir / f"{img_name}_{modality}_mask.png"
            
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Ensure binary
                    mask = (mask > 127).astype(np.uint8) * 255
                    return mask
        
        # Compute mask
        if modality == "iris":
            # TODO pass eye side
            exclude_pupil = bool(self.config.extra_params.get("iris_exclude_pupil", True))
            mask = create_iris_mask(img, exclude_pupil=exclude_pupil)
        elif modality == "face":
            mask = create_face_mask(img)
        elif modality == "hand":
            mask = create_hand_mask(img)
        elif modality == "fingervein":
            # Fingervein typically doesn't need masking (ROI already extracted)
            mask = None
        
        # Save to cache if successful and cache is enabled
        if mask is not None and self.config.mask_cache_dir:
            cache_dir = Path(self.config.mask_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            img_name = Path(img_path).stem
            mask_path = cache_dir / f"{img_name}_{modality}_mask.png"
            cv2.imwrite(str(mask_path), mask)
        
        return mask
    
    def _estimate_homography(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate homography using RANSAC.
        
        Args:
            pts1: Nx2 array of points in first image
            pts2: Nx2 array of corresponding points in second image
            
        Returns:
            Tuple of (homography, inliers):
            - homography: 3x3 matrix or None if estimation fails
            - inliers: Boolean array of inlier mask or None
        """
        if len(pts1) < 4:
            return None, None
        
        
        H, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            self.config.ransac_thresh
            #maxIters=self.config.ransac_max_iters,
        )
        
        if H is None:
            return None, None
        
        inliers = mask.ravel().astype(bool)
        return H, inliers
    
    def _compute_reprojection_error(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        homography: np.ndarray,
    ) -> float:
        """
        Compute mean reprojection error.
        
        Args:
            pts1: Nx2 array of points in first image
            pts2: Nx2 array of corresponding points in second image
            homography: 3x3 homography matrix
            
        Returns:
            Mean reprojection error in pixels
        """
        if len(pts1) == 0:
            return float("inf")
        
        # Transform pts1 using homography
        pts1_homogeneous = np.column_stack([pts1, np.ones(len(pts1))])
        pts1_transformed = (homography @ pts1_homogeneous.T).T
        pts1_transformed = pts1_transformed[:, :2] / pts1_transformed[:, 2:3]
        
        # Compute Euclidean distances
        errors = np.linalg.norm(pts1_transformed - pts2, axis=1)
        return float(np.mean(errors))
    
    def _get_device(self) -> Any:
        """
        Get torch device based on config.
        
        Returns:
            torch.device instance
        """
        if torch is None:
            return "cpu"
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
