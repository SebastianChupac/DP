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
    enhance_fingervein_image,
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
        use_enhancement: Whether to apply image enhancement for fingervein modality
        enhancement_clip_limit: CLAHE clip limit for fingervein enhancement (2.0-4.0)
        enhancement_tile_size: CLAHE tile grid size for fingervein enhancement
        device: Device for torch models ('cuda' or 'cpu')
        extra_params: Method-specific parameters
    """
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    ransac_thresh: float = 3.0
    ransac_max_iters: int = 10000
    min_matches: int = 4
    use_masking: bool = False
    use_enhancement: bool = True
    enhancement_clip_limit: float = 3.0
    enhancement_tile_size: int = 8
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
            "use_enhancement", "enhancement_clip_limit", "enhancement_tile_size",
            "device"
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
        img1_original: np.ndarray,
        img2_original: np.ndarray,
        img1_processed: np.ndarray,
        img2_processed: np.ndarray,
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
        transform_img1_orig_to_processed: Optional[np.ndarray] = None,
        transform_img2_orig_to_processed: Optional[np.ndarray] = None,
    ) -> VisualizationResult:
        """
        Create rich VisualizationResult for debugging and visualization.
        
        Default implementation stores all matching artifacts (images, keypoints,
        matches, descriptors, masks). Subclasses can override for custom behavior.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            img1_original: First image before matcher preprocessing
            img2_original: Second image before matcher preprocessing
            img1_processed: First image at matcher input stage
            img2_processed: Second image at matcher input stage
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

        img1_vis_processed = self._build_visualization_processed_image(img1_processed, mask1)
        img2_vis_processed = self._build_visualization_processed_image(img2_processed, mask2)

        metadata: Dict[str, Any] = {
            "visualization_transform_img1_orig_to_processed": (
                transform_img1_orig_to_processed.tolist()
                if transform_img1_orig_to_processed is not None else np.eye(3, dtype=np.float32).tolist()
            ),
            "visualization_transform_img2_orig_to_processed": (
                transform_img2_orig_to_processed.tolist()
                if transform_img2_orig_to_processed is not None else np.eye(3, dtype=np.float32).tolist()
            ),
            "visualization_processed_shape_img1": list(img1_vis_processed.shape[:2]) if img1_vis_processed is not None else None,
            "visualization_processed_shape_img2": list(img2_vis_processed.shape[:2]) if img2_vis_processed is not None else None,
            "visualization_original_shape_img1": list(img1_original.shape[:2]) if img1_original is not None else None,
            "visualization_original_shape_img2": list(img2_original.shape[:2]) if img2_original is not None else None,
        }
        
        return VisualizationResult(
            method_name=self.get_name(),
            modality=modality,
            image1=ImageData(
                original=img1_original,
                processed=img1_vis_processed,
                image_type=ImageType.GRAYSCALE,
                mask=mask1,
                filename=Path(img1_path).name,
            ) if img1_original is not None else None,
            image2=ImageData(
                original=img2_original,
                processed=img2_vis_processed,
                image_type=ImageType.GRAYSCALE,
                mask=mask2,
                filename=Path(img2_path).name,
            ) if img2_original is not None else None,
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
            metadata=metadata,
        )
    
    def match(
        self,
        img1_path: str,
        img2_path: str,
        modality: Optional[str] = None,
        visualize: bool = False,
        ground_truth: Optional[bool] = None,
        matcher_name: Optional[str] = None,
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
            matcher_name: Optional override for the matcher name in results (e.g., "sift-v1").
                         If not provided, uses get_name() which returns the base matcher name.
            
        Returns:
            VerificationResult (lightweight) or VisualizationResult (rich) depending on visualize flag
        """
        # Load images
        img1_original = self._load_image(img1_path)
        img2_original = self._load_image(img2_path)

        # Keep a pre-pipeline snapshot only when rich visualization is requested.
        if visualize:
            img1_vis_original = img1_original.copy()
            img2_vis_original = img2_original.copy()
        else:
            img1_vis_original = None
            img2_vis_original = None

        img1 = img1_original
        img2 = img2_original
        transform_img1_orig_to_processed = np.eye(3, dtype=np.float32)
        transform_img2_orig_to_processed = np.eye(3, dtype=np.float32)
        
        # Preprocess
        img1 = self._preprocess_image(img1)
        img2 = self._preprocess_image(img2)
        transform_img1_orig_to_processed = self._compose_with_resize_transform(
            transform_img1_orig_to_processed,
            img1_original.shape[:2],
            img1.shape[:2],
        )
        transform_img2_orig_to_processed = self._compose_with_resize_transform(
            transform_img2_orig_to_processed,
            img2_original.shape[:2],
            img2.shape[:2],
        )
        
        # Get masks if needed (AFTER preprocessing so masks match image size)
        mask1 = None
        mask2 = None
        if self.config.use_masking and modality:
            mask1 = self._get_or_compute_mask(img1_path, img1, modality)
            mask2 = self._get_or_compute_mask(img2_path, img2, modality)

        # Optional ROI focusing step for sparse masks (e.g., iris and hand geometry).
        # Crops image+mask to mask bounding box, then resizes back to target shape.
        if self._should_crop_to_mask_roi(modality):
            if mask1 is not None:
                img1, mask1, crop_transform1 = self._crop_and_resize_to_mask_roi(img1, mask1)
                transform_img1_orig_to_processed = crop_transform1 @ transform_img1_orig_to_processed
            if mask2 is not None:
                img2, mask2, crop_transform2 = self._crop_and_resize_to_mask_roi(img2, mask2)
                transform_img2_orig_to_processed = crop_transform2 @ transform_img2_orig_to_processed
        
        # Apply enhancement for fingervein images (instead of masking)
        if self.config.use_enhancement and modality and modality.lower() in ["fingervein", "finger_vein", "finger"]:
            img1 = enhance_fingervein_image(
                img1,
                clip_limit=self.config.enhancement_clip_limit,
                tile_grid_size=(self.config.enhancement_tile_size, self.config.enhancement_tile_size),
            )
            img2 = enhance_fingervein_image(
                img2,
                clip_limit=self.config.enhancement_clip_limit,
                tile_grid_size=(self.config.enhancement_tile_size, self.config.enhancement_tile_size),
            )
        
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
        verification_result.num_keypoints_image1 = len(keypoints1) if keypoints1 is not None else 0
        verification_result.num_keypoints_image2 = len(keypoints2) if keypoints2 is not None else 0
        verification_result.modality = modality
        
        # Override matcher name if provided (for versioned matchers like "sift-v1")
        if matcher_name is not None:
            verification_result.method_name = matcher_name

        if visualize:
            viz_result = self._create_visualization_result(
                img1_path=img1_path,
                img2_path=img2_path,
                img1_original=img1_vis_original,
                img2_original=img2_vis_original,
                img1_processed=img1,
                img2_processed=img2,
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
                transform_img1_orig_to_processed=transform_img1_orig_to_processed,
                transform_img2_orig_to_processed=transform_img2_orig_to_processed,
            )
            # Override matcher name in visualization result too
            if matcher_name is not None:
                viz_result.method_name = matcher_name
            return viz_result
        return verification_result

    def _build_visualization_processed_image(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build visualization image that reflects effective matcher inputs.

        The pipeline-level processed image already includes resize, optional ROI crop,
        and optional enhancement. This helper applies mask visualization when
        masking is enabled, mirroring matchers that either consume masked pixels
        directly or use detector masks to constrain features.
        """
        if img is None:
            return img

        out = img.copy()
        if self._visualization_uses_grayscale_input() and out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        if not self.config.use_masking or mask is None:
            return out

        mask_u8 = mask
        if mask_u8.ndim == 3:
            mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
        if mask_u8.dtype != np.uint8:
            mask_u8 = mask_u8.astype(np.uint8)
        if mask_u8.max() <= 1:
            mask_u8 = mask_u8 * 255

        if mask_u8.shape[:2] != out.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)

        return cv2.bitwise_and(out, out, mask=mask_u8)

    def _visualization_uses_grayscale_input(self) -> bool:
        """Whether matcher internally consumes grayscale image inputs."""
        return False

    @staticmethod
    def _compose_with_resize_transform(
        current_transform: np.ndarray,
        src_shape: Tuple[int, int],
        dst_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Compose current transform with a shape-based resize transform."""
        src_h, src_w = src_shape
        dst_h, dst_w = dst_shape

        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            return current_transform

        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        resize_transform = np.array(
            [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return resize_transform @ current_transform

    def _should_crop_to_mask_roi(self, modality: Optional[str]) -> bool:
        """Check whether mask-bbox ROI crop should be applied for this modality.

        Behavior is controlled via matcher config extra params:
        - roi_crop_from_mask: bool (default: False)
        - roi_crop_modalities: list[str] or comma-separated str
          (default: ["iris", "hand", "handgeometry"])
        """
        if modality is None:
            return False

        enabled = bool(self.config.extra_params.get("roi_crop_from_mask", False))
        if not enabled:
            return False

        raw_modalities = self.config.extra_params.get(
            "roi_crop_modalities",
            ["iris", "hand", "handgeometry"],
        )
        if isinstance(raw_modalities, str):
            modalities = {m.strip().lower() for m in raw_modalities.split(",") if m.strip()}
        else:
            modalities = {
                str(m).strip().lower() for m in raw_modalities
                if str(m).strip()
            }

        modality_norm = str(modality).strip().lower().replace("_", "")
        if modality_norm == "handgeometry":
            return "handgeometry" in modalities or "hand" in modalities
        if modality_norm == "hand":
            return "hand" in modalities or "handgeometry" in modalities

        return modality_norm in modalities

    def _crop_and_resize_to_mask_roi(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop image/mask to mask bbox with padding and resize back to original shape.

        Returns:
            (cropped_resized_image, cropped_resized_mask, transform_pre_to_post)
        """
        target_h, target_w = img.shape[:2]

        if mask.ndim == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        mask_bin = (mask_gray > 127).astype(np.uint8) * 255
        nonzero = cv2.findNonZero(mask_bin)
        if nonzero is None:
            return img, mask_bin, np.eye(3, dtype=np.float32)

        x, y, w, h = cv2.boundingRect(nonzero)

        # Skip tiny ROIs that are likely noise.
        min_area_frac = float(self.config.extra_params.get("roi_crop_min_area_frac", 0.01))
        if (w * h) < (min_area_frac * target_w * target_h):
            return img, mask_bin, np.eye(3, dtype=np.float32)

        pad_frac = float(self.config.extra_params.get("roi_crop_padding_frac", 0.10))
        pad_px = int(round(max(w, h) * max(0.0, pad_frac)))

        x0 = max(0, x - pad_px)
        y0 = max(0, y - pad_px)
        x1 = min(target_w, x + w + pad_px)
        y1 = min(target_h, y + h + pad_px)

        if x1 <= x0 or y1 <= y0:
            return img, mask_bin, np.eye(3, dtype=np.float32)

        cropped_img = img[y0:y1, x0:x1]
        cropped_mask = mask_bin[y0:y1, x0:x1]
        if cropped_img.size == 0 or cropped_mask.size == 0:
            return img, mask_bin, np.eye(3, dtype=np.float32)

        resized_img = cv2.resize(
            cropped_img,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
        resized_mask = cv2.resize(
            cropped_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_mask = (resized_mask > 127).astype(np.uint8) * 255

        crop_h, crop_w = cropped_img.shape[:2]
        sx = float(target_w) / float(crop_w)
        sy = float(target_h) / float(crop_h)
        transform = np.array(
            [[sx, 0.0, -sx * float(x0)], [0.0, sy, -sy * float(y0)], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        return resized_img, resized_mask, transform
    
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

    def supports_identification_template_cache(self) -> bool:
        """Whether this matcher can reuse precomputed identification templates."""
        return True

    def prepare_identification_template(self, img_path: str, modality: Optional[str] = None) -> Dict[str, Any]:
        """Prepare a reusable template for identification matching.

        The default implementation caches the fully prepared matcher input:
        loaded image, preprocessed image, and optional mask.
        Subclasses can extend this with matcher-specific cached features.
        """
        original_img = self._load_image(img_path)
        processed_img = self._preprocess_image(original_img)
        mask = None

        if self.config.use_masking and modality:
            mask = self._get_or_compute_mask(img_path, processed_img, modality)

        if self._should_crop_to_mask_roi(modality):
            if mask is not None:
                processed_img, mask, _ = self._crop_and_resize_to_mask_roi(processed_img, mask)

        if self.config.use_enhancement and modality and modality.lower() in ["fingervein", "finger_vein", "finger"]:
            processed_img = enhance_fingervein_image(
                processed_img,
                clip_limit=self.config.enhancement_clip_limit,
                tile_grid_size=(self.config.enhancement_tile_size, self.config.enhancement_tile_size),
            )

        return {
            "img_path": img_path,
            "modality": modality,
            "original": original_img,
            "processed": processed_img,
            "mask": mask,
            "cache": {},
        }

    def compare_identification_templates(
        self,
        template1: Dict[str, Any],
        template2: Dict[str, Any],
        ground_truth: Optional[bool] = None,
        matcher_name: Optional[str] = None,
    ) -> VerificationResult:
        """Compare two precomputed identification templates.

        The default implementation reuses the matcher's existing image-based
        matching logic, but skips image loading and preprocessing.
        """
        keypoints1, keypoints2, matches = self._match_impl(
            template1["processed"],
            template2["processed"],
            template1.get("mask"),
            template2.get("mask"),
        )

        homography = None
        inliers = None
        reprojection_error = None

        if len(matches) >= self.config.min_matches:
            pts1 = keypoints1[matches[:, 0]]
            pts2 = keypoints2[matches[:, 1]]
            homography, inliers = self._estimate_homography(pts1, pts2)
            if homography is not None and inliers is not None and np.any(inliers):
                reprojection_error = self._compute_reprojection_error(
                    pts1[inliers],
                    pts2[inliers],
                    homography,
                )

        verification_result = self._create_verification_result(
            img1_path=template1["img_path"],
            img2_path=template2["img_path"],
            keypoints1=keypoints1,
            keypoints2=keypoints2,
            matches=matches,
            homography=homography,
            inliers=inliers,
            reprojection_error=reprojection_error,
            ground_truth=ground_truth,
        )
        verification_result.num_keypoints_image1 = len(keypoints1) if keypoints1 is not None else 0
        verification_result.num_keypoints_image2 = len(keypoints2) if keypoints2 is not None else 0
        verification_result.modality = template1.get("modality") or template2.get("modality")

        if matcher_name is not None:
            verification_result.method_name = matcher_name

        return verification_result
    
    def _get_or_compute_mask(
        self,
        img_path: str,
        img: np.ndarray,
        modality: str,
    ) -> Optional[np.ndarray]:
        """
        Load precomputed mask from _masks folder structure.
        
        Expects masks to be pre-computed at:
          PUBLIC_DATASET_ROOT/_masks/{Modality}/{dataset_path}/{image_stem}_mask.png
        
        Example:
          Image: PublicDataset/Iris/001-CASIA/S001.jpg
          Mask:  PublicDataset/_masks/Iris/001-CASIA/S001_mask.png
        
        IMPORTANT: Must be called AFTER image preprocessing so mask dimensions
        match the preprocessed image size.
        
        Args:
            img_path: Path to image (absolute or relative to PUBLIC_DATASET_ROOT)
            img: Image array (AFTER preprocessing/resizing)
            modality: Modality type ('iris', 'face', 'hand', 'handGeometry', 'fingervein')
            
        Returns:
            Binary mask matching image dimensions, or raises error if not found
            
        Raises:
            FileNotFoundError: If mask not found (run precompute_masks.py first)
        """
        if modality in ["fingervein", "finger_vein", "finger"]:
            # Fingervein doesn't need masking (ROI already extracted)
            return None
        
        # Resolve image path to absolute
        img_abs_path = Path(img_path)
        if not img_abs_path.is_absolute():
            img_abs_path = Path(PUBLIC_DATASET_ROOT) / img_path
        
        # Extract relative path from modality folder
        # E.g., PublicDataset/Iris/001-CASIA/S001.jpg -> 001-CASIA/S001.jpg
        try:
            # Normalize modality name to match possible folder names
            # Map lowercase/variant names to actual folder names in PublicDataset
            modality_variants = {
                'hand': ['HandGeometry', 'Hand'],
                'handgeometry': ['HandGeometry', 'Hand'],
                'iris': ['Iris'],
                'face': ['Face'],
            }
            
            # Try to find the modality folder in the path
            parts = img_abs_path.parts
            modality_idx = None
            modality_folder = None
            
            # Get list of possible folder names for this modality
            possible_names = modality_variants.get(modality.lower(), [modality.capitalize()])
            
            # Search for any of these folder names in the path
            for i, part in enumerate(parts):
                if part in possible_names:
                    modality_idx = i
                    modality_folder = part
                    break
            
            if modality_idx is None:
                raise FileNotFoundError(
                    f"Cannot find modality folder for '{modality}' in path: {img_abs_path}\n"
                    f"Expected one of: {possible_names}"
                )
            
            # Relative path from modality folder
            rel_from_modality = Path(*parts[modality_idx+1:])
        except Exception as e:
            raise FileNotFoundError(f"Error extracting relative path from {img_abs_path}: {e}")
        
        # Look for mask in _masks/{ActualFolder}/{relative_path}/{image_stem}_mask.png
        mask_path = Path(PUBLIC_DATASET_ROOT) / "_masks" / modality_folder / rel_from_modality.parent / f"{rel_from_modality.stem}_mask.png"
        
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}\n"
                f"Run: python -m bioverify.experiments.precompute_masks --dataset-root {PUBLIC_DATASET_ROOT}"
            )
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
        
        # Ensure binary
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Resize to match preprocessed image dimensions
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(
                mask,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        return mask
    
    def _estimate_homography(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate homography using RANSAC.
        
        Rejects degenerate or ill-conditioned homographies (e.g., singular matrices
        or matrices with very small condition numbers).
        
        Args:
            pts1: Nx2 array of points in first image
            pts2: Nx2 array of corresponding points in second image
            
        Returns:
            Tuple of (homography, inliers):
            - homography: 3x3 matrix or None if estimation fails or is degenerate
            - inliers: Boolean array of inlier mask or None
        """
        if len(pts1) < 4:
            return None, None
        
        H, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            self.config.ransac_thresh
        )
        
        if H is None:
            return None, None
        
        # Check for degenerate homography
        try:
            # Check determinant (singular matrix has det ≈ 0)
            det = float(np.linalg.det(H))
            if abs(det) < 1e-6:
                # Singular or near-singular matrix
                return None, None
            
            # Check condition number (measures numerical stability)
            # High condition number means slight perturbations in input cause large changes in output
            cond = float(np.linalg.cond(H))
            if cond > 1e6:  # Ill-conditioned matrix
                return None, None
        
        except Exception:
            # If any matrix operation fails, treat as degenerate
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
        
        # Check for degenerate homography (divide by zero)
        z_coords = pts1_transformed[:, 2:3]
        if np.any(np.abs(z_coords) < 1e-8):
            # Degenerate homography - return infinite error
            return float("inf")
        
        pts1_transformed = pts1_transformed[:, :2] / z_coords
        
        # Compute Euclidean distances
        errors = np.linalg.norm(pts1_transformed - pts2, axis=1)
        return float(np.mean(errors))
    
    def _compute_confidence_score(
        self,
        num_matches: int,
        num_inliers: int,
        reprojection_error: Optional[float] = None,
    ) -> float:
        """
          Compute a calibrated confidence score in [0, 1].

          This uses a gentle calibration around inlier_ratio instead of the previous
          aggressive error damping. The prior variant over-penalized reprojection
          error and reduced ROC separability for several learned matchers.

          Strategy:
          - Base signal remains inlier_ratio.
          - A light significance term penalizes tiny inlier counts.
          - A light reprojection penalty only starts after an error margin.
          - DeepDetect keeps ratio-only confidence to avoid regression.
        
        Args:
            num_matches: Number of matches found between images
            num_inliers: Number of matches that fit the homography
            reprojection_error: Mean reprojection error in pixels (None → 0)
            
        Returns:
            Confidence score in [0, 1]
        """
        # No matches = no confidence
        if num_matches == 0 or num_inliers == 0:
            return 0.0
        
        # 1. Base geometric consistency
        inlier_ratio = num_inliers / num_matches

        matcher_name = self.get_name().lower()

        # DeepDetect is strongly bimodal; ratio-only score preserves its separation.
        if matcher_name == "deepdetect":
            return float(np.clip(inlier_ratio, 0.0, 1.0))

        # 2. Light statistical significance
        significance_weight = 1.0 / (1.0 + np.exp(-(num_inliers - 3.0) / 3.0))
        scaled_significance = 0.2 * significance_weight + 0.8

        # 3. Light quality term with margin (no penalty for small errors)
        if reprojection_error is None or not np.isfinite(reprojection_error):
            quality_weight = 1.0
        else:
            effective_error = max(0.0, reprojection_error - 1.5)
            quality_weight = float(np.exp(-0.05 * effective_error))

        score = inlier_ratio * scaled_significance * quality_weight
        
        return float(np.clip(score, 0.0, 1.0))
    
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
