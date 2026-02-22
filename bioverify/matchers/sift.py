"""
SIFT matcher implementation.

Ported from SIFT/sift-orb.py into the unified matcher framework.
Visualization logic is intentionally omitted (visualization handled by VisualizationResult).
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseMatcher, MatcherConfig
from ..results import VerificationResult


class SIFTMatcher(BaseMatcher):
    """SIFT-based matcher using FLANN and Lowe's ratio test.
    
    Stateless matcher with no instance state beyond configuration.
    Each match() call is independent.
    """

    def __init__(self, config: MatcherConfig):
        super().__init__(config)
        params = config.extra_params
        self._sift = cv2.SIFT_create(
            nfeatures=int(params.get("nfeatures", 0)),
            nOctaveLayers=int(params.get("n_octave_layers", 3)),
            contrastThreshold=float(params.get("contrast_threshold", 0.04)),
            edgeThreshold=float(params.get("edge_threshold", 10.0)),
            sigma=float(params.get("sigma", 1.6)),
        )
        self._ratio_thresh = float(params.get("ratio_threshold", 0.45))
        self._lowe_ratio = float(params.get("lowe_ratio", 0.75))
        self._flann_trees = int(params.get("flann_trees", 5))
        self._flann_checks = int(params.get("flann_checks", 50))

    def get_name(self) -> str:
        return "SIFT"

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image based on config (keep color for mask computation)."""
        if self.config.resize_width and self.config.resize_height:
            keep_aspect = bool(
                self.config.extra_params.get("resize_keep_aspect", False)
            )
            if keep_aspect:
                img = self._resize_keep_aspect(
                    img, 
                    (self.config.resize_width, self.config.resize_height)
                )
            else:
                img = cv2.resize(
                    img,
                    (self.config.resize_width, self.config.resize_height),
                    interpolation=cv2.INTER_AREA,
                )
        return img

    def _match_impl(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stateless SIFT matching."""
        # Convert to grayscale
        gray1 = self._to_grayscale(img1)
        gray2 = self._to_grayscale(img2)

        # Prepare masks
        mask1 = self._prepare_mask(mask1)
        mask2 = self._prepare_mask(mask2)

        # Detect and compute
        kpts1, des1 = self._sift.detectAndCompute(gray1, mask1)
        kpts2, des2 = self._sift.detectAndCompute(gray2, mask2)

        # Convert keypoints to array
        keypoints1 = self._keypoints_to_array(kpts1)
        keypoints2 = self._keypoints_to_array(kpts2)

        # Check if we have enough features
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return keypoints1, keypoints2, np.empty((0, 2), dtype=int)

        # Matching with FLANN
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

        index_params = dict(algorithm=1, trees=self._flann_trees)
        search_params = dict(checks=self._flann_checks)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = matcher.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) != 2:
                continue
            m, n = match_pair
            if m.distance < self._lowe_ratio * n.distance:
                good_matches.append(m)

        if not good_matches:
            return keypoints1, keypoints2, np.empty((0, 2), dtype=int)

        # Convert to match indices array
        matches = np.array(
            [[m.queryIdx, m.trainIdx] for m in good_matches],
            dtype=int,
        )
        return keypoints1, keypoints2, matches

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
        """Create lightweight verification result for experiment tracking."""
        # Compute metrics
        inlier_mask = inliers.astype(bool) if inliers is not None else None
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        num_matches = len(matches) if matches is not None else 0
        inlier_ratio = num_inliers / max(1, num_matches)

        # Make prediction based on inlier ratio
        is_match = inlier_ratio >= self._ratio_thresh if inlier_mask is not None else False
        confidence = inlier_ratio

        return VerificationResult(
            method_name=self.get_name(),
            is_same_person_pred=is_match,
            verification_confidence=confidence,
            ground_truth=ground_truth,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error=reprojection_error,
            homography_confidence=confidence if homography is not None else 0.0,
            matcher_params=self._get_matcher_params(),
            metadata={
                "lowe_ratio": self._lowe_ratio,
                "ratio_threshold": self._ratio_thresh,
            },
        )

    def _get_matcher_params(self) -> dict:
        """Return SIFT-specific matcher parameters."""
        return {
            "nfeatures": self.config.extra_params.get("nfeatures", 0),
            "n_octave_layers": self.config.extra_params.get("n_octave_layers", 3),
            "contrast_threshold": self.config.extra_params.get("contrast_threshold", 0.04),
            "edge_threshold": self.config.extra_params.get("edge_threshold", 10.0),
            "sigma": self.config.extra_params.get("sigma", 1.6),
            "lowe_ratio": self._lowe_ratio,
            "ratio_threshold": self._ratio_thresh,
            "flann_trees": self._flann_trees,
            "flann_checks": self._flann_checks,
        }

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _prepare_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Prepare mask to uint8 format as expected by SIFT."""
        if mask is None:
            return None
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        return mask

    @staticmethod
    def _keypoints_to_array(kpts) -> np.ndarray:
        """Convert cv2.KeyPoint list to Nx2 array of (x, y) coordinates."""
        if not kpts:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([kp.pt for kp in kpts], dtype=np.float32)

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

