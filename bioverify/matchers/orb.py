"""
ORB matcher implementation.

Ported from SIFT/sift-orb.py into the unified matcher framework.
Supports both BruteForce and FLANN-LSH matching strategies.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseMatcher, MatcherConfig
from ..results import VerificationResult


class ORBMatcher(BaseMatcher):
    """ORB-based matcher with configurable matching strategy.
    
    Supports two matching strategies:
    - BruteForce: With optional cross-check validation
    - FLANN-LSH: For faster matching with binary descriptors
    
    Stateless matcher with no instance state beyond configuration.
    Each match() call is independent.
    """

    def __init__(self, config: MatcherConfig):
        super().__init__(config)
        params = config.extra_params
        
        # ORB detector parameters
        self._orb = cv2.ORB_create(
            nfeatures=int(params.get("nfeatures", 1000)),
        )
        
        # Matching parameters
        self._ratio_thresh = float(params.get("ratio_threshold", 0.45))
        self._lowe_ratio = float(params.get("lowe_ratio", 0.75))
        self._matcher_type = params.get("matcher_type", "BF").upper()  # "BF" or "FLANN"
        self._use_cross_check = bool(params.get("use_cross_check", False))
        
        # FLANN-LSH parameters (used when matcher_type="FLANN")
        self._flann_table_number = int(params.get("flann_lsh_table_number", 6))
        self._flann_key_size = int(params.get("flann_lsh_key_size", 12))
        self._flann_multi_probe = int(params.get("flann_lsh_multi_probe_level", 1))
        self._flann_checks = int(params.get("flann_checks", 50))

    def get_name(self) -> str:
        return "ORB"

    def _match_impl(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stateless ORB matching."""
        # Convert to grayscale
        gray1 = self._to_grayscale(img1)
        gray2 = self._to_grayscale(img2)

        # Prepare masks
        mask1 = self._prepare_mask(mask1)
        mask2 = self._prepare_mask(mask2)

        # Detect and compute
        kpts1, des1 = self._orb.detectAndCompute(gray1, mask1)
        kpts2, des2 = self._orb.detectAndCompute(gray2, mask2)

        # Convert keypoints to array
        keypoints1 = self._keypoints_to_array(kpts1)
        keypoints2 = self._keypoints_to_array(kpts2)

        # Check if we have enough features
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return keypoints1, keypoints2, np.empty((0, 2), dtype=int)

        # Match based on configured strategy
        if self._matcher_type == "FLANN":
            good_matches = self._match_flann(des1, des2)
        else:  # BruteForce
            good_matches = self._match_bruteforce(des1, des2)

        if not good_matches:
            return keypoints1, keypoints2, np.empty((0, 2), dtype=int)

        # Convert to match indices array
        matches = np.array(
            [[m.queryIdx, m.trainIdx] for m in good_matches],
            dtype=int,
        )
        return keypoints1, keypoints2, matches

    def _match_bruteforce(self, des1: np.ndarray, des2: np.ndarray) -> list:
        """Match using BruteForce matcher with optional cross-check."""
        if self._use_cross_check:
            # Cross-check: only consistent matches in both directions
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            # Convert to list for consistency
            return matches
        else:
            # Standard BF with ratio test
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            raw_matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in raw_matches:
                if len(match_pair) != 2:
                    continue
                m, n = match_pair
                if m.distance < self._lowe_ratio * n.distance:
                    good_matches.append(m)
            return good_matches

    def _match_flann(self, des1: np.ndarray, des2: np.ndarray) -> list:
        """Match using FLANN-LSH for binary descriptors."""
        # FLANN with LSH index for binary descriptors
        index_params = dict(
            algorithm=6,  # FLANN_INDEX_LSH
            table_number=self._flann_table_number,
            key_size=self._flann_key_size,
            multi_probe_level=self._flann_multi_probe,
        )
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
        return good_matches

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
        """Create lightweight verification result for experiment tracking.
        
        Uses improved confidence scoring that accounts for geometric consistency,
        statistical significance, and measurement quality.
        """
        # Compute metrics
        inlier_mask = inliers.astype(bool) if inliers is not None else None
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        num_matches = len(matches) if matches is not None else 0
        inlier_ratio = num_inliers / max(1, num_matches)

        # Make prediction based on inlier ratio
        #is_match = inlier_ratio >= self._ratio_thresh if inlier_mask is not None else False
        confidence = self._compute_confidence_score(num_matches, num_inliers, reprojection_error)
                # rewrite confidence with inlier ratio for this experiment
        #confidence = inlier_ratio
        is_match = confidence >= self._ratio_thresh

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
                "matcher_type": self._matcher_type,
                "use_cross_check": self._use_cross_check,
                "lowe_ratio": self._lowe_ratio,
                "ratio_threshold": self._ratio_thresh,
            },
        )

    def _get_matcher_params(self) -> dict:
        """Return ORB-specific matcher parameters."""
        params = {
            "nfeatures": self.config.extra_params.get("nfeatures", 1000),
            "matcher_type": self._matcher_type,
            "lowe_ratio": self._lowe_ratio,
            "ratio_threshold": self._ratio_thresh,
        }
        
        if self._matcher_type == "BF":
            params["use_cross_check"] = self._use_cross_check
        else:  # FLANN
            params["flann_lsh_table_number"] = self._flann_table_number
            params["flann_lsh_key_size"] = self._flann_key_size
            params["flann_lsh_multi_probe_level"] = self._flann_multi_probe
            params["flann_checks"] = self._flann_checks
        
        return params

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _prepare_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Prepare mask to uint8 format as expected by ORB."""
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
