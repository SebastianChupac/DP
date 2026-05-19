# sift.py - SIFT detector and matching implementation for biometric images.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026

"""
SIFT matcher implementation.
"""


from typing import Optional, Tuple, Dict

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

    def get_cache_type(self) -> str:
        """SIFT caches full features (keypoints + descriptors)."""
        return "full_features"

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image based on config."""
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

    def _visualization_uses_grayscale_input(self) -> bool:
        return True

    def _extract_features(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract SIFT keypoints and descriptors from a prepared image."""
        gray = self._to_grayscale(img)
        mask = self._prepare_mask(mask)
        kpts, des = self._sift.detectAndCompute(gray, mask)
        keypoints = self._keypoints_to_array(kpts)
        return keypoints, des

    def prepare_identification_template(self, img_path: str, modality: Optional[str] = None) -> dict:
        template = super().prepare_identification_template(img_path, modality)
        keypoints, descriptors = self._extract_features(template["processed"], template.get("mask"))
        template["cache"] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
        }
        return template

    def _match_impl(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
        timings_ms: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stateless SIFT matching."""
        with self._profile_stage(timings_ms, "feature_extraction_ms"):
            kpts1, des1 = self._extract_features(img1, mask1)
            kpts2, des2 = self._extract_features(img2, mask2)

        # Convert keypoints to array - already done in _extract_features, 
        #keypoints1 = self._keypoints_to_array(kpts1)
        #keypoints2 = self._keypoints_to_array(kpts2)

        keypoints1 = kpts1
        keypoints2 = kpts2

        # Check if we have enough features
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return keypoints1, keypoints2, np.empty((0, 2), dtype=int)

        # Matching with FLANN
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

        index_params = dict(algorithm=1, trees=self._flann_trees)
        search_params = dict(checks=self._flann_checks)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        with self._profile_stage(timings_ms, "matching_ms"):
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

    def compare_identification_templates(
        self,
        template1: dict,
        template2: dict,
        ground_truth: Optional[bool] = None,
        matcher_name: Optional[str] = None,
    ) -> VerificationResult:
        cache1 = template1.get("cache", {})
        cache2 = template2.get("cache", {})
        des1 = cache1.get("descriptors")
        des2 = cache2.get("descriptors")
        keypoints1 = cache1.get("keypoints")
        keypoints2 = cache2.get("keypoints")

        if des1 is None or des2 is None or keypoints1 is None or keypoints2 is None:
            return super().compare_identification_templates(template1, template2, ground_truth, matcher_name)

        if len(des1) < 2 or len(des2) < 2:
            matches = np.empty((0, 2), dtype=int)
        else:
            des1f = des1.astype(np.float32)
            des2f = des2.astype(np.float32)
            index_params = dict(algorithm=1, trees=self._flann_trees)
            search_params = dict(checks=self._flann_checks)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            raw_matches = matcher.knnMatch(des1f, des2f, k=2)

            good_matches = []
            for match_pair in raw_matches:
                if len(match_pair) != 2:
                    continue
                m, n = match_pair
                if m.distance < self._lowe_ratio * n.distance:
                    good_matches.append(m)

            matches = np.array([[m.queryIdx, m.trainIdx] for m in good_matches], dtype=int) if good_matches else np.empty((0, 2), dtype=int)

        homography = None
        inliers = None
        reprojection_error = None
        if len(matches) >= self.config.min_matches:
            pts1 = keypoints1[matches[:, 0]]
            pts2 = keypoints2[matches[:, 1]]
            homography, inliers = self._estimate_homography(pts1, pts2)
            if homography is not None and inliers is not None and np.any(inliers):
                reprojection_error = self._compute_reprojection_error(pts1[inliers], pts2[inliers], homography)

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
        if matcher_name is not None:
            verification_result.method_name = matcher_name
        return verification_result

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
        
        Uses improved confidence scoring that accounts for:
        - Geometric consistency (inlier ratio)
        - Statistical significance (sample size)
        - Measurement quality (reprojection error)
        """
        # Compute metrics
        inlier_mask = inliers.astype(bool) if inliers is not None else None
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        num_matches = len(matches) if matches is not None else 0
        inlier_ratio = num_inliers / max(1, num_matches)

        # Compute improved confidence score
        confidence = self._compute_confidence_score(num_matches, num_inliers, reprojection_error)

        # Make prediction based on ratio threshold (simplified approach)
        # User can adjust ratio_threshold parameter for sensitivity
        #is_match = inlier_ratio >= self._ratio_thresh if inlier_mask is not None else False
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

