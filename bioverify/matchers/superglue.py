"""
SuperPoint + SuperGlue matcher implementation.

Uses the SuperGlue model code stored in matchers/superglue_models.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - guard for missing/broken torch installs
    torch = None

from .base import BaseMatcher, MatcherConfig
from ..results import VerificationResult
from .superglue_models.matching import Matching


class SuperGlueMatcher(BaseMatcher):
    """SuperPoint + SuperGlue matcher wrapper.

    Loads the SuperGlue model once in __init__ and runs inference per match.
    """

    def __init__(self, config: MatcherConfig):
        super().__init__(config)
        if torch is None:
            raise ImportError("torch is required for SuperGlueMatcher")

        params = config.extra_params
        self._weights = params.get("weights", "indoor")
        self._nms_radius = int(params.get("nms_radius", 4))
        self._keypoint_threshold = float(params.get("keypoint_threshold", 0.005))
        self._max_keypoints = int(params.get("max_keypoints", 1024))
        self._sinkhorn_iterations = int(params.get("sinkhorn_iterations", 20))
        self._match_threshold = float(params.get("match_threshold", 0.2))
        self._ratio_thresh = float(params.get("ratio_threshold", 0.45))

        self._device = self._get_device()
        model_config = {
            "superpoint": {
                "nms_radius": self._nms_radius,
                "keypoint_threshold": self._keypoint_threshold,
                "max_keypoints": self._max_keypoints,
            },
            "superglue": {
                "weights": self._weights,
                "sinkhorn_iterations": self._sinkhorn_iterations,
                "match_threshold": self._match_threshold,
            },
        }
        self._matching = Matching(model_config).eval().to(self._device)

    def get_name(self) -> str:
        return "SuperGlue"

    def _match_impl(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SuperGlue matching on a pair of images."""
        gray1 = self._to_grayscale(img1)
        gray2 = self._to_grayscale(img2)

        if mask1 is not None:
            mask1 = self._prepare_mask(mask1)
            gray1 = cv2.bitwise_and(gray1, gray1, mask=mask1)
        if mask2 is not None:
            mask2 = self._prepare_mask(mask2)
            gray2 = cv2.bitwise_and(gray2, gray2, mask=mask2)

        img1_tensor = self._to_tensor(gray1)
        img2_tensor = self._to_tensor(gray2)

        with torch.no_grad():
            pred = self._matching({"image0": img1_tensor, "image1": img2_tensor})

        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        keypoints1 = pred.get("keypoints0", np.empty((0, 2), dtype=np.float32))
        keypoints2 = pred.get("keypoints1", np.empty((0, 2), dtype=np.float32))
        matches0 = pred.get("matches0", np.empty((0,), dtype=np.int32))

        if keypoints1.size == 0 or keypoints2.size == 0 or matches0.size == 0:
            return keypoints1, keypoints2, np.empty((0, 2), dtype=int)

        valid = matches0 > -1
        idx0 = np.where(valid)[0]
        if idx0.size == 0:
            return keypoints1, keypoints2, np.empty((0, 2), dtype=int)

        idx1 = matches0[valid].astype(int)
        matches = np.stack([idx0, idx1], axis=1)
        return keypoints1.astype(np.float32), keypoints2.astype(np.float32), matches

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
        inlier_mask = inliers.astype(bool) if inliers is not None else None
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        num_matches = len(matches) if matches is not None else 0
        inlier_ratio = num_inliers / max(1, num_matches)

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
                "weights": self._weights,
                "nms_radius": self._nms_radius,
                "keypoint_threshold": self._keypoint_threshold,
                "max_keypoints": self._max_keypoints,
                "sinkhorn_iterations": self._sinkhorn_iterations,
                "match_threshold": self._match_threshold,
                "ratio_threshold": self._ratio_thresh,
            },
        )

    def _get_matcher_params(self) -> dict:
        return {
            "weights": self._weights,
            "nms_radius": self._nms_radius,
            "keypoint_threshold": self._keypoint_threshold,
            "max_keypoints": self._max_keypoints,
            "sinkhorn_iterations": self._sinkhorn_iterations,
            "match_threshold": self._match_threshold,
            "ratio_threshold": self._ratio_thresh,
        }

    def _to_tensor(self, gray: np.ndarray):
        """Convert grayscale image to torch tensor [1,1,H,W] on device."""
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        return tensor[None, None].to(self._device)

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _prepare_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        return mask
