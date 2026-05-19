# loftr.py - LoFTR matcher implementation for biometric images.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026


"""
LoFTR (Local Feature Transformer) matcher implementation.

Uses Kornia's LoFTR implementation via kornia.feature (KF).
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import kornia.feature as KF
except Exception as exc:  # pragma: no cover - guard for missing/broken torch/kornia installs
    torch = None
    KF = None

from .base import BaseMatcher, MatcherConfig
from ..results import VerificationResult


class LoFTRMatcher(BaseMatcher):
    """LoFTR (Local Feature Transformer) matcher wrapper.

    Uses Kornia's implementation. Loads the model once in __init__
    and runs inference per match.
    """

    def __init__(self, config: MatcherConfig):
        super().__init__(config)
        if torch is None or KF is None:
            raise ImportError("torch and kornia are required for LoFTRMatcher")

        params = config.extra_params
        self._model_type = params.get("model_type", "indoor")  # 'indoor' or 'outdoor'
        self._confidence_threshold = float(params.get("confidence_threshold", 0.9))
        self._ratio_thresh = float(params.get("ratio_threshold", 0.45))
        self._loftr_tuning = self._extract_safe_loftr_tuning(params)
        self._loftr_config = self._build_loftr_config(self._loftr_tuning)

        self._device = self._get_device()
        self._matcher = KF.LoFTR(pretrained=self._model_type, config=self._loftr_config).to(self._device)
        self._matcher.eval()

    @staticmethod
    def _default_loftr_config() -> Dict[str, Any]:
        """Return LoFTR config defaults aligned with Kornia pretrained settings."""
        return {
            "backbone_type": "ResNetFPN",
            "resolution": (8, 2),
            "fine_window_size": 5,
            "fine_concat_coarse_feat": True,
            "resnetfpn": {
                "initial_dim": 128,
                "block_dims": [128, 196, 256],
            },
            "coarse": {
                "d_model": 256,
                "d_ffn": 256,
                "nhead": 8,
                "layer_names": [
                    "self",
                    "cross",
                    "self",
                    "cross",
                    "self",
                    "cross",
                    "self",
                    "cross",
                ],
                "attention": "linear",
                "temp_bug_fix": False,
            },
            "match_coarse": {
                "thr": 0.2,
                "border_rm": 2,
                "match_type": "dual_softmax",
                "dsmax_temperature": 0.1,
                "skh_iters": 3,
                "skh_init_bin_score": 1.0,
                "skh_prefilter": True,
                "train_coarse_percent": 0.4,
                "train_pad_num_gt_min": 200,
            },
            "fine": {
                "d_model": 128,
                "d_ffn": 128,
                "nhead": 8,
                "layer_names": ["self", "cross"],
                "attention": "linear",
            },
        }

    @staticmethod
    def _extract_safe_loftr_tuning(params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only inference-safe LoFTR tuning keys from config.

        We intentionally expose a small subset that does not alter model
        architecture, keeping compatibility with pretrained checkpoints.
        """
        raw = params.get("loftr_config", {})
        if not isinstance(raw, dict):
            return {}

        match_coarse = raw.get("match_coarse", {})
        if not isinstance(match_coarse, dict):
            return {}

        out: Dict[str, Any] = {}

        if "thr" in match_coarse:
            try:
                out["thr"] = float(match_coarse["thr"])
            except (TypeError, ValueError):
                pass

        if "border_rm" in match_coarse:
            try:
                out["border_rm"] = int(match_coarse["border_rm"])
            except (TypeError, ValueError):
                pass

        if "dsmax_temperature" in match_coarse:
            try:
                out["dsmax_temperature"] = float(match_coarse["dsmax_temperature"])
            except (TypeError, ValueError):
                pass

        return out

    @classmethod
    def _build_loftr_config(cls, tuning: Dict[str, Any]) -> Dict[str, Any]:
        """Build LoFTR init config from Kornia defaults plus safe tuning keys."""
        config = deepcopy(cls._default_loftr_config())
        if not tuning:
            # ensure resolution is a tuple (Kornia expects tuples for exact match)
            if isinstance(config.get("resolution"), list):
                config["resolution"] = tuple(config["resolution"])
            return config

        config["match_coarse"].update(tuning)
        # ensure resolution is a tuple (user input might be a list)
        if isinstance(config.get("resolution"), list):
            config["resolution"] = tuple(config["resolution"])
        return config

    def get_name(self) -> str:
        return f"LoFTR_{self._model_type}"

    def _visualization_uses_grayscale_input(self) -> bool:
        return True

    def _match_impl(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
        timings_ms: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run LoFTR matching on a pair of images."""
        gray1 = self._to_grayscale(img1)
        gray2 = self._to_grayscale(img2)

        # Apply masks if provided
        if mask1 is not None:
            mask1 = self._prepare_mask(mask1)
            gray1 = cv2.bitwise_and(gray1, gray1, mask=mask1)
        if mask2 is not None:
            mask2 = self._prepare_mask(mask2)
            gray2 = cv2.bitwise_and(gray2, gray2, mask=mask2)

        # Convert to tensors
        img1_tensor = self._to_tensor(gray1)
        img2_tensor = self._to_tensor(gray2)

        # Run LoFTR matching
        batch = {"image0": img1_tensor, "image1": img2_tensor}
        with self._profile_stage(timings_ms, "inference_ms"):
            with torch.no_grad():
                pred = self._matcher(batch)

        # Extract results
        keypoints1 = pred["keypoints0"].cpu().numpy()
        keypoints2 = pred["keypoints1"].cpu().numpy()
        confidence = pred["confidence"].cpu().numpy()

        # Filter matches by confidence threshold
        valid_mask = confidence >= self._confidence_threshold
        if valid_mask.sum() == 0:
            return (
                keypoints1.astype(np.float32),
                keypoints2.astype(np.float32),
                np.empty((0, 2), dtype=int),
            )

        # Filter keypoints by confidence
        keypoints1_filtered = keypoints1[valid_mask]
        keypoints2_filtered = keypoints2[valid_mask]

        # Create sequential match indices for filtered keypoints
        # LoFTR outputs 1:1 matched pairs, so match indices are sequential
        num_valid = keypoints1_filtered.shape[0]
        matches = np.stack([np.arange(num_valid), np.arange(num_valid)], axis=1)

        return (
            keypoints1_filtered.astype(np.float32),
            keypoints2_filtered.astype(np.float32),
            matches,
        )

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
        inlier_mask = inliers.astype(bool) if inliers is not None else None
        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        num_matches = len(matches) if matches is not None else 0
        inlier_ratio = num_inliers / max(1, num_matches)

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
                "model_type": self._model_type,
                "confidence_threshold": self._confidence_threshold,
                "ratio_threshold": self._ratio_thresh,
            },
        )

    def _get_matcher_params(self) -> dict:
        return {
            "model_type": self._model_type,
            "confidence_threshold": self._confidence_threshold,
            "ratio_threshold": self._ratio_thresh,
            "loftr_tuning": self._loftr_tuning,
        }

    def _to_tensor(self, gray: np.ndarray) -> Any:
        """Convert grayscale image to torch tensor [1,1,H,W] on device."""
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        return tensor[None, None].to(self._device)

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _prepare_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Prepare mask for cv2.bitwise_and operation."""
        if mask is None:
            return None
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        return mask
