"""
ASpanFormer matcher implementation.

Uses the ASpanFormer model for robust feature matching.
"""

from typing import Optional, Tuple
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .base import BaseMatcher, MatcherConfig
from ..results import VerificationResult


class ASpanFormerMatcher(BaseMatcher):
    """ASpanFormer matcher wrapper.

    Uses the ASpanFormer model for dense feature matching.
    Loads the model once in __init__ and runs inference per match.
    """

    def __init__(self, config: MatcherConfig):
        super().__init__(config)
        if torch is None:
            raise ImportError("torch is required for ASpanFormerMatcher")

        params = config.extra_params
        self._model_type = params.get("model_type", "indoor")  # 'indoor' or 'outdoor'
        self._confidence_threshold = float(params.get("confidence_threshold", 0.3))
        self._ratio_thresh = float(params.get("ratio_threshold", 0.45))

        self._device = self._get_device()
        
        # Import ASpanFormer components
        try:
            # Add aspanformer_models to sys.path for config file imports
            # (config files use absolute imports like 'from config.default import ...')
            aspanformer_models_path = Path(__file__).parent / "aspanformer_models"
            if str(aspanformer_models_path) not in sys.path:
                sys.path.insert(0, str(aspanformer_models_path))
            
            from .aspanformer_models.ASpanFormer.aspanformer import ASpanFormer
            from .aspanformer_models.config.default import get_cfg_defaults
            from .aspanformer_models.utils.misc import lower_config
            
            # Load config
            config_obj = get_cfg_defaults()
            config_path = aspanformer_models_path / "config" / "aspan" / self._model_type / "aspan_test.py"
            if config_path.exists():
                config_obj.merge_from_file(str(config_path))
            _config = lower_config(config_obj)
            _config = lower_config(config_obj)
            
            # Load model
            weights_path = aspanformer_models_path / "weights" / f"{self._model_type}.ckpt"
            if not weights_path.exists():
                raise FileNotFoundError(f"ASpanFormer weights not found: {weights_path}")
            
            self._matcher = ASpanFormer(config=_config['aspan'])
            # weights_only=False is safe here as we control the model file
            state_dict = torch.load(str(weights_path), map_location='cpu', weights_only=False)['state_dict']
            self._matcher.load_state_dict(state_dict, strict=False)
            self._matcher.eval().to(self._device)
            
        except ImportError as e:
            raise ImportError(f"Failed to import ASpanFormer components: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ASpanFormer model: {e}")

    def get_name(self) -> str:
        return f"ASpanFormer_{self._model_type}"

    def _match_impl(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run ASpanFormer matching on a pair of images."""
        gray1 = self._to_grayscale(img1)
        gray2 = self._to_grayscale(img2)

        # Apply masks if provided
        if mask1 is not None:
            mask1 = self._prepare_mask(mask1)
            gray1 = cv2.bitwise_and(gray1, gray1, mask=mask1)
        if mask2 is not None:
            mask2 = self._prepare_mask(mask2)
            gray2 = cv2.bitwise_and(gray2, gray2, mask=mask2)

        # Convert to tensors (normalize to [0, 1])
        img1_tensor = self._to_tensor(gray1)
        img2_tensor = self._to_tensor(gray2)

        # Run ASpanFormer matching
        batch = {
            "image0": img1_tensor.to(self._device),
            "image1": img2_tensor.to(self._device),
        }
        
        with torch.no_grad():
            self._matcher(batch, online_resize=True)

        # Extract results
        keypoints1 = batch["mkpts0_f"].cpu().numpy()
        keypoints2 = batch["mkpts1_f"].cpu().numpy()
        confidence = batch["mconf"].cpu().numpy()

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
        # ASpanFormer outputs 1:1 matched pairs, so match indices are sequential
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
        }

    def _to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        """Convert grayscale image to torch tensor [1,1,H,W] on device."""
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        return tensor[None, None]

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
