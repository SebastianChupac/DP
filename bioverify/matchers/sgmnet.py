"""
SGMNet matcher implementation.

Uses the SGMNet components for extractor + matcher pipelines.
"""

from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - guard for missing/broken torch installs
    torch = None

from .base import BaseMatcher, MatcherConfig
from ..results import VerificationResult
from ..utils.preprocessing import resize_image


class SGMNetMatcher(BaseMatcher):
    """SGMNet matcher wrapper.

    Loads extractor and matcher components once and runs inference per match.
    """

    def __init__(self, config: MatcherConfig):
        super().__init__(config)
        if torch is None:
            raise ImportError("torch is required for SGMNetMatcher")
        if not torch.cuda.is_available():
            raise RuntimeError("SGMNet requires CUDA for extractor and matcher components")

        params = config.extra_params
        self._color = bool(params.get("color", False))
        self._ratio_thresh = float(params.get("ratio_threshold", 0.4))
        self._max_reprojection_error = params.get("max_reprojection_error", 5.0)

        self._extractor_config = params.get("extractor", {})
        self._matcher_config = params.get("matcher", {})

        self._device = self._get_device()

        # Import SGMNet components
        try:
            sgmnet_models_path = Path(__file__).parent / "sgmnet_models"
            if str(sgmnet_models_path) not in sys.path:
                sys.path.insert(0, str(sgmnet_models_path))

            from components.load_component import load_component

            self._load_component = load_component
            self._sgmnet_models_path = sgmnet_models_path

            self._extractor = self._load_component(
                "extractor",
                self._extractor_config.get("name", "root"),
                self._prepare_extractor_config(self._extractor_config),
            )
            self._matcher = self._load_component(
                "matcher",
                self._matcher_config.get("name", "SGM"),
                self._prepare_matcher_config(self._matcher_config),
            )
        except ImportError as exc:
            raise ImportError(f"Failed to import SGMNet components: {exc}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load SGMNet model: {exc}")

    def get_name(self) -> str:
        extractor_name = self._extractor_config.get("name", "root")
        matcher_name = self._matcher_config.get("name", "SGM")
        return f"SGMNet_{extractor_name}_{matcher_name}"

    def _visualization_uses_grayscale_input(self) -> bool:
        return not bool(self._color)

    def _prepare_extractor_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = dict(config)
        if config.get("name") == "sp":
            model_path = config.get("model_path", "SGMNet/weights/sp/superpoint_v1.pth")
            config["model_path"] = self._resolve_model_path(model_path)
        return config

    def _prepare_matcher_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = dict(config)
        model_dir = config.get("model_dir")
        if model_dir:
            config["model_dir"] = self._resolve_model_path(model_dir)
        return config

    def _resolve_model_path(self, path_value: str) -> str:
        if not path_value:
            return path_value
        path = Path(path_value)
        if path.is_absolute() and path.exists():
            return str(path)
        if path_value.startswith("SGMNet/"):
            rel = path_value.split("SGMNet/", 1)[1]
            return str(self._sgmnet_models_path / rel)
        candidate = self._sgmnet_models_path / path_value
        if candidate.exists():
            return str(candidate)
        return path_value

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image based on config while preserving original behavior."""
        if self.config.resize_width and self.config.resize_height:
            keep_aspect = bool(self.config.extra_params.get("resize_keep_aspect", False))
            img = resize_image(
                img,
                target_size=(int(self.config.resize_width), int(self.config.resize_height)),
                keep_aspect=keep_aspect,
            )
        return img

    def _match_impl(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: Optional[np.ndarray],
        mask2: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run SGMNet matching on a pair of images."""
        if not self._color:
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if mask1 is not None:
            mask1 = self._prepare_mask(mask1)
            img1 = cv2.bitwise_and(img1, img1, mask=mask1)
        if mask2 is not None:
            mask2 = self._prepare_mask(mask2)
            img2 = cv2.bitwise_and(img2, img2, mask=mask2)

        size1 = np.flip(np.asarray(img1.shape[:2]))
        size2 = np.flip(np.asarray(img2.shape[:2]))

        kpt1, desc1 = self._extractor.run(img1)
        kpt2, desc2 = self._extractor.run(img2)

        data = {
            "x1": kpt1,
            "x2": kpt2,
            "desc1": desc1,
            "desc2": desc2,
            "size1": size1,
            "size2": size2,
        }

        corr1, corr2, index1, index2 = self._matcher.run(data)
        if corr1 is None or corr2 is None or len(corr1) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=int),
            )

        keypoints1 = kpt1[:, :2] if kpt1 is not None and len(kpt1) > 0 else np.empty((0, 2), dtype=np.float32)
        keypoints2 = kpt2[:, :2] if kpt2 is not None and len(kpt2) > 0 else np.empty((0, 2), dtype=np.float32)

        index1 = np.asarray(index1, dtype=int)
        index2 = np.asarray(index2, dtype=int)
        if index1.size == 0 or index2.size == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.stack([index1, index2], axis=1)

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
                "extractor": self._extractor_config.get("name", "root"),
                "matcher": self._matcher_config.get("name", "SGM"),
                "ratio_threshold": self._ratio_thresh,
                "max_reprojection_error": self._max_reprojection_error,
            },
        )

    def _get_matcher_params(self) -> dict:
        return {
            "color": self._color,
            "extractor": self._extractor_config,
            "matcher": self._matcher_config,
            "ratio_threshold": self._ratio_thresh,
            "max_reprojection_error": self._max_reprojection_error,
        }

    @staticmethod
    def _prepare_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        return mask
