# results.py - Data structures for storing and representing biometric verification and identification results.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2
from enum import Enum
import time

class ImageType(Enum):
    GRAYSCALE = "grayscale"
    COLOR = "color"

@dataclass
class ImageData:
    """Stores image data and metadata"""
    original: np.ndarray
    processed: Optional[np.ndarray] = None
    image_type: ImageType = ImageType.GRAYSCALE
    mask: Optional[np.ndarray] = None
    filename: Optional[str] = None
    
    def __post_init__(self):
        if self.processed is None:
            self.processed = self.original.copy()

    def __str__(self) -> str:
        mask_info = f", mask: {self.mask.shape} {self.mask.dtype}" if self.mask is not None else ""
        return f"ImageData({self.filename}, {self.image_type.value}, {self.original.shape} {self.original.dtype}{mask_info})"

@dataclass
class Keypoint:
    """Unified keypoint representation"""
    x: float
    y: float
    confidence: Optional[float] = None
    size: Optional[float] = None
    angle: Optional[float] = None
    response: Optional[float] = None
    octave: Optional[int] = None
    class_id: Optional[int] = None
    descriptor: Optional[np.ndarray] = None

    def __str__(self) -> str:
        desc_info = f", desc: {self.descriptor.shape} {self.descriptor.dtype}" if self.descriptor is not None else ""
        return f"Keypoint({self.x:.1f}, {self.y:.1f}{desc_info})"

@dataclass  
class Match:
    """Unified match representation"""
    kp1_idx: int
    kp2_idx: int
    distance: float
    confidence: Optional[float] = None
    is_inlier: Optional[bool] = None

    def __str__(self) -> str:
        inlier_info = f", inlier: {self.is_inlier}" if self.is_inlier is not None else ""
        conf_info = f", conf: {self.confidence:.3f}" if self.confidence is not None else ""
        return f"Match(kp1[{self.kp1_idx}] -> kp2[{self.kp2_idx}], dist: {self.distance:.3f}{conf_info}{inlier_info})"

@dataclass
class VerificationResult:
    """Lightweight result for experiment tracking and parameter tuning.
    
    Stores decision metrics and matcher configuration for batch processing
    and results aggregation. Does NOT store images or keypoints (to save memory
    for large experiments).
    """
    # Identification
    method_name: str
    modality: Optional[str] = None  # e.g., "face", "iris", etc.
    
    # Verification decision
    is_same_person_pred: Optional[bool] = None
    verification_confidence: float = 0.0
    ground_truth: Optional[bool] = None
    is_correct: Optional[bool] = None
    
    # Quality metrics
    num_keypoints_image1: int = 0
    num_keypoints_image2: int = 0
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: Optional[float] = None
    homography_confidence: Optional[float] = None
    
    # Matcher configuration (for parameter tuning)
    matcher_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata and tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    
    def __post_init__(self):
        if self.ground_truth is None or self.is_same_person_pred is None:
            self.is_correct = None
        else:
            self.is_correct = (self.is_same_person_pred == self.ground_truth)

    def __str__(self) -> str:
        """Concise string representation of verification result."""
        lines = [
            f"=== Verification Result: {self.method_name} ===",
            f"Prediction: {self.is_same_person_pred} (confidence: {self.verification_confidence:.3f})",
            f"Ground truth: {self.ground_truth}",
            f"Correct: {'✅ Yes' if self.is_correct else '❌ No'}",
            f"Keypoints: img1={self.num_keypoints_image1}, img2={self.num_keypoints_image2}",
            f"Matches: {self.num_matches} total, {self.num_inliers} inliers (ratio: {self.inlier_ratio:.3f})",
        ]
        if self.reprojection_error is not None:
            lines.append(f"Reprojection error: {self.reprojection_error:.2f} px")
        timings_ms = self.metadata.get("timings_ms")
        if isinstance(timings_ms, dict) and timings_ms:
            lines.append("Timings (ms):")
            for stage_name, stage_value in timings_ms.items():
                try:
                    lines.append(f"  {stage_name}: {float(stage_value):.2f}")
                except (TypeError, ValueError):
                    lines.append(f"  {stage_name}: {stage_value}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a condensed summary of verification results."""
        print(str(self))


@dataclass
class IdentificationResult:
    """Closed-set identification result for one probe sample."""

    method_name: str
    probe_record_id: str
    probe_sample_id: str
    probe_identity: str
    modality: Optional[str] = None

    ranked_identities: List[Tuple[str, float]] = field(default_factory=list)
    rank_of_true_identity: Optional[int] = None

    gallery_size: int = 0
    ranking_strategy: str = "bruteforce"           # bruteforce | cascade
    samples_per_gallery: str = "single"            # single | multiple
    aggregation_method: Optional[str] = None

    matcher_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def is_rank_k_hit(self, k: int) -> bool:
        if self.rank_of_true_identity is None:
            return False
        return self.rank_of_true_identity <= k

    def is_rank_1_hit(self) -> bool:
        return self.is_rank_k_hit(1)


@dataclass
class VisualizationResult:
    """Rich result container for visualization and debugging.
    
    Stores complete matching artifacts including images, keypoints, descriptors,
    and matches. Used for visualizing and analyzing single image pairs.
    """
    # Identification
    method_name: str
    modality: Optional[str] = None
    
    # Input images (kept for visualization)
    image1: ImageData = None
    image2: ImageData = None
    
    # Keypoints (converted to unified format)
    keypoints1: List[Keypoint] = field(default_factory=list)
    keypoints2: List[Keypoint] = field(default_factory=list)
    
    # Matches (converted to unified format)
    matches: List[Match] = field(default_factory=list)
    
    # Homography results
    homography: Optional[np.ndarray] = None
    homography_confidence: Optional[float] = None
    inlier_mask: Optional[np.ndarray] = None
    
    # Verification decision
    is_same_person_pred: Optional[bool] = None
    verification_confidence: float = 0.0
    ground_truth: Optional[bool] = None
    is_correct: Optional[bool] = None
    
    # Quality metrics
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: Optional[float] = None
    
    # Matcher configuration
    matcher_params: Dict[str, Any] = field(default_factory=dict)
    
    # Additional method-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)

    timestamp: float = field(default_factory=lambda: time.time())
    def __post_init__(self):
        self.num_matches = len(self.matches)
        if self.ground_truth is None or self.is_same_person_pred is None:
            self.is_correct = None
        else:
            self.is_correct = (self.is_same_person_pred == self.ground_truth)
        if self.inlier_mask is not None:
            self.num_inliers = np.sum(self.inlier_mask)
            self.inlier_ratio = self.num_inliers / max(1, self.num_matches)

    def __str__(self) -> str:
        """Detailed string representation for visualization."""
        lines = []
        lines.append(f"=== Visualization Result: {self.method_name} ===")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}")
        lines.append("")
        
        # Image information
        lines.append("--- Images ---")
        lines.append(f"Modality: {self.modality or 'N/A'}")
        if self.image1:
            lines.append(f"Image type: {self.image1.image_type.value}")
            lines.append(f"Image 1: {self._format_image_data(self.image1)}")
            lines.append(f"Image 2: {self._format_image_data(self.image2)}")
        lines.append("")
        
        # Keypoints and matches
        lines.append("--- Features & Matches ---")
        lines.append(f"Keypoints 1: {len(self.keypoints1)}")
        lines.append(f"Keypoints 2: {len(self.keypoints2)}")
        lines.append(f"Total matches: {self.num_matches}")
        lines.append(f"Inlier matches: {self.num_inliers}")
        lines.append(f"Inlier ratio: {self.inlier_ratio:.3f}")
        lines.append("")
        
        # Homography results
        lines.append("--- Homography ---")
        if self.homography is not None:
            lines.append(f"Homography: {self.homography.shape} {self.homography.dtype}")
            lines.append(f"Homography confidence: {self._format_value(self.homography_confidence)}")
        else:
            lines.append("Homography: None")
        lines.append(f"Inlier mask: {self._format_array(self.inlier_mask)}")
        lines.append("")
        
        # Verification decision
        lines.append("--- Verification ---")
        lines.append(f"Same person prediction: {self._format_value(self.is_same_person_pred)}")
        lines.append(f"Verification confidence: {self.verification_confidence:.3f}")
        lines.append(f"Reprojection error: {self._format_value(self.reprojection_error)}")
        lines.append(f"Ground truth: {self._format_value(self.ground_truth)}")
        lines.append(f"Correct decision: {'✅ Yes' if self.is_correct else '❌ No'}")
        lines.append("")
        
        # Metadata
        lines.append("--- Metadata ---")
        if self.metadata:
            for key, value in self.metadata.items():
                formatted_value = self._format_metadata_value(value)
                lines.append(f"{key}: {formatted_value}")
        else:
            lines.append("No additional metadata")
        
        return "\n".join(lines)
    
    def _format_image_data(self, image_data: ImageData) -> str:
        """Format image data for display"""
        if image_data is None:
            return "None"
        
        original_shape = f"{image_data.original.shape} {image_data.original.dtype}"
        processed_shape = f"{image_data.processed.shape} {image_data.processed.dtype}" if image_data.processed is not None else "None"
        mask_info = f"mask {image_data.mask.shape} {image_data.mask.dtype}" if image_data.mask is not None else "no mask"
        
        return f"{image_data.filename or 'unnamed'} ({image_data.image_type.value}), original: {original_shape}, processed: {processed_shape}, {mask_info}"
    
    def _format_array(self, array) -> str:
        """Format numpy arrays showing shape and dtype instead of values"""
        if array is None:
            return "None"
        elif hasattr(array, 'shape') and hasattr(array, 'dtype'):
            return f"{array.shape} {array.dtype}"
        elif isinstance(array, (list, tuple)):
            return f"list[{len(array)}]"
        else:
            return str(type(array).__name__)
    
    def _format_value(self, value) -> str:
        """Format any value, handling None and numeric precision"""
        if value is None:
            return "None"
        elif isinstance(value, float):
            return f"{value:.3f}"
        elif isinstance(value, bool):
            return str(value)
        else:
            return str(value)
    
    def _format_metadata_value(self, value) -> str:
        """Format metadata values, handling arrays specially"""
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            return f"array{value.shape} {value.dtype}"
        elif isinstance(value, (list, tuple, np.ndarray)):
            if hasattr(value, 'shape'):
                return f"array{value.shape} {value.dtype}"
            else:
                return f"{type(value).__name__}[{len(value)}]"
        elif isinstance(value, dict):
            return f"dict[{len(value)} keys]"
        else:
            return self._format_value(value)

    def print_summary(self) -> None:
        """Print a condensed summary of visualization results."""
        print(f"{self.method_name} Results:")
        print(f"   Ground Truth: {self.ground_truth}")
        print(f"   Prediction: {self.is_same_person_pred} "
              f"(confidence: {self.verification_confidence:.3f})")
        print(f"   Correct decision: {'Yes' if self.is_correct else 'No'}")
        print(f"   Matches: {self.num_matches} total, {self.num_inliers} inliers "
              f"(ratio: {self.inlier_ratio:.3f})")
        if self.reprojection_error is not None:
            print(f"   Reprojection error: {self.reprojection_error:.2f} px")
        if self.homography is not None:
            print(f"   Homography: present (confidence: {self.homography_confidence:.3f})")
        else:
            print(f"   Homography: absent")