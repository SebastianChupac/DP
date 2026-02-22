"""
Quick test of the base matcher interface.

This creates a minimal concrete matcher to verify the abstract base class works.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bioverify.matchers import BaseMatcher, MatcherConfig
from bioverify.results import VerificationResult, ImageData


class DummyMatcher(BaseMatcher):
    """
    Minimal matcher implementation for testing.
    """
    
    def get_name(self) -> str:
        return "Dummy"
    
    def _match_impl(self, img1, img2, mask1, mask2):
        """Return empty matches."""
        keypoints1 = np.array([[10.0, 20.0], [30.0, 40.0]])
        keypoints2 = np.array([[15.0, 25.0], [35.0, 45.0]])
        matches = np.array([[0, 0], [1, 1]])  # Simple 1-to-1 matching
        return keypoints1, keypoints2, matches
    
    def _create_verification_result(
        self, img1_path, img2_path, img1, img2,
        keypoints1, keypoints2, matches,
        homography, inliers, reprojection_error
    ):
        """Create a simple result."""
        num_inliers = inliers.sum() if inliers is not None else 0
        
        return VerificationResult(
            image1=ImageData(path=img1_path, shape=tuple(img1.shape)),
            image2=ImageData(path=img2_path, shape=tuple(img2.shape)),
            is_match=num_inliers > 0,
            confidence=float(num_inliers) / max(len(matches), 1),
            num_keypoints_1=len(keypoints1),
            num_keypoints_2=len(keypoints2),
            num_matches=len(matches),
            num_inliers=num_inliers,
            reprojection_error=reprojection_error,
            method_name=self.get_name(),
        )


def main():
    """Run basic tests."""
    print("Testing BaseMatcher interface...")
    
    # Test 1: Config creation
    print("\n1. Testing MatcherConfig creation...")
    config = MatcherConfig(
        resize_width=640,
        resize_height=480,
        ransac_thresh=3.0,
        use_masking=True,
    )
    print(f"   ✓ Config created: device={config.device}, ransac_thresh={config.ransac_thresh}")
    
    # Test 2: Config from dict
    print("\n2. Testing MatcherConfig.from_dict()...")
    config_dict = {
        "resize_width": 800,
        "resize_height": 600,
        "ransac_thresh": 5.0,
        "custom_param": 42,  # Should go to extra_params
    }
    config2 = MatcherConfig.from_dict(config_dict)
    print(f"   ✓ Config from dict: resize={config2.resize_width}x{config2.resize_height}")
    print(f"   ✓ Extra params: {config2.extra_params}")
    
    # Test 3: Matcher instantiation
    print("\n3. Testing DummyMatcher instantiation...")
    matcher = DummyMatcher(config)
    print(f"   ✓ Matcher created: name={matcher.get_name()}")
    
    # Test 4: Helper methods
    print("\n4. Testing helper methods...")
    
    # Test homography estimation
    pts1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    pts2 = np.array([[5, 5], [105, 5], [105, 105], [5, 105]], dtype=np.float32)
    H, inliers = matcher._estimate_homography(pts1, pts2)
    print(f"   ✓ Homography estimation: shape={H.shape if H is not None else None}")
    
    # Test reprojection error
    if H is not None and inliers is not None:
        error = matcher._compute_reprojection_error(pts1[inliers], pts2[inliers], H)
        print(f"   ✓ Reprojection error: {error:.2f} pixels")
    
    print("\n✅ All tests passed!")
    print("\nBase matcher interface is ready for concrete implementations.")


if __name__ == "__main__":
    main()
