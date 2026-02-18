"""
BioVerify - Biometric Verification Experimentation Framework

A unified framework for evaluating and comparing homography-based biometric
verification methods across multiple datasets and modalities.
"""

__version__ = "0.1.0"
__author__ = "Sebastian Chupac"

from .results import VerificationResult, ImageData, Keypoint, Match, ImageType

__all__ = [
    "VerificationResult",
    "ImageData", 
    "Keypoint",
    "Match",
    "ImageType",
]
