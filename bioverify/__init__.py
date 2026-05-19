# __init__.py - Initialization of the BioVerify package.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026

"""
BioVerify - Biometric Verification Experimentation Framework

A unified framework for evaluating and comparing homography-based methods across multiple datasets and modalities on biometric data.
"""

__version__ = "0.1.0"
__author__ = "Sebastian Chupac"

from .results import VerificationResult, VisualizationResult, ImageData, Keypoint, Match, ImageType

__all__ = [
    "VerificationResult",
    "VisualizationResult",
    "ImageData", 
    "Keypoint",
    "Match",
    "ImageType",
]
